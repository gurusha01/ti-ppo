"""Benchmark: MATH (Hendrycks) Competition Math with TI-PPO.

Tests token-importance methods on competition-level math (MATH dataset).
Unlike GSM8K (grade-school), MATH problems require algebra, geometry,
number theory, calculus, etc. Base model accuracy is ~20-30%, giving
PPO significant room to improve.

Training data: 500 examples sampled from EleutherAI/hendrycks_math (seed=42)
Test data: HuggingFaceH4/MATH-500 (split='test', 500 examples)

Methods tested:
0. PPO baseline (uniform weighting)
1. AITI-Advantage (linear decay, decay_steps=100)
2. AITI-Entropy
3. MOAI-Advantage mono (ema=0.80)
4. MOAI-Advantage mono (ema=0.90)
5. MOAI-Entropy mono (ema=0.80)
6. MOAI-Entropy mono (ema=0.90)
7. Advantage (fixed, no adaptive decay)

Usage (parallel across GPUs):
    python scripts/benchmark_math.py --method 0 --gpu 0 &
    python scripts/benchmark_math.py --method 1 --gpu 1 &
    ...

Or sequential on one GPU:
    python scripts/benchmark_math.py --method all --gpu 4
"""

import json, os, sys, time, math, re, random
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from peft import LoraConfig, get_peft_model

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ti_ppo import TIPPOConfig, TIPPOTrainer, CausalLMWithValueHead

# -----------------------------------------------------------------------
# Method definitions
# -----------------------------------------------------------------------

METHODS = [
    # (method, label, extra_config)
    # NOTE: Advantage-based importance is meaningless in PPO with scalar rewards
    # because GAE makes advantage a purely positional decay (0.95^(T-t)),
    # not a content signal. Only entropy is content-based.

    # GPU 0: baseline
    ("uniform", "PPO baseline", {}),
    # GPU 1: entropy (fixed intensity, no decay)
    ("entropy", "Entropy (fixed)", {}),
    # GPU 2: AITI-Entropy (linear decay 200 steps)
    ("aiti_entropy", "AITI-Entropy (decay=200)", {
        "aiti_epsilon_max": 1.0, "aiti_decay_steps": 200,
        "aiti_power": 1.0, "aiti_min_epsilon": 0.0}),
    # GPU 3: AITI-Entropy (linear decay 100 steps, faster)
    ("aiti_entropy", "AITI-Entropy (decay=100)", {
        "aiti_epsilon_max": 1.0, "aiti_decay_steps": 100,
        "aiti_power": 1.0, "aiti_min_epsilon": 0.0}),
    # GPU 4: MOAI-Entropy mono (ema=0.80)
    ("moai_entropy_mono", "MOAI-Ent mono 0.80", {
        "moai_ema_decay": 0.80, "moai_warmup_steps": 5}),
    # GPU 5: MOAI-Entropy mono (ema=0.90)
    ("moai_entropy_mono", "MOAI-Ent mono 0.90", {
        "moai_ema_decay": 0.90, "moai_warmup_steps": 5}),
    # GPU 6: MOAI-Entropy mono (ema=0.95, slower adaptation)
    ("moai_entropy_mono", "MOAI-Ent mono 0.95", {
        "moai_ema_decay": 0.95, "moai_warmup_steps": 5}),
    # GPU 7: AITI-Entropy (no decay, residual min=0.3)
    ("aiti_entropy", "AITI-Entropy (residual=0.3)", {
        "aiti_epsilon_max": 1.0, "aiti_decay_steps": 200,
        "aiti_power": 1.0, "aiti_min_epsilon": 0.3}),
]

# -----------------------------------------------------------------------
# MATH dataset loading
# -----------------------------------------------------------------------

MATH_SUBJECTS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]


def load_math_train(n_samples=500, seed=42):
    """Load MATH training data from EleutherAI/hendrycks_math.

    Loads all 7 subjects, concatenates, then samples n_samples with the
    given seed. Returns list of dicts with 'problem', 'solution', 'answer',
    'subject', 'level'.
    """
    from datasets import load_dataset

    all_examples = []
    for subject in MATH_SUBJECTS:
        try:
            ds = load_dataset("EleutherAI/hendrycks_math", subject,
                              split="train", trust_remote_code=True)
            for row in ds:
                answer = extract_boxed_answer(row["solution"])
                if answer is not None:
                    all_examples.append({
                        "problem": row["problem"],
                        "solution": row["solution"],
                        "answer": answer,
                        "subject": subject,
                        "level": row.get("level", "unknown"),
                    })
        except Exception as e:
            print(f"  Warning: failed to load subject '{subject}': {e}")

    print(f"Loaded {len(all_examples)} total MATH train examples (all subjects)")

    # Sample n_samples with fixed seed for reproducibility
    rng = random.Random(seed)
    if len(all_examples) > n_samples:
        examples = rng.sample(all_examples, n_samples)
    else:
        examples = all_examples
    print(f"Sampled {len(examples)} training examples (seed={seed})")

    # Print subject distribution
    from collections import Counter
    subject_counts = Counter(ex["subject"] for ex in examples)
    for subj, cnt in sorted(subject_counts.items()):
        print(f"  {subj}: {cnt}")

    return examples


def load_math_test():
    """Load MATH-500 test set from HuggingFaceH4/MATH-500.

    Returns list of dicts with 'problem', 'answer', 'subject', 'level'.
    The dataset has an 'answer' field already extracted.
    """
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceH4/MATH-500", split="test", trust_remote_code=True)
    examples = []
    for row in ds:
        examples.append({
            "problem": row["problem"],
            "answer": row["answer"],
            "subject": row.get("subject", row.get("type", "unknown")),
            "level": row.get("level", "unknown"),
        })

    print(f"Loaded {len(examples)} MATH-500 test examples")

    from collections import Counter
    subject_counts = Counter(ex["subject"] for ex in examples)
    for subj, cnt in sorted(subject_counts.items()):
        print(f"  {subj}: {cnt}")

    return examples


# -----------------------------------------------------------------------
# Answer extraction and normalization
# -----------------------------------------------------------------------

def extract_boxed_answer(text):
    """Extract the content of the last \\boxed{...} in text.

    Handles nested braces, e.g. \\boxed{\\frac{3}{4}}.
    """
    # Find all \boxed{ positions
    idx = text.rfind("\\boxed{")
    if idx == -1:
        idx = text.rfind("\\boxed ")
        if idx == -1:
            return None
        # Handle \boxed X (no braces, single token)
        rest = text[idx + 7:].strip()
        # Take the first non-whitespace token
        match = re.match(r"(\S+)", rest)
        return match.group(1) if match else None

    # Find matching closing brace
    start = idx + 7  # position after \boxed{
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1

    if depth == 0:
        return text[start:i - 1].strip()
    return None


def normalize_answer(answer):
    """Normalize a LaTeX math answer for comparison.

    Handles common patterns:
    - Strip whitespace and dollar signs
    - Remove \\text{}, \\mathrm{}, \\textbf{}, etc.
    - Remove \\left, \\right
    - Normalize spacing
    - Convert simple \\frac{a}{b} to decimal
    """
    if answer is None:
        return None

    s = answer.strip()

    # Strip surrounding dollar signs
    s = s.strip("$")

    # Remove \text{...}, \mathrm{...}, \textbf{...}, \mbox{...}
    s = re.sub(r"\\(?:text|mathrm|textbf|mbox|operatorname)\{([^}]*)\}", r"\1", s)

    # Remove \left and \right
    s = re.sub(r"\\left\s*", "", s)
    s = re.sub(r"\\right\s*", "", s)

    # Remove \, \; \: \! (spacing commands)
    s = re.sub(r"\\[,;:!]", "", s)

    # Remove \displaystyle, \tfrac -> \frac, \dfrac -> \frac
    s = s.replace("\\displaystyle", "")
    s = s.replace("\\tfrac", "\\frac")
    s = s.replace("\\dfrac", "\\frac")

    # Normalize whitespace
    s = re.sub(r"\s+", " ", s).strip()

    # Remove trailing period
    s = s.rstrip(".")

    return s


def try_parse_float(s):
    """Try to parse a string as a float. Returns None on failure."""
    if s is None:
        return None
    # Remove commas
    s = s.replace(",", "")
    try:
        return float(s)
    except (ValueError, OverflowError):
        return None


def try_eval_fraction(s):
    """Try to evaluate a \\frac{a}{b} expression as a float.

    Only handles simple cases: \\frac{number}{number}.
    """
    if s is None:
        return None
    match = re.match(r"^\\frac\{([^{}]+)\}\{([^{}]+)\}$", s.strip())
    if match:
        num = try_parse_float(match.group(1))
        den = try_parse_float(match.group(2))
        if num is not None and den is not None and den != 0:
            return num / den
    # Also try a/b format
    match = re.match(r"^(-?\d+(?:\.\d+)?)\s*/\s*(-?\d+(?:\.\d+)?)$", s.strip())
    if match:
        num = try_parse_float(match.group(1))
        den = try_parse_float(match.group(2))
        if num is not None and den is not None and den != 0:
            return num / den
    return None


def answers_match(predicted, ground_truth):
    """Check if predicted answer matches ground truth.

    Uses multiple strategies:
    1. Exact string match (after normalization)
    2. Numeric comparison with tolerance
    3. Fraction evaluation and comparison
    """
    if predicted is None or ground_truth is None:
        return False

    norm_pred = normalize_answer(predicted)
    norm_gt = normalize_answer(ground_truth)

    if norm_pred is None or norm_gt is None:
        return False

    # Strategy 1: Exact string match
    if norm_pred == norm_gt:
        return True

    # Strategy 2: Try numeric comparison
    float_pred = try_parse_float(norm_pred)
    float_gt = try_parse_float(norm_gt)

    if float_pred is not None and float_gt is not None:
        if abs(float_pred - float_gt) < 1e-6:
            return True
        if float_gt != 0 and abs(float_pred - float_gt) / abs(float_gt) < 1e-4:
            return True

    # Strategy 3: Evaluate fractions
    frac_pred = try_eval_fraction(norm_pred)
    frac_gt = try_eval_fraction(norm_gt)

    if frac_pred is not None and frac_gt is not None:
        if abs(frac_pred - frac_gt) < 1e-6:
            return True

    # Cross-compare: fraction vs float
    if frac_pred is not None and float_gt is not None:
        if abs(frac_pred - float_gt) < 1e-6:
            return True
    if float_pred is not None and frac_gt is not None:
        if abs(float_pred - frac_gt) < 1e-6:
            return True

    # Strategy 4: Remove all whitespace and compare
    stripped_pred = re.sub(r"\s", "", norm_pred)
    stripped_gt = re.sub(r"\s", "", norm_gt)
    if stripped_pred == stripped_gt:
        return True

    return False


def extract_model_answer(response):
    """Extract the answer from the model's response.

    Looks for \\boxed{} first, then falls back to other patterns.
    """
    # Pattern 1: \boxed{...} (our prompt asks for this)
    answer = extract_boxed_answer(response)
    if answer is not None:
        return answer

    # Pattern 2: "the answer is ..."
    match = re.search(
        r"(?:the\s+(?:final\s+)?answer\s+is|answer\s*[:=])\s*[\\$]*(.*?)(?:\.|$)",
        response, re.IGNORECASE,
    )
    if match:
        ans = match.group(1).strip().strip("$").strip()
        if ans:
            return ans

    # Pattern 3: Last line with "= ..."
    match = re.search(r"=\s*([^=\n]+?)\s*$", response, re.MULTILINE)
    if match:
        ans = match.group(1).strip().strip("$").strip(".")
        if ans:
            return ans

    return None


def math_reward(response, ground_truth_answer):
    """Binary reward: +1.0 if correct, -1.0 if incorrect."""
    predicted = extract_model_answer(response)
    if predicted is None:
        return -1.0

    if answers_match(predicted, ground_truth_answer):
        return 1.0

    return -1.0


def format_prompt(problem, tokenizer=None):
    """Format a MATH problem using the model's chat template."""
    if tokenizer is not None and hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template:
        messages = [{"role": "user", "content": f"Solve this math problem. Show your work step by step, then put your final answer in \\boxed{{}}.\n\n{problem}"}]
        # enable_thinking=False for Qwen3 to avoid thinking mode
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"Q: {problem}\nA: Let's solve step by step, then give the final answer in \\boxed{{}}.\n"


# -----------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------

PRIMARY_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
FALLBACK_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def load_model_and_tokenizer(device):
    """Load model + LoRA + ValueHead and reference model.

    Returns (model, ref_model, tokenizer, model_name, lora_target_modules).
    """
    for model_name, lora_targets in [
        (PRIMARY_MODEL, ["q_proj", "v_proj"]),
        (FALLBACK_MODEL, ["c_attn", "c_proj"]),
    ]:
        try:
            print(f"Loading {model_name}...")
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = CausalLMWithValueHead.from_pretrained(
                model_name, dtype=torch.float16, device_map={"": device},
                trust_remote_code=True,
            )

            lora_cfg = LoraConfig(
                r=16, lora_alpha=32, lora_dropout=0.05,
                bias="none", task_type="CAUSAL_LM",
                target_modules=lora_targets,
            )
            model.pretrained_model = get_peft_model(model.pretrained_model, lora_cfg)

            ref_model = CausalLMWithValueHead.from_pretrained(
                model_name, dtype=torch.float16, device_map={"": device},
                trust_remote_code=True,
            )
            for p in ref_model.parameters():
                p.requires_grad = False

            print(f"Successfully loaded {model_name}")
            return model, ref_model, tokenizer, model_name, lora_targets
        except Exception as e:
            print(f"Failed to load {model_name}: {e}")
            if model_name == FALLBACK_MODEL:
                raise RuntimeError("Could not load any model") from e
            print("Falling back...")
            continue


# -----------------------------------------------------------------------
# Test evaluation
# -----------------------------------------------------------------------

def evaluate_on_test(model, tokenizer, test_data, device, n_samples=500):
    """Evaluate the model on MATH-500 test examples.

    Uses greedy decoding for deterministic evaluation.

    Returns:
        dict with accuracy, correct, total, per-subject accuracy.
    """
    print(f"\n  Evaluating on {min(n_samples, len(test_data))} test examples (greedy)...")
    if len(test_data) > n_samples:
        eval_data = random.sample(test_data, n_samples)
    else:
        eval_data = test_data

    gen_kwargs = {
        "max_new_tokens": 512,
        "do_sample": False,
        "pad_token_id": tokenizer.pad_token_id,
    }

    correct = 0
    total = 0
    from collections import Counter
    subject_correct = Counter()
    subject_total = Counter()

    model.eval()
    for i, ex in enumerate(eval_data):
        prompt_text = format_prompt(ex["problem"], tokenizer)
        tokens = tokenizer(
            prompt_text, truncation=True, max_length=256,
            return_tensors="pt",
        ).input_ids.to(device)

        with torch.no_grad():
            out = model.generate(input_ids=tokens, **gen_kwargs)
        response_tokens = out[0, tokens.shape[1]:]
        if response_tokens.shape[0] == 0:
            total += 1
            subject_total[ex["subject"]] += 1
            continue

        response_text = tokenizer.decode(response_tokens, skip_special_tokens=True)
        r = math_reward(response_text, ex["answer"])

        total += 1
        subject_total[ex["subject"]] += 1
        if r > 0:
            correct += 1
            subject_correct[ex["subject"]] += 1

        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{len(eval_data)}: {correct}/{total} = {correct/max(1,total):.4f}")

    accuracy = correct / max(1, total)
    print(f"  Test accuracy: {correct}/{total} = {accuracy:.4f}")

    # Per-subject breakdown
    subject_acc = {}
    for subj in sorted(subject_total.keys()):
        sc = subject_correct[subj]
        st = subject_total[subj]
        acc = sc / max(1, st)
        subject_acc[subj] = {"correct": sc, "total": st, "accuracy": round(acc, 4)}
        print(f"    {subj}: {sc}/{st} = {acc:.4f}")

    return {
        "accuracy": round(accuracy, 4),
        "correct": correct,
        "total": total,
        "subject_accuracy": subject_acc,
    }


# -----------------------------------------------------------------------
# Training loop for a single method
# -----------------------------------------------------------------------

def run_method(method, label, extra_config, tokenizer, math_data, test_data,
               device, model_name, lora_targets, episodes=100):
    """Run a single TI-PPO method on MATH.

    Args:
        method: importance method name
        label: human-readable label
        extra_config: dict of extra config overrides
        tokenizer: tokenizer
        math_data: list of MATH training examples (500)
        test_data: list of MATH-500 test examples
        device: torch device
        model_name: which model to load
        lora_targets: LoRA target modules
        episodes: number of training episodes

    Returns:
        (result_dict, history_dict)
    """
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"  method={method} | model={model_name} | episodes={episodes}")
    print(f"  train_pool={len(math_data)} | test_set={len(test_data)}")
    print(f"{'='*70}")

    # Load fresh model for this method
    model = CausalLMWithValueHead.from_pretrained(
        model_name, dtype=torch.float16, device_map={"": device},
        trust_remote_code=True,
    )
    lora_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM",
        target_modules=lora_targets,
    )
    model.pretrained_model = get_peft_model(model.pretrained_model, lora_cfg)

    ref_model = CausalLMWithValueHead.from_pretrained(
        model_name, dtype=torch.float16, device_map={"": device},
        trust_remote_code=True,
    )
    for p in ref_model.parameters():
        p.requires_grad = False

    # Configure TI-PPO
    config = TIPPOConfig(
        importance_method=method,
        use_triplet_loss=False,
        ppo_epochs=4,
        learning_rate=5e-5,
        lora_r=16,
        lora_alpha=32,
        max_new_tokens=512,
        clip_epsilon=0.2,
        lambda_blend=0.6,
    )
    for k, v in extra_config.items():
        setattr(config, k, v)

    trainer = TIPPOTrainer(
        config=config, model=model, ref_model=ref_model, tokenizer=tokenizer,
    )

    gen_kwargs = {
        "max_new_tokens": 512,
        "do_sample": True,
        "top_k": 50,
        "top_p": 0.95,
        "temperature": 0.7,
        "pad_token_id": tokenizer.pad_token_id,
    }

    # Use all 500 training samples
    pool_size = len(math_data)
    pool_indices = list(range(pool_size))
    prompt_cache = {}

    def get_prompt_tokens(idx):
        if idx not in prompt_cache:
            prompt_text = format_prompt(math_data[idx]["problem"], tokenizer)
            tokens = tokenizer(
                prompt_text, truncation=True, max_length=256,
                return_tensors="pt",
            ).input_ids.squeeze(0).to(device)
            prompt_cache[idx] = tokens
        return prompt_cache[idx]

    # Tracking
    history = {
        "rewards": [], "kl": [], "importance": [], "epsilon": [],
        "moai_C": [], "moai_rho": [], "moai_tau2": [],
        "accuracy": [], "correct_count": [], "total_count": [],
    }

    running_correct = 0
    running_total = 0
    batch_size = 4
    start = time.time()

    for ep in range(episodes):
        # Sample a batch of MATH problems
        batch_indices = random.sample(pool_indices, batch_size)
        qts = [get_prompt_tokens(i) for i in batch_indices]

        # Generate responses
        rts = []
        for qt in qts:
            with torch.no_grad():
                out = model.generate(input_ids=qt.unsqueeze(0), **gen_kwargs)
            rts.append(out[0, qt.shape[0]:])

        # Filter empty responses
        valid = [(q, r, i) for q, r, i in zip(qts, rts, batch_indices) if r.shape[0] > 0]
        if not valid:
            continue
        qts = [v[0] for v in valid]
        rts = [v[1] for v in valid]
        valid_indices = [v[2] for v in valid]

        # Score responses
        rewards = []
        ep_correct = 0
        for j, rt in enumerate(rts):
            resp_text = tokenizer.decode(rt, skip_special_tokens=True)
            ground_truth = math_data[valid_indices[j]]["answer"]
            r = math_reward(resp_text, ground_truth)
            rewards.append(torch.tensor(r))
            if r > 0:
                ep_correct += 1

        running_correct += ep_correct
        running_total += len(valid)

        # PPO step
        try:
            stats = trainer.step(qts, rts, rewards)
        except Exception as e:
            print(f"  ep={ep} error: {e}")
            continue

        # Record history
        accuracy = running_correct / max(1, running_total)
        history["rewards"].append(stats["ppo/mean_reward"])
        history["kl"].append(stats["ppo/mean_kl"])
        history["importance"].append(stats["ti_ppo/mean_importance"])
        history["accuracy"].append(accuracy)
        history["correct_count"].append(running_correct)
        history["total_count"].append(running_total)

        if "moai/epsilon" in stats:
            history["epsilon"].append(stats["moai/epsilon"])
            history["moai_C"].append(stats["moai/C"])
            history["moai_rho"].append(stats["moai/rho"])
            history["moai_tau2"].append(stats["moai/tau2"])
        elif hasattr(trainer.scorer, 'epsilon'):
            history["epsilon"].append(trainer.scorer.epsilon)
        else:
            history["epsilon"].append(1.0)

        # Logging
        if ep % 10 == 0 or ep == episodes - 1:
            eps_str = ""
            if "moai/epsilon" in stats:
                eps_str = (f"  eps={stats['moai/epsilon']:.3f}"
                          f"  C={stats['moai/C']:.4f}"
                          f"  rho={stats['moai/rho']:.6f}")
            elif hasattr(trainer.scorer, 'epsilon'):
                eps_str = f"  eps={trainer.scorer.epsilon:.3f}"

            batch_acc = ep_correct / len(valid)
            print(f"  ep={ep:>3d}  r={stats['ppo/mean_reward']:.3f}  "
                  f"kl={stats['ppo/mean_kl']:.4f}  "
                  f"imp={stats['ti_ppo/mean_importance']:.3f}  "
                  f"batch_acc={batch_acc:.2f}  "
                  f"running_acc={accuracy:.4f}{eps_str}")

    elapsed = time.time() - start

    # Test evaluation on all 500 MATH-500 examples
    test_results = evaluate_on_test(model, tokenizer, test_data, device, n_samples=500)

    # Cleanup
    del ref_model, trainer
    prompt_cache.clear()
    torch.cuda.empty_cache()

    # Note: we keep model alive until after evaluate_on_test, then delete
    del model
    torch.cuda.empty_cache()

    # Compute summary statistics
    n = len(history["rewards"])
    q = max(1, n // 4)

    def quartile_mean(lst, q_idx):
        s = q_idx * q
        e = min((q_idx + 1) * q, len(lst))
        if s >= len(lst):
            return 0.0
        return sum(lst[s:e]) / max(1, e - s)

    result = {
        "label": label,
        "method": method,
        "model": model_name,
        "episodes": n,
        "time": round(elapsed, 1),
        "reward_q1": round(quartile_mean(history["rewards"], 0), 4),
        "reward_q2": round(quartile_mean(history["rewards"], 1), 4),
        "reward_q3": round(quartile_mean(history["rewards"], 2), 4),
        "reward_q4": round(quartile_mean(history["rewards"], 3), 4),
        "reward_max": round(max(history["rewards"]), 4) if n else 0,
        "kl_q1": round(quartile_mean(history["kl"], 0), 4),
        "kl_q4": round(quartile_mean(history["kl"], 3), 4),
        "accuracy_q1": round(quartile_mean(history["accuracy"], 0), 4),
        "accuracy_q2": round(quartile_mean(history["accuracy"], 1), 4),
        "accuracy_q3": round(quartile_mean(history["accuracy"], 2), 4),
        "accuracy_q4": round(quartile_mean(history["accuracy"], 3), 4),
        "final_accuracy": round(history["accuracy"][-1], 4) if n else 0,
        "total_correct": running_correct,
        "total_samples": running_total,
        "reward_improvement": round(
            quartile_mean(history["rewards"], 3) - quartile_mean(history["rewards"], 0), 4),
        "test_accuracy": test_results["accuracy"],
        "test_correct": test_results["correct"],
        "test_total": test_results["total"],
        "test_subject_accuracy": test_results["subject_accuracy"],
    }

    abs_kl = abs(result["kl_q4"])
    result["kl_efficiency"] = round(result["reward_q4"] / abs_kl, 2) if abs_kl > 0.001 else float("inf")

    if n > 10:
        diffs = [abs(history["rewards"][i] - history["rewards"][i-1]) for i in range(1, n)]
        result["reward_volatility"] = round(sum(diffs) / len(diffs), 4)
    else:
        result["reward_volatility"] = 0.0

    print(f"\n  Done {elapsed:.0f}s | R: Q1={result['reward_q1']:.4f} Q2={result['reward_q2']:.4f} "
          f"Q3={result['reward_q3']:.4f} Q4={result['reward_q4']:.4f}")
    print(f"  Train Acc: Q1={result['accuracy_q1']:.4f} Q2={result['accuracy_q2']:.4f} "
          f"Q3={result['accuracy_q3']:.4f} Q4={result['accuracy_q4']:.4f} "
          f"final={result['final_accuracy']:.4f}")
    print(f"  Test Acc: {result['test_accuracy']:.4f} ({result['test_correct']}/{result['test_total']})")
    print(f"  KL: Q1={result['kl_q1']:.4f} Q4={result['kl_q4']:.4f} | "
          f"KL-eff={result['kl_efficiency']} | vol={result['reward_volatility']:.4f}")

    return result, history


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def print_results_table(results):
    """Print a formatted results table."""
    baseline = next((r for r in results if r["method"] == "uniform"), None)

    print(f"\n\n{'='*140}")
    print(f"{'MATH (Hendrycks) BENCHMARK RESULTS -- TI-PPO TOKEN IMPORTANCE METHODS':^140}")
    print(f"{'='*140}")
    print(f"{'Method':<30} {'R(Q1)':<8} {'R(Q4)':<8} {'dR':<8} "
          f"{'Acc(Q1)':<8} {'Acc(Q4)':<8} {'FinalAcc':<9} {'TestAcc':<9} "
          f"{'KL(Q4)':<9} {'KL-Eff':<8} {'Vol':<7} {'vs PPO':<8} {'Time':<7}")
    print(f"{'-'*140}")

    for r in sorted(results, key=lambda x: x["test_accuracy"], reverse=True):
        vs_ppo = (r["test_accuracy"] - baseline["test_accuracy"]) if baseline else 0
        kl_eff = f"{r['kl_efficiency']:.1f}" if r['kl_efficiency'] != float('inf') else "inf"
        print(
            f"{r['label']:<30} "
            f"{r['reward_q1']:<8.4f} "
            f"{r['reward_q4']:<8.4f} "
            f"{'+' if r['reward_improvement']>=0 else ''}{r['reward_improvement']:<7.4f} "
            f"{r['accuracy_q1']:<8.4f} "
            f"{r['accuracy_q4']:<8.4f} "
            f"{r['final_accuracy']:<9.4f} "
            f"{r['test_accuracy']:<9.4f} "
            f"{r['kl_q4']:<9.4f} "
            f"{kl_eff:<8} "
            f"{r['reward_volatility']:<7.4f} "
            f"{'+' if vs_ppo>=0 else ''}{vs_ppo:<7.4f} "
            f"{r['time']:<7.0f}"
        )
    print(f"{'='*140}")

    # Pareto analysis
    print(f"\n--- Pareto Frontier Analysis (Test Accuracy vs |KL|) ---")
    for r in sorted(results, key=lambda x: x["test_accuracy"], reverse=True):
        dominated_by = []
        for r2 in results:
            if r2["label"] == r["label"]:
                continue
            if (r2["test_accuracy"] >= r["test_accuracy"]
                    and abs(r2["kl_q4"]) <= abs(r["kl_q4"])):
                dominated_by.append(r2["label"])
        status = ("*** PARETO-OPTIMAL ***" if not dominated_by
                  else f"dominated by: {', '.join(dominated_by[:3])}")
        print(f"  {r['label']:<30} TestAcc={r['test_accuracy']:.4f}  "
              f"|KL|={abs(r['kl_q4']):.4f}  -> {status}")

    # Per-subject test accuracy comparison
    print(f"\n--- Per-Subject Test Accuracy ---")
    subjects = set()
    for r in results:
        subjects.update(r.get("test_subject_accuracy", {}).keys())
    subjects = sorted(subjects)

    if subjects:
        header = f"{'Method':<30} " + " ".join(f"{s[:12]:<13}" for s in subjects)
        print(header)
        print("-" * len(header))
        for r in sorted(results, key=lambda x: x["test_accuracy"], reverse=True):
            sa = r.get("test_subject_accuracy", {})
            parts = []
            for s in subjects:
                if s in sa:
                    parts.append(f"{sa[s]['accuracy']:<13.4f}")
                else:
                    parts.append(f"{'N/A':<13}")
            print(f"{r['label']:<30} " + " ".join(parts))


def main():
    import argparse
    parser = argparse.ArgumentParser(description="MATH (Hendrycks) TI-PPO Benchmark")
    parser.add_argument("--episodes", type=int, default=100,
                        help="Number of training episodes per method")
    parser.add_argument("--gpu", type=int, default=4,
                        help="GPU index to use")
    parser.add_argument("--method", type=str, default="all",
                        help="Method index (0-7) or 'all' to run all sequentially")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--test-only", action="store_true",
                        help="Only run test evaluation (no training)")
    parser.add_argument("--n_train", type=int, default=0,
                        help="Number of training examples (0 = use all)")
    args = parser.parse_args()

    # Set seeds
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(f"cuda:{args.gpu}")
    print(f"=== MATH (Hendrycks) TI-PPO Benchmark ===")
    print(f"Episodes: {args.episodes} | GPU: {device} | Seed: {args.seed}")

    # Ensure output directory exists
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "benchmark")
    os.makedirs(log_dir, exist_ok=True)

    # Load datasets
    n_train = args.n_train if args.n_train > 0 else 99999
    print(f"\n--- Loading training data (n={n_train if n_train < 99999 else 'all'} from EleutherAI/hendrycks_math) ---")
    math_train = load_math_train(n_samples=n_train, seed=42)
    print(f"\n--- Loading test data (HuggingFaceH4/MATH-500) ---")
    math_test = load_math_test()

    # Probe which model works on this GPU
    print(f"\nProbing model availability...")
    model, ref_model, tokenizer, model_name, lora_targets = load_model_and_tokenizer(device)

    if args.test_only:
        # Just evaluate the base model
        print("\n--- Base model test evaluation ---")
        test_results = evaluate_on_test(model, tokenizer, math_test, device, n_samples=500)
        del model, ref_model
        torch.cuda.empty_cache()

        out_path = os.path.join(log_dir, "benchmark_math_base_test.json")
        with open(out_path, "w") as f:
            json.dump({"test_results": test_results, "model": model_name}, f, indent=2)
        print(f"\nBase test results saved to {out_path}")
        return

    # Free the probe models; each method loads its own fresh copy
    del model, ref_model
    torch.cuda.empty_cache()

    # Determine which methods to run
    if args.method == "all":
        methods_to_run = list(enumerate(METHODS))
    else:
        idx = int(args.method)
        if idx < 0 or idx >= len(METHODS):
            print(f"Error: method index {idx} out of range [0, {len(METHODS)-1}]")
            sys.exit(1)
        methods_to_run = [(idx, METHODS[idx])]

    print(f"\nMethods to run: {len(methods_to_run)}")
    for i, (method, label, _) in methods_to_run:
        print(f"  [{i}] {label} ({method})")
    print()

    results = []
    histories = {}

    for i, (method, label, extra) in methods_to_run:
        r, h = run_method(
            method, label, extra, tokenizer, math_train, math_test,
            device, model_name, lora_targets, args.episodes,
        )
        results.append(r)
        histories[label] = h

        # Save intermediate results after each method
        method_tag = f"{i}_{method}"
        out_path = os.path.join(log_dir, f"benchmark_math_results_{method_tag}.json")
        with open(out_path, "w") as f:
            json.dump(
                {"results": [r], "history": h, "config": {
                    "episodes": args.episodes, "gpu": args.gpu,
                    "seed": args.seed, "model": model_name,
                    "train_samples": len(math_train),
                    "test_samples": len(math_test),
                }},
                f, indent=2,
                default=lambda x: None if x == float('inf') else x,
            )
        print(f"  Saved to {out_path}")

    # Print combined results table if we ran multiple methods
    if len(results) > 1:
        print_results_table(results)

        # Epsilon trajectory for AITI/MOAI methods
        print(f"\n--- Epsilon Trajectory (quartile averages) ---")
        for label, h in histories.items():
            eps_list = h.get("epsilon", [])
            if eps_list and any(e != 1.0 for e in eps_list):
                n = len(eps_list)
                q = max(1, n // 4)
                parts = []
                for qi in range(4):
                    s, e = qi * q, min((qi + 1) * q, n)
                    if s < n:
                        eps_avg = sum(eps_list[s:e]) / max(1, e - s)
                        parts.append(f"Q{qi+1}={eps_avg:.3f}")
                print(f"  {label}: {' -> '.join(parts)}")

        # Accuracy trajectory
        print(f"\n--- Accuracy Trajectory (quartile averages) ---")
        for label, h in histories.items():
            acc_list = h.get("accuracy", [])
            if acc_list:
                n = len(acc_list)
                q = max(1, n // 4)
                parts = []
                for qi in range(4):
                    s, e = qi * q, min((qi + 1) * q, n)
                    if s < n:
                        acc_avg = sum(acc_list[s:e]) / max(1, e - s)
                        parts.append(f"Q{qi+1}={acc_avg:.4f}")
                print(f"  {label}: {' -> '.join(parts)}")

    # Save combined results
    combined_path = os.path.join(log_dir, "benchmark_math_results_combined.json")
    with open(combined_path, "w") as f:
        json.dump(
            {"results": results, "histories": histories, "config": {
                "episodes": args.episodes, "gpu": args.gpu,
                "seed": args.seed, "model": model_name,
                "train_samples": len(math_train),
                "test_samples": len(math_test),
            }},
            f, indent=2,
            default=lambda x: None if x == float('inf') else x,
        )
    print(f"\nCombined results saved to {combined_path}")


if __name__ == "__main__":
    main()
