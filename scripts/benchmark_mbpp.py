"""Benchmark: MBPP (Mostly Basic Python Problems) with TI-PPO.

Tests token-importance methods on code generation (MBPP dataset).
The model generates Python functions, and the reward is binary:
+1 if all unit tests pass, -1 otherwise. Code execution uses
Python's exec() in a restricted sandbox with a timeout.

Training data: 120 examples from google-research-datasets/mbpp (sanitized, prompt split)
Test data: 257 examples from google-research-datasets/mbpp (sanitized, test split)

Methods tested:
0. PPO baseline (uniform weighting)
1. Entropy (fixed intensity)
2. AITI-Entropy (linear decay, decay_steps=100)
3. MOAI-Entropy mono (ema=0.90)

Usage (parallel across GPUs):
    python scripts/benchmark_mbpp.py --method 0 --gpu 0 &
    python scripts/benchmark_mbpp.py --method 1 --gpu 1 &
    ...

Or sequential on one GPU:
    python scripts/benchmark_mbpp.py --method all --gpu 4
"""

import json, os, sys, time, re, random, traceback
import multiprocessing
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
    ("uniform", "PPO baseline", {}),
    ("entropy", "Entropy (fixed)", {}),
    ("aiti_entropy", "AITI-Entropy", {
        "aiti_epsilon_max": 1.0, "aiti_decay_steps": 100,
        "aiti_power": 1.0, "aiti_min_epsilon": 0.0}),
    ("moai_entropy_mono", "MOAI-Entropy", {
        "moai_ema_decay": 0.90, "moai_warmup_steps": 5}),
]

# -----------------------------------------------------------------------
# MBPP dataset loading
# -----------------------------------------------------------------------


def load_mbpp_train():
    """Load MBPP sanitized training (prompt) split.

    Returns list of dicts with 'prompt', 'code', 'test_list', 'test_imports'.
    """
    from datasets import load_dataset

    ds = load_dataset("google-research-datasets/mbpp", "sanitized",
                      split="prompt", trust_remote_code=True)
    examples = []
    for row in ds:
        examples.append({
            "prompt": row["prompt"],
            "code": row["code"],
            "test_list": row["test_list"],
            "test_imports": row.get("test_imports", []),
        })

    print(f"Loaded {len(examples)} MBPP training (prompt) examples")
    return examples


def load_mbpp_test():
    """Load MBPP sanitized test split.

    Returns list of dicts with 'prompt', 'code', 'test_list', 'test_imports'.
    """
    from datasets import load_dataset

    ds = load_dataset("google-research-datasets/mbpp", "sanitized",
                      split="test", trust_remote_code=True)
    examples = []
    for row in ds:
        examples.append({
            "prompt": row["prompt"],
            "code": row["code"],
            "test_list": row["test_list"],
            "test_imports": row.get("test_imports", []),
        })

    print(f"Loaded {len(examples)} MBPP test examples")
    return examples


# -----------------------------------------------------------------------
# Code extraction
# -----------------------------------------------------------------------

def extract_code(response):
    """Extract Python code from the model's response.

    Looks for ```python ... ``` blocks first, then tries the raw response.
    Strips any explanation text before/after the function.
    """
    # Strategy 1: look for ```python ... ``` fenced block
    pattern = r"```python\s*\n(.*?)```"
    matches = re.findall(pattern, response, re.DOTALL)
    if matches:
        # Take the longest match (most likely the full function)
        return max(matches, key=len).strip()

    # Strategy 2: look for ``` ... ``` (no language tag)
    pattern = r"```\s*\n(.*?)```"
    matches = re.findall(pattern, response, re.DOTALL)
    if matches:
        code = max(matches, key=len).strip()
        # Only use if it looks like Python code
        if "def " in code or "import " in code or "return " in code:
            return code

    # Strategy 3: look for lines starting with 'def ' and take everything
    # from there to the end of the indented block
    lines = response.split("\n")
    code_lines = []
    in_function = False
    for line in lines:
        if line.strip().startswith("def "):
            in_function = True
            code_lines = [line]
        elif in_function:
            # Continue if indented or blank line within function
            if line.strip() == "" or line.startswith(" ") or line.startswith("\t"):
                code_lines.append(line)
            else:
                # End of function block
                break

    if code_lines:
        return "\n".join(code_lines).strip()

    # Strategy 4: return raw response stripped of obvious explanation
    # Remove lines that look like natural language
    cleaned = []
    for line in lines:
        stripped = line.strip()
        # Skip empty lines at start
        if not cleaned and not stripped:
            continue
        # Skip lines that look like explanation (start with common patterns)
        if stripped and not any(stripped.startswith(p) for p in
                                ["Here", "This", "The ", "Note", "I ",
                                 "We ", "You ", "Sure", "Let me"]):
            cleaned.append(line)
        elif cleaned:
            # Once we've started collecting, keep everything
            cleaned.append(line)

    return "\n".join(cleaned).strip() if cleaned else response.strip()


# -----------------------------------------------------------------------
# Safe code execution with timeout
# -----------------------------------------------------------------------

def _run_tests_in_process(code_str, test_list, test_imports, result_queue):
    """Worker function for multiprocessing-based sandbox execution.

    Runs code + test assertions in a restricted environment.
    Puts True/False into result_queue.
    """
    try:
        # Build restricted globals: no file system, no network, no os
        restricted_builtins = {
            k: v for k, v in __builtins__.__dict__.items()
            if k not in ("open", "exec", "eval", "compile", "__import__",
                         "input", "breakpoint")
        } if hasattr(__builtins__, '__dict__') else {
            k: v for k, v in __builtins__.items()
            if k not in ("open", "exec", "eval", "compile", "__import__",
                         "input", "breakpoint")
        }

        # We need __import__ for test_imports but restrict what can be imported
        ALLOWED_MODULES = {
            "math", "collections", "itertools", "functools", "operator",
            "string", "re", "heapq", "bisect", "copy", "typing",
            "statistics", "decimal", "fractions", "random", "hashlib",
            "sys", "os.path",
        }

        original_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__

        def safe_import(name, *args, **kwargs):
            top_level = name.split(".")[0]
            if top_level not in ALLOWED_MODULES and name not in ALLOWED_MODULES:
                raise ImportError(f"Import of '{name}' is not allowed")
            return original_import(name, *args, **kwargs)

        restricted_builtins["__import__"] = safe_import
        restricted_builtins["__builtins__"] = restricted_builtins

        exec_globals = {"__builtins__": restricted_builtins}

        # Execute test imports
        for imp in test_imports:
            if imp.strip():
                exec(imp, exec_globals)

        # Execute the generated code
        exec(code_str, exec_globals)

        # Run each test assertion
        for test in test_list:
            if test.strip():
                exec(test, exec_globals)

        result_queue.put(True)
    except Exception:
        result_queue.put(False)


def execute_tests_safe(code_str, test_list, test_imports, timeout=5):
    """Execute code + tests in a separate process with timeout.

    Returns True if all tests pass, False otherwise.
    """
    ctx = multiprocessing.get_context("spawn")
    result_queue = ctx.Queue()
    proc = ctx.Process(
        target=_run_tests_in_process,
        args=(code_str, test_list, test_imports, result_queue),
    )
    proc.start()
    proc.join(timeout=timeout)

    if proc.is_alive():
        proc.kill()
        proc.join(timeout=2)
        return False

    if result_queue.empty():
        return False

    return result_queue.get()


# -----------------------------------------------------------------------
# Reward function
# -----------------------------------------------------------------------

def mbpp_reward(response, test_list, test_imports):
    """Binary reward: +1.0 if all tests pass, -1.0 otherwise.

    1. Extract code from model response
    2. Prepend test_imports
    3. Execute each test assertion with 5-second timeout
    4. All pass -> +1.0, any fail -> -1.0
    """
    code = extract_code(response)
    if not code or not code.strip():
        return -1.0

    passed = execute_tests_safe(code, test_list, test_imports, timeout=5)
    return 1.0 if passed else -1.0


# -----------------------------------------------------------------------
# Prompt formatting
# -----------------------------------------------------------------------

def format_prompt(task_prompt, tokenizer=None):
    """Format an MBPP task using the model's chat template."""
    content = (
        "Write a Python function to solve the following task. "
        "Only output the function, no explanation.\n\n"
        f"{task_prompt}"
    )
    if tokenizer is not None and hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template:
        messages = [{"role": "user", "content": content}]
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False)
        except TypeError:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
    return f"Q: {content}\nA:\n"


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

def evaluate_on_test(model, tokenizer, test_data, device, n_samples=257):
    """Evaluate the model on MBPP test examples.

    Uses greedy decoding for deterministic evaluation.

    Returns:
        dict with accuracy, correct, total, and per-example details.
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

    model.eval()
    for i, ex in enumerate(eval_data):
        prompt_text = format_prompt(ex["prompt"], tokenizer)
        tokens = tokenizer(
            prompt_text, truncation=True, max_length=256,
            return_tensors="pt",
        ).input_ids.to(device)

        with torch.no_grad():
            out = model.generate(input_ids=tokens, **gen_kwargs)
        response_tokens = out[0, tokens.shape[1]:]
        if response_tokens.shape[0] == 0:
            total += 1
            continue

        response_text = tokenizer.decode(response_tokens, skip_special_tokens=True)
        r = mbpp_reward(response_text, ex["test_list"], ex["test_imports"])

        total += 1
        if r > 0:
            correct += 1

        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{len(eval_data)}: {correct}/{total} = {correct/max(1,total):.4f}")

    accuracy = correct / max(1, total)
    print(f"  Test accuracy: {correct}/{total} = {accuracy:.4f}")

    return {
        "accuracy": round(accuracy, 4),
        "correct": correct,
        "total": total,
    }


# -----------------------------------------------------------------------
# Training loop for a single method
# -----------------------------------------------------------------------

def run_method(method, label, extra_config, tokenizer, train_data, test_data,
               device, model_name, lora_targets, episodes=200):
    """Run a single TI-PPO method on MBPP.

    Args:
        method: importance method name
        label: human-readable label
        extra_config: dict of extra config overrides
        tokenizer: tokenizer
        train_data: list of MBPP training examples (120)
        test_data: list of MBPP test examples (257)
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
    print(f"  train_pool={len(train_data)} | test_set={len(test_data)}")
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

    # Use all training samples
    pool_size = len(train_data)
    pool_indices = list(range(pool_size))
    prompt_cache = {}

    def get_prompt_tokens(idx):
        if idx not in prompt_cache:
            prompt_text = format_prompt(train_data[idx]["prompt"], tokenizer)
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
        # Sample a batch of MBPP problems
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
            ex = train_data[valid_indices[j]]
            r = mbpp_reward(resp_text, ex["test_list"], ex["test_imports"])
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

    # Test evaluation on all 257 test examples
    test_results = evaluate_on_test(model, tokenizer, test_data, device, n_samples=257)

    # Cleanup
    del ref_model, trainer
    prompt_cache.clear()
    torch.cuda.empty_cache()

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
# Results table
# -----------------------------------------------------------------------

def print_results_table(results):
    """Print a formatted results table."""
    baseline = next((r for r in results if r["method"] == "uniform"), None)

    print(f"\n\n{'='*120}")
    print(f"{'MBPP BENCHMARK RESULTS -- TI-PPO TOKEN IMPORTANCE METHODS':^120}")
    print(f"{'='*120}")
    print(f"{'Method':<25} {'R(Q1)':<8} {'R(Q4)':<8} {'dR':<8} "
          f"{'Acc(Q1)':<8} {'Acc(Q4)':<8} {'FinalAcc':<9} {'TestAcc':<9} "
          f"{'KL(Q4)':<9} {'KL-Eff':<8} {'Vol':<7} {'vs PPO':<8} {'Time':<7}")
    print(f"{'-'*120}")

    for r in sorted(results, key=lambda x: x["test_accuracy"], reverse=True):
        vs_ppo = (r["test_accuracy"] - baseline["test_accuracy"]) if baseline else 0
        kl_eff = f"{r['kl_efficiency']:.1f}" if r['kl_efficiency'] != float('inf') else "inf"
        print(
            f"{r['label']:<25} "
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
    print(f"{'='*120}")

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
        print(f"  {r['label']:<25} TestAcc={r['test_accuracy']:.4f}  "
              f"|KL|={abs(r['kl_q4']):.4f}  -> {status}")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="MBPP TI-PPO Benchmark")
    parser.add_argument("--episodes", type=int, default=200,
                        help="Number of training episodes per method")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU index to use")
    parser.add_argument("--method", type=str, default="all",
                        help="Method index (0-3) or 'all' to run all sequentially")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--test-only", action="store_true",
                        help="Only run test evaluation (no training)")
    args = parser.parse_args()

    # Set seeds
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(f"cuda:{args.gpu}")
    print(f"=== MBPP TI-PPO Benchmark ===")
    print(f"Episodes: {args.episodes} | GPU: {device} | Seed: {args.seed}")

    # Ensure output directory exists
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "benchmark")
    os.makedirs(log_dir, exist_ok=True)

    # Load datasets
    print(f"\n--- Loading MBPP training data (sanitized, prompt split) ---")
    mbpp_train = load_mbpp_train()
    print(f"\n--- Loading MBPP test data (sanitized, test split) ---")
    mbpp_test = load_mbpp_test()

    # Probe which model works on this GPU
    print(f"\nProbing model availability...")
    model, ref_model, tokenizer, model_name, lora_targets = load_model_and_tokenizer(device)

    if args.test_only:
        # Just evaluate the base model
        print("\n--- Base model test evaluation ---")
        test_results = evaluate_on_test(model, tokenizer, mbpp_test, device, n_samples=257)
        del model, ref_model
        torch.cuda.empty_cache()

        out_path = os.path.join(log_dir, "benchmark_mbpp_base_test.json")
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
            method, label, extra, tokenizer, mbpp_train, mbpp_test,
            device, model_name, lora_targets, args.episodes,
        )
        results.append(r)
        histories[label] = h

        # Save intermediate results after each method
        out_path = os.path.join(log_dir, f"benchmark_mbpp_results_{i}_{method}.json")
        with open(out_path, "w") as f:
            json.dump(
                {"results": [r], "history": h, "config": {
                    "episodes": args.episodes, "gpu": args.gpu,
                    "seed": args.seed, "model": model_name,
                    "train_samples": len(mbpp_train),
                    "test_samples": len(mbpp_test),
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
    combined_path = os.path.join(log_dir, "benchmark_mbpp_results_combined.json")
    with open(combined_path, "w") as f:
        json.dump(
            {"results": results, "histories": histories, "config": {
                "episodes": args.episodes, "gpu": args.gpu,
                "seed": args.seed, "model": model_name,
                "train_samples": len(mbpp_train),
                "test_samples": len(mbpp_test),
            }},
            f, indent=2,
            default=lambda x: None if x == float('inf') else x,
        )
    print(f"\nCombined results saved to {combined_path}")


if __name__ == "__main__":
    main()
