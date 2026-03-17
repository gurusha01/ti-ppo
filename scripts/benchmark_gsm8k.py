"""Benchmark: GSM8K Math Reasoning with TI-PPO.

Tests token-importance methods on a real task: grade-school math (GSM8K).
Unlike synthetic rewards, this uses binary correctness (answer matches ground truth).

Methods tested:
1. PPO baseline (uniform weighting)
2. AITI-Advantage (linear decay, decay_steps=200)
3. MOAI-Advantage mono (ema=0.80)
4. MOAI-Advantage mono (ema=0.90)

Usage (parallel across GPUs 4-7):
    python scripts/benchmark_gsm8k.py --method 0 --gpu 4 &
    python scripts/benchmark_gsm8k.py --method 1 --gpu 5 &
    python scripts/benchmark_gsm8k.py --method 2 --gpu 6 &
    python scripts/benchmark_gsm8k.py --method 3 --gpu 7 &

Or sequential on one GPU:
    python scripts/benchmark_gsm8k.py --method all --gpu 4
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
    # Entropy-only methods (advantage is positional, not content-based)
    ("uniform", "PPO baseline", {}),
    ("entropy", "Entropy (fixed)", {}),
    ("aiti_entropy", "AITI-Entropy", {
        "aiti_epsilon_max": 1.0, "aiti_decay_steps": 100,
        "aiti_power": 1.0, "aiti_min_epsilon": 0.0}),
    ("moai_entropy_mono", "MOAI-Entropy", {
        "moai_ema_decay": 0.90, "moai_warmup_steps": 5}),
]

# -----------------------------------------------------------------------
# GSM8K dataset loading
# -----------------------------------------------------------------------

def load_gsm8k():
    """Load GSM8K dataset from HuggingFace.

    Returns list of dicts with 'question' and 'answer' (numeric) keys.
    """
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split="train")
    examples = []
    for row in ds:
        question = row["question"]
        # GSM8K answers are formatted as "... #### <number>"
        answer_text = row["answer"]
        numeric_answer = extract_gsm8k_answer(answer_text)
        if numeric_answer is not None:
            examples.append({
                "question": question,
                "answer": numeric_answer,
                "answer_text": answer_text,
            })
    print(f"Loaded {len(examples)} GSM8K training examples")
    return examples


def extract_gsm8k_answer(text):
    """Extract the numeric answer from GSM8K format '#### <number>'.

    Handles integers and decimals, with optional commas (e.g., 1,234).
    """
    match = re.search(r"####\s*([\-\d,\.]+)", text)
    if match:
        num_str = match.group(1).replace(",", "")
        try:
            return float(num_str)
        except ValueError:
            return None
    return None


def extract_model_answer(response):
    """Extract the final numeric answer from the model's response.

    Tries patterns in priority order — structured formats first,
    then natural language, then fallback to last number.
    """
    # Pattern 1: \boxed{number} (our prompt asks for this)
    matches = re.findall(r"\\boxed\{([\-\d,\.]+)\}", response)
    if matches:
        try:
            return float(matches[-1].replace(",", ""))
        except ValueError:
            pass

    # Pattern 2: GSM8K format #### number
    match = re.search(r"####\s*([\-\d,\.]+)", response)
    if match:
        try:
            return float(match.group(1).replace(",", ""))
        except ValueError:
            pass

    # Pattern 3: "the answer is X" / "answer: X"
    match = re.search(r"(?:the\s+answer\s+is|answer\s*[:=])\s*([\-\d,\.]+)", response, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1).replace(",", ""))
        except ValueError:
            pass

    # Pattern 4: "= X" at end of line
    match = re.search(r"=\s*([\-\d,\.]+)\s*[.\n]?\s*$", response, re.MULTILINE)
    if match:
        try:
            return float(match.group(1).replace(",", ""))
        except ValueError:
            pass

    # Pattern 5: Last standalone number (fallback — can be noisy)
    numbers = re.findall(r"(?<!\w)([\-]?\d[\d,]*\.?\d*)", response)
    if numbers:
        try:
            return float(numbers[-1].replace(",", ""))
        except ValueError:
            pass

    return None


def gsm8k_reward(response, ground_truth_answer):
    """Binary reward: +1.0 if correct, -1.0 if incorrect.

    Comparison uses approximate equality for floating point.
    """
    predicted = extract_model_answer(response)
    if predicted is None:
        return -1.0

    # Approximate equality: within 0.01 or within 0.1% relative
    if abs(predicted - ground_truth_answer) < 0.01:
        return 1.0
    if ground_truth_answer != 0 and abs(predicted - ground_truth_answer) / abs(ground_truth_answer) < 0.001:
        return 1.0

    return -1.0


def format_prompt(question, tokenizer=None):
    """Format a GSM8K question using the model's chat template."""
    if tokenizer is not None and hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template:
        messages = [{"role": "user", "content": f"Solve this math problem step by step. Put your final answer in \\boxed{{}}.\n\n{question}"}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"Q: {question}\nA: Let's solve step by step, then give the final numeric answer in \\boxed{{}}.\n"


# -----------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------

PRIMARY_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
FALLBACK_MODEL = "Qwen/Qwen2.5-0.5B"


def load_model_and_tokenizer(device):
    """Load model + LoRA + ValueHead and reference model.

    Tries Qwen2.5-0.5B first. Falls back to GPT-2 if that fails.
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
            print("Falling back to GPT-2...")
            continue


# -----------------------------------------------------------------------
# Training loop for a single method
# -----------------------------------------------------------------------

def run_method(method, label, extra_config, tokenizer, gsm8k_data,
               device, model_name, lora_targets, episodes=300):
    """Run a single TI-PPO method on GSM8K.

    Args:
        method: importance method name
        label: human-readable label
        extra_config: dict of extra config overrides
        tokenizer: tokenizer
        gsm8k_data: list of GSM8K examples
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
        max_new_tokens=384,
        clip_epsilon=0.2,
        lambda_blend=0.6,
    )
    for k, v in extra_config.items():
        setattr(config, k, v)

    trainer = TIPPOTrainer(
        config=config, model=model, ref_model=ref_model, tokenizer=tokenizer,
    )

    gen_kwargs = {
        "max_new_tokens": 384,
        "do_sample": True,
        "top_k": 50,
        "top_p": 0.95,
        "temperature": 0.7,
        "pad_token_id": tokenizer.pad_token_id,
    }

    # Pre-tokenize a pool of prompts (use up to 1000 for variety)
    pool_size = len(gsm8k_data)  # use full training set to avoid overfitting
    pool_indices = list(range(pool_size))
    prompt_cache = {}

    def get_prompt_tokens(idx):
        if idx not in prompt_cache:
            prompt_text = format_prompt(gsm8k_data[idx]["question"], tokenizer)
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
        # Sample a batch of GSM8K problems
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
            ground_truth = gsm8k_data[valid_indices[j]]["answer"]
            r = gsm8k_reward(resp_text, ground_truth)
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

    # Cleanup
    del model, ref_model, trainer
    prompt_cache.clear()
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
    print(f"  Acc: Q1={result['accuracy_q1']:.4f} Q2={result['accuracy_q2']:.4f} "
          f"Q3={result['accuracy_q3']:.4f} Q4={result['accuracy_q4']:.4f} "
          f"final={result['final_accuracy']:.4f}")
    print(f"  KL: Q1={result['kl_q1']:.4f} Q4={result['kl_q4']:.4f} | "
          f"KL-eff={result['kl_efficiency']} | vol={result['reward_volatility']:.4f}")

    return result, history


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def print_results_table(results):
    """Print a formatted results table."""
    baseline = next((r for r in results if r["method"] == "uniform"), None)

    print(f"\n\n{'='*130}")
    print(f"{'GSM8K BENCHMARK RESULTS — TI-PPO TOKEN IMPORTANCE METHODS':^130}")
    print(f"{'='*130}")
    print(f"{'Method':<30} {'R(Q1)':<8} {'R(Q4)':<8} {'dR':<8} "
          f"{'Acc(Q1)':<8} {'Acc(Q4)':<8} {'FinalAcc':<9} "
          f"{'KL(Q4)':<9} {'KL-Eff':<8} {'Vol':<7} {'vs PPO':<8} {'Time':<7}")
    print(f"{'-'*130}")

    for r in sorted(results, key=lambda x: x["accuracy_q4"], reverse=True):
        vs_ppo = (r["accuracy_q4"] - baseline["accuracy_q4"]) if baseline else 0
        kl_eff = f"{r['kl_efficiency']:.1f}" if r['kl_efficiency'] != float('inf') else "inf"
        print(
            f"{r['label']:<30} "
            f"{r['reward_q1']:<8.4f} "
            f"{r['reward_q4']:<8.4f} "
            f"{'+' if r['reward_improvement']>=0 else ''}{r['reward_improvement']:<7.4f} "
            f"{r['accuracy_q1']:<8.4f} "
            f"{r['accuracy_q4']:<8.4f} "
            f"{r['final_accuracy']:<9.4f} "
            f"{r['kl_q4']:<9.4f} "
            f"{kl_eff:<8} "
            f"{r['reward_volatility']:<7.4f} "
            f"{'+' if vs_ppo>=0 else ''}{vs_ppo:<7.4f} "
            f"{r['time']:<7.0f}"
        )
    print(f"{'='*130}")

    # Pareto analysis
    print(f"\n--- Pareto Frontier Analysis (Accuracy vs |KL|) ---")
    for r in sorted(results, key=lambda x: x["accuracy_q4"], reverse=True):
        dominated_by = []
        for r2 in results:
            if r2["label"] == r["label"]:
                continue
            if (r2["accuracy_q4"] >= r["accuracy_q4"]
                    and abs(r2["kl_q4"]) <= abs(r["kl_q4"])):
                dominated_by.append(r2["label"])
        status = ("*** PARETO-OPTIMAL ***" if not dominated_by
                  else f"dominated by: {', '.join(dominated_by[:3])}")
        print(f"  {r['label']:<30} Acc={r['accuracy_q4']:.4f}  "
              f"|KL|={abs(r['kl_q4']):.4f}  -> {status}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="GSM8K TI-PPO Benchmark")
    parser.add_argument("--episodes", type=int, default=150,
                        help="Number of training episodes per method")
    parser.add_argument("--gpu", type=int, default=4,
                        help="GPU index to use")
    parser.add_argument("--method", type=str, default="all",
                        help="Method index (0-3) or 'all' to run all sequentially")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()

    # Set seeds
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(f"cuda:{args.gpu}")
    print(f"=== GSM8K TI-PPO Benchmark ===")
    print(f"Episodes: {args.episodes} | GPU: {device} | Seed: {args.seed}")

    # Load dataset
    gsm8k_data = load_gsm8k()

    # Probe which model works on this GPU
    print(f"\nProbing model availability...")
    model, ref_model, tokenizer, model_name, lora_targets = load_model_and_tokenizer(device)
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

    print(f"Methods to run: {len(methods_to_run)}")
    for i, (method, label, _) in methods_to_run:
        print(f"  [{i}] {label} ({method})")
    print()

    results = []
    histories = {}

    for i, (method, label, extra) in methods_to_run:
        r, h = run_method(
            method, label, extra, tokenizer, gsm8k_data,
            device, model_name, lora_targets, args.episodes,
        )
        results.append(r)
        histories[label] = h

        # Save intermediate results after each method
        method_tag = f"{i}_{method}"
        out_path = os.path.join(
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "benchmark"),
            f"benchmark_gsm8k_results_{method_tag}.json",
        )
        with open(out_path, "w") as f:
            json.dump(
                {"results": [r], "history": h, "config": {
                    "episodes": args.episodes, "gpu": args.gpu,
                    "seed": args.seed, "model": model_name,
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
    combined_path = os.path.join(
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "benchmark"),
        "benchmark_gsm8k_results_combined.json",
    )
    with open(combined_path, "w") as f:
        json.dump(
            {"results": results, "histories": histories, "config": {
                "episodes": args.episodes, "gpu": args.gpu,
                "seed": args.seed, "model": model_name,
            }},
            f, indent=2,
            default=lambda x: None if x == float('inf') else x,
        )
    print(f"\nCombined results saved to {combined_path}")


if __name__ == "__main__":
    main()
