"""Evaluate trained TI-PPO models on the GSM8K test set.

Trains a model using a given TI-PPO method (same config as benchmark_gsm8k.py),
then evaluates on the full GSM8K test split (1319 examples) with greedy decoding.

Also evaluates the base model (no training) as a comparison point.

Usage (parallel across GPUs):
    python scripts/eval_gsm8k.py --method 0 --gpu 0 --train_episodes 200 &
    python scripts/eval_gsm8k.py --method 1 --gpu 1 --train_episodes 200 &
    ...
    python scripts/eval_gsm8k.py --method 7 --gpu 7 --train_episodes 200 &

Or evaluate base model only (no training):
    python scripts/eval_gsm8k.py --method base --gpu 0

Or run all methods sequentially:
    python scripts/eval_gsm8k.py --method all --gpu 0 --train_episodes 200
"""

import json, os, sys, time, math, re, random
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ti_ppo import TIPPOConfig, TIPPOTrainer, CausalLMWithValueHead

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# -----------------------------------------------------------------------
# Method definitions (same as benchmark_gsm8k.py)
# -----------------------------------------------------------------------

METHODS = [
    ("uniform", "PPO baseline", {}),
    ("aiti_advantage", "AITI-Advantage", {
        "aiti_epsilon_max": 1.0, "aiti_decay_steps": 100,
        "aiti_power": 1.0, "aiti_min_epsilon": 0.0}),
    ("aiti_entropy", "AITI-Entropy", {
        "aiti_epsilon_max": 1.0, "aiti_decay_steps": 100,
        "aiti_power": 1.0, "aiti_min_epsilon": 0.0}),
    ("moai_advantage_mono", "MOAI-Adv mono 0.80", {
        "moai_ema_decay": 0.80, "moai_warmup_steps": 5}),
    ("moai_advantage_mono", "MOAI-Adv mono 0.90", {
        "moai_ema_decay": 0.90, "moai_warmup_steps": 5}),
    ("moai_entropy_mono", "MOAI-Ent mono 0.80", {
        "moai_ema_decay": 0.80, "moai_warmup_steps": 5}),
    ("moai_entropy_mono", "MOAI-Ent mono 0.90", {
        "moai_ema_decay": 0.90, "moai_warmup_steps": 5}),
    ("advantage", "Advantage (fixed)", {}),
]

# -----------------------------------------------------------------------
# Model config
# -----------------------------------------------------------------------

PRIMARY_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
FALLBACK_MODEL = "Qwen/Qwen2.5-0.5B"

# -----------------------------------------------------------------------
# GSM8K dataset loading
# -----------------------------------------------------------------------

def load_gsm8k_split(split):
    """Load GSM8K dataset split from HuggingFace.

    Args:
        split: "train" or "test"

    Returns list of dicts with 'question', 'answer' (numeric), and 'answer_text' keys.
    """
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split=split)
    examples = []
    for row in ds:
        question = row["question"]
        answer_text = row["answer"]
        numeric_answer = extract_gsm8k_answer(answer_text)
        if numeric_answer is not None:
            examples.append({
                "question": question,
                "answer": numeric_answer,
                "answer_text": answer_text,
            })
    print(f"Loaded {len(examples)} GSM8K {split} examples")
    return examples


def extract_gsm8k_answer(text):
    """Extract the numeric answer from GSM8K format '#### <number>'."""
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

    Tries patterns in priority order -- structured formats first,
    then natural language, then fallback to last number.
    """
    # Pattern 1: \boxed{number}
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

    # Pattern 5: Last standalone number (fallback)
    numbers = re.findall(r"(?<!\w)([\-]?\d[\d,]*\.?\d*)", response)
    if numbers:
        try:
            return float(numbers[-1].replace(",", ""))
        except ValueError:
            pass

    return None


def check_answer(predicted, ground_truth):
    """Check if predicted answer matches ground truth (approximate equality)."""
    if predicted is None:
        return False
    if abs(predicted - ground_truth) < 0.01:
        return True
    if ground_truth != 0 and abs(predicted - ground_truth) / abs(ground_truth) < 0.001:
        return True
    return False


def gsm8k_reward(response, ground_truth_answer):
    """Binary reward: +1.0 if correct, -1.0 if incorrect."""
    predicted = extract_model_answer(response)
    if check_answer(predicted, ground_truth_answer):
        return 1.0
    return -1.0


def format_prompt(question, tokenizer=None):
    """Format a GSM8K question using the model's chat template."""
    if tokenizer is not None and hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template:
        messages = [{"role": "user", "content": f"Solve this math problem step by step. Put your final answer in \\boxed{{}}.\n\n{question}"}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"Q: {question}\nA: Let's solve step by step, then give the final numeric answer in \\boxed{{}}.\n"


def solution_step_count(answer_text):
    """Count the number of reasoning steps in a GSM8K ground truth solution.

    Used to classify problems as 'short' vs 'long'.
    """
    lines = [l.strip() for l in answer_text.strip().split("\n") if l.strip()]
    # Exclude the final "#### answer" line
    return max(1, len(lines) - 1)


# -----------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------

def probe_model(device):
    """Probe which model works on this device.

    Returns (model_name, lora_targets).
    """
    for model_name, lora_targets in [
        (PRIMARY_MODEL, ["q_proj", "v_proj"]),
        (FALLBACK_MODEL, ["c_attn", "c_proj"]),
    ]:
        try:
            print(f"Probing {model_name}...")
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left"
            # Quick load test
            m = CausalLMWithValueHead.from_pretrained(
                model_name, dtype=torch.float16, device_map={"": device},
                trust_remote_code=True,
            )
            del m
            torch.cuda.empty_cache()
            print(f"Using {model_name}")
            return model_name, lora_targets, tokenizer
        except Exception as e:
            print(f"Failed to load {model_name}: {e}")
            continue
    raise RuntimeError("Could not load any model")


def load_fresh_model_with_lora(model_name, lora_targets, device):
    """Load a fresh model with LoRA for training."""
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

    return model, ref_model


def load_base_model_for_eval(model_name, device):
    """Load the base pretrained model (no LoRA, no value head) for evaluation."""
    print(f"Loading base model {model_name} for evaluation...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map={"": device},
        trust_remote_code=True,
    )
    model.eval()
    return model


# -----------------------------------------------------------------------
# Training (replicates benchmark_gsm8k.py logic)
# -----------------------------------------------------------------------

def train_method(method, label, extra_config, tokenizer, train_data,
                 device, model_name, lora_targets, episodes=200):
    """Train a single TI-PPO method on GSM8K (same as benchmark_gsm8k.py).

    Returns the trained model (for subsequent evaluation).
    """
    print(f"\n{'='*70}")
    print(f"  TRAINING: {label}")
    print(f"  method={method} | model={model_name} | episodes={episodes}")
    print(f"{'='*70}")

    model, ref_model = load_fresh_model_with_lora(model_name, lora_targets, device)

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

    pool_size = len(train_data)  # use full training set to avoid overfitting
    pool_indices = list(range(pool_size))
    prompt_cache = {}

    def get_prompt_tokens(idx):
        if idx not in prompt_cache:
            prompt_text = format_prompt(train_data[idx]["question"], tokenizer)
            tokens = tokenizer(
                prompt_text, truncation=True, max_length=256,
                return_tensors="pt",
            ).input_ids.squeeze(0).to(device)
            prompt_cache[idx] = tokens
        return prompt_cache[idx]

    running_correct = 0
    running_total = 0
    batch_size = 4
    start = time.time()

    for ep in range(episodes):
        batch_indices = random.sample(pool_indices, batch_size)
        qts = [get_prompt_tokens(i) for i in batch_indices]

        rts = []
        for qt in qts:
            with torch.no_grad():
                out = model.generate(input_ids=qt.unsqueeze(0), **gen_kwargs)
            rts.append(out[0, qt.shape[0]:])

        valid = [(q, r, i) for q, r, i in zip(qts, rts, batch_indices) if r.shape[0] > 0]
        if not valid:
            continue
        qts = [v[0] for v in valid]
        rts = [v[1] for v in valid]
        valid_indices = [v[2] for v in valid]

        rewards = []
        ep_correct = 0
        for j, rt in enumerate(rts):
            resp_text = tokenizer.decode(rt, skip_special_tokens=True)
            ground_truth = train_data[valid_indices[j]]["answer"]
            r = gsm8k_reward(resp_text, ground_truth)
            rewards.append(torch.tensor(r))
            if r > 0:
                ep_correct += 1

        running_correct += ep_correct
        running_total += len(valid)

        try:
            stats = trainer.step(qts, rts, rewards)
        except Exception as e:
            print(f"  ep={ep} error: {e}")
            continue

        if ep % 20 == 0 or ep == episodes - 1:
            accuracy = running_correct / max(1, running_total)
            batch_acc = ep_correct / len(valid)
            print(f"  ep={ep:>3d}  r={stats['ppo/mean_reward']:.3f}  "
                  f"kl={stats['ppo/mean_kl']:.4f}  "
                  f"batch_acc={batch_acc:.2f}  "
                  f"running_acc={accuracy:.4f}")

    elapsed = time.time() - start
    final_acc = running_correct / max(1, running_total)
    print(f"  Training done in {elapsed:.0f}s | running_acc={final_acc:.4f}")

    # Free ref model and trainer but keep the trained model
    del ref_model, trainer
    prompt_cache.clear()
    torch.cuda.empty_cache()

    return model


# -----------------------------------------------------------------------
# Test set evaluation
# -----------------------------------------------------------------------

def evaluate_on_test(model, tokenizer, test_data, device, eval_batch_size=8,
                     use_value_head_model=True):
    """Evaluate a model on the GSM8K test set with greedy decoding.

    Args:
        model: the model to evaluate (CausalLMWithValueHead or AutoModelForCausalLM)
        tokenizer: tokenizer
        test_data: list of GSM8K test examples
        device: torch device
        eval_batch_size: batch size for generation
        use_value_head_model: if True, model is CausalLMWithValueHead (has .generate);
                              if False, model is a standard HF model

    Returns:
        dict with evaluation results
    """
    model.eval()

    gen_kwargs = {
        "max_new_tokens": 384,
        "do_sample": False,  # greedy decoding for evaluation
        "pad_token_id": tokenizer.pad_token_id,
    }

    total = len(test_data)
    correct = 0
    no_answer = 0
    results_per_example = []

    # Classify examples by difficulty (solution length)
    median_steps = sorted([solution_step_count(ex["answer_text"]) for ex in test_data])[total // 2]

    short_correct = 0
    short_total = 0
    long_correct = 0
    long_total = 0

    print(f"\n  Evaluating on {total} test examples (batch_size={eval_batch_size})...")
    start = time.time()

    for batch_start in range(0, total, eval_batch_size):
        batch_end = min(batch_start + eval_batch_size, total)
        batch_examples = test_data[batch_start:batch_end]

        # Tokenize prompts
        prompts = [format_prompt(ex["question"], tokenizer) for ex in batch_examples]
        inputs = tokenizer(
            prompts, truncation=True, max_length=256,
            padding=True, return_tensors="pt",
        ).to(device)

        # Generate
        with torch.no_grad():
            outputs = model.generate(input_ids=inputs.input_ids,
                                     attention_mask=inputs.attention_mask,
                                     **gen_kwargs)

        # Decode and check each response
        for i, ex in enumerate(batch_examples):
            prompt_len = inputs.input_ids[i].shape[0]
            response_tokens = outputs[i, prompt_len:]
            response_text = tokenizer.decode(response_tokens, skip_special_tokens=True)

            predicted = extract_model_answer(response_text)
            is_correct = check_answer(predicted, ex["answer"])

            if predicted is None:
                no_answer += 1

            steps = solution_step_count(ex["answer_text"])
            is_short = steps <= median_steps

            if is_short:
                short_total += 1
                if is_correct:
                    short_correct += 1
            else:
                long_total += 1
                if is_correct:
                    long_correct += 1

            if is_correct:
                correct += 1

            results_per_example.append({
                "question": ex["question"],
                "ground_truth": ex["answer"],
                "predicted": predicted,
                "correct": is_correct,
                "response": response_text[:500],  # truncate for storage
                "steps": steps,
                "is_short": is_short,
            })

        # Progress
        processed = batch_end
        if processed % 100 < eval_batch_size or processed == total:
            acc_so_far = correct / processed
            print(f"    {processed}/{total} done | accuracy so far: {acc_so_far:.4f}")

    elapsed = time.time() - start
    accuracy = correct / total
    short_acc = short_correct / max(1, short_total)
    long_acc = long_correct / max(1, long_total)

    print(f"\n  Evaluation complete in {elapsed:.0f}s")
    print(f"  Total accuracy:  {correct}/{total} = {accuracy:.4f}")
    print(f"  Short solutions ({short_total} examples, <={median_steps} steps): {short_acc:.4f}")
    print(f"  Long solutions  ({long_total} examples, >{median_steps} steps):  {long_acc:.4f}")
    print(f"  No answer extracted: {no_answer}/{total}")

    # Print some example predictions
    correct_examples = [r for r in results_per_example if r["correct"]]
    incorrect_examples = [r for r in results_per_example if not r["correct"]]

    print(f"\n  --- Example CORRECT predictions (up to 3) ---")
    for ex in correct_examples[:3]:
        print(f"    Q: {ex['question'][:100]}...")
        print(f"    Ground truth: {ex['ground_truth']} | Predicted: {ex['predicted']}")
        print(f"    Response: {ex['response'][:200]}...")
        print()

    print(f"  --- Example INCORRECT predictions (up to 3) ---")
    for ex in incorrect_examples[:3]:
        print(f"    Q: {ex['question'][:100]}...")
        print(f"    Ground truth: {ex['ground_truth']} | Predicted: {ex['predicted']}")
        print(f"    Response: {ex['response'][:200]}...")
        print()

    return {
        "total": total,
        "correct": correct,
        "accuracy": round(accuracy, 4),
        "no_answer": no_answer,
        "short_total": short_total,
        "short_correct": short_correct,
        "short_accuracy": round(short_acc, 4),
        "long_total": long_total,
        "long_correct": long_correct,
        "long_accuracy": round(long_acc, 4),
        "median_steps": median_steps,
        "eval_time_seconds": round(elapsed, 1),
        "examples": results_per_example,
    }


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def run_single_method(method_idx, tokenizer, train_data, test_data,
                      device, model_name, lora_targets, train_episodes, seed):
    """Train one method and evaluate on the test set."""
    method, label, extra_config = METHODS[method_idx]

    # Set seeds for reproducibility
    random.seed(seed)
    torch.manual_seed(seed)

    # Train
    trained_model = train_method(
        method, label, extra_config, tokenizer, train_data,
        device, model_name, lora_targets, episodes=train_episodes,
    )

    # Evaluate
    print(f"\n{'='*70}")
    print(f"  EVALUATING: {label} (after {train_episodes} episodes)")
    print(f"{'='*70}")

    eval_result = evaluate_on_test(
        trained_model, tokenizer, test_data, device,
        eval_batch_size=8, use_value_head_model=True,
    )

    # Free model
    del trained_model
    torch.cuda.empty_cache()

    # Build output
    output = {
        "method_idx": method_idx,
        "method": method,
        "label": label,
        "model": model_name,
        "train_episodes": train_episodes,
        "seed": seed,
        "eval": {k: v for k, v in eval_result.items() if k != "examples"},
        "examples": eval_result["examples"],
    }

    # Save
    out_path = os.path.join(PROJECT_ROOT, "logs", "eval", f"eval_gsm8k_results_{method_idx}.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2,
                  default=lambda x: None if x == float('inf') else x)
    print(f"\n  Results saved to {out_path}")

    return output


def run_base_model(tokenizer, test_data, device, model_name):
    """Evaluate the base pretrained model (no training) on the test set."""
    print(f"\n{'='*70}")
    print(f"  EVALUATING: Base model (no training) -- {model_name}")
    print(f"{'='*70}")

    base_model = load_base_model_for_eval(model_name, device)

    eval_result = evaluate_on_test(
        base_model, tokenizer, test_data, device,
        eval_batch_size=8, use_value_head_model=False,
    )

    del base_model
    torch.cuda.empty_cache()

    output = {
        "method_idx": "base",
        "method": "none",
        "label": f"Base model ({model_name})",
        "model": model_name,
        "train_episodes": 0,
        "seed": None,
        "eval": {k: v for k, v in eval_result.items() if k != "examples"},
        "examples": eval_result["examples"],
    }

    out_path = os.path.join(PROJECT_ROOT, "eval_gsm8k_results_base.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2,
                  default=lambda x: None if x == float('inf') else x)
    print(f"\n  Results saved to {out_path}")

    return output


def print_summary_table(all_results):
    """Print a summary comparison table."""
    print(f"\n\n{'='*100}")
    print(f"{'GSM8K TEST SET EVALUATION RESULTS':^100}")
    print(f"{'='*100}")
    print(f"{'Method':<30} {'Accuracy':>10} {'Correct':>10} {'Total':>8} "
          f"{'Short Acc':>10} {'Long Acc':>10} {'No Ans':>8} {'Time':>8}")
    print(f"{'-'*100}")

    for r in sorted(all_results, key=lambda x: x["eval"]["accuracy"], reverse=True):
        e = r["eval"]
        print(f"{r['label']:<30} "
              f"{e['accuracy']:>10.4f} "
              f"{e['correct']:>10d} "
              f"{e['total']:>8d} "
              f"{e['short_accuracy']:>10.4f} "
              f"{e['long_accuracy']:>10.4f} "
              f"{e['no_answer']:>8d} "
              f"{e['eval_time_seconds']:>7.0f}s")
    print(f"{'='*100}")

    # Find base model result for comparison
    base = next((r for r in all_results if r["method_idx"] == "base"), None)
    if base:
        print(f"\n  Base model accuracy: {base['eval']['accuracy']:.4f}")
        print(f"  Improvement over base model:")
        for r in sorted(all_results, key=lambda x: x["eval"]["accuracy"], reverse=True):
            if r["method_idx"] == "base":
                continue
            delta = r["eval"]["accuracy"] - base["eval"]["accuracy"]
            print(f"    {r['label']:<30} {'+' if delta >= 0 else ''}{delta:.4f}")


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Evaluate TI-PPO methods on GSM8K test set")
    parser.add_argument("--method", type=str, default="0",
                        help="Method index (0-7), 'base' for base model only, "
                             "'all' to run all methods + base")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU index to use")
    parser.add_argument("--train_episodes", type=int, default=200,
                        help="Number of training episodes before evaluation")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--eval_batch_size", type=int, default=8,
                        help="Batch size for test set evaluation")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(f"cuda:{args.gpu}")
    print(f"=== GSM8K Test Set Evaluation ===")
    print(f"Method: {args.method} | GPU: {device} | "
          f"Train episodes: {args.train_episodes} | Seed: {args.seed}")

    # Load datasets
    train_data = load_gsm8k_split("train")
    test_data = load_gsm8k_split("test")

    # Probe model
    model_name, lora_targets, tokenizer = probe_model(device)

    all_results = []

    if args.method == "base":
        # Evaluate base model only
        result = run_base_model(tokenizer, test_data, device, model_name)
        all_results.append(result)

    elif args.method == "all":
        # Run base model + all methods
        result = run_base_model(tokenizer, test_data, device, model_name)
        all_results.append(result)

        for idx in range(len(METHODS)):
            result = run_single_method(
                idx, tokenizer, train_data, test_data,
                device, model_name, lora_targets,
                args.train_episodes, args.seed,
            )
            all_results.append(result)

        print_summary_table(all_results)

        # Save combined results
        combined_path = os.path.join(PROJECT_ROOT, "eval_gsm8k_results_combined.json")
        combined_output = {
            "config": {
                "train_episodes": args.train_episodes,
                "gpu": args.gpu,
                "seed": args.seed,
                "model": model_name,
                "eval_batch_size": args.eval_batch_size,
            },
            "results": [
                {k: v for k, v in r.items() if k != "examples"}
                for r in all_results
            ],
        }
        with open(combined_path, "w") as f:
            json.dump(combined_output, f, indent=2,
                      default=lambda x: None if x == float('inf') else x)
        print(f"\nCombined results saved to {combined_path}")

    else:
        # Single method
        idx = int(args.method)
        if idx < 0 or idx >= len(METHODS):
            print(f"Error: method index {idx} out of range [0, {len(METHODS)-1}]")
            sys.exit(1)

        result = run_single_method(
            idx, tokenizer, train_data, test_data,
            device, model_name, lora_targets,
            args.train_episodes, args.seed,
        )
        all_results.append(result)

    print("\nDone.")


if __name__ == "__main__":
    main()
