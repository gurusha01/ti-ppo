"""Batch-size sweep for top TI-PPO methods on GSM8K.

Sweeps batch_size in [1, 2, 4, 8] for the PPO baseline and top TI-PPO methods
to show how token-importance weighting interacts with batch size.

Usage (parallel across GPUs):
    python scripts/sweep_batchsize.py --method 0 --batch_size 1 --gpu 0 --episodes 100
    python scripts/sweep_batchsize.py --method 0 --batch_size 2 --gpu 1 --episodes 100
    python scripts/sweep_batchsize.py --method 1 --batch_size 4 --gpu 2 --episodes 100
    # etc.

Or run all batch sizes for one method sequentially:
    python scripts/sweep_batchsize.py --method 0 --batch_size all --gpu 0 --episodes 100
"""

import json, os, sys, time, re, random
import torch
from transformers import AutoTokenizer
from peft import LoraConfig, get_peft_model

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ti_ppo import TIPPOConfig, TIPPOTrainer, CausalLMWithValueHead

# -----------------------------------------------------------------------
# Methods to sweep (PPO baseline + best TI-PPO methods)
# -----------------------------------------------------------------------

METHODS = [
    ("uniform", "PPO baseline", {}),
    ("aiti_advantage", "AITI-Advantage", {
        "aiti_epsilon_max": 1.0, "aiti_decay_steps": 100,
        "aiti_power": 1.0, "aiti_min_epsilon": 0.0,
    }),
    ("moai_entropy_mono", "MOAI-Ent mono 0.80", {
        "moai_ema_decay": 0.80, "moai_warmup_steps": 5,
    }),
]

BATCH_SIZES = [1, 2, 4, 8]

# -----------------------------------------------------------------------
# GSM8K helpers (mirrored from benchmark_gsm8k.py)
# -----------------------------------------------------------------------

def load_gsm8k():
    """Load GSM8K dataset from HuggingFace."""
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split="train")
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
    print(f"Loaded {len(examples)} GSM8K training examples")
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
    """Extract the final numeric answer from the model's response."""
    # Pattern 1: \boxed{number}
    matches = re.findall(r"\\boxed\{([\-\d,\.]+)\}", response)
    if matches:
        try:
            return float(matches[-1].replace(",", ""))
        except ValueError:
            pass

    # Pattern 2: #### number
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


def gsm8k_reward(response, ground_truth_answer):
    """Binary reward: +1.0 if correct, -1.0 if incorrect."""
    predicted = extract_model_answer(response)
    if predicted is None:
        return -1.0
    if abs(predicted - ground_truth_answer) < 0.01:
        return 1.0
    if ground_truth_answer != 0 and abs(predicted - ground_truth_answer) / abs(ground_truth_answer) < 0.001:
        return 1.0
    return -1.0


def format_prompt(question, tokenizer=None):
    """Format a GSM8K question as a prompt for the model."""
    # Old prompt
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
# Training loop for one (method, batch_size) combo
# -----------------------------------------------------------------------

def run_sweep_combo(method, label, extra_config, batch_size, tokenizer,
                    gsm8k_data, device, model_name, lora_targets, episodes=100):
    """Train one (method, batch_size) combination and return results."""
    print(f"\n{'='*70}")
    print(f"  {label}  |  batch_size={batch_size}")
    print(f"  method={method} | model={model_name} | episodes={episodes}")
    print(f"{'='*70}")

    # Load fresh model
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

    # Pre-tokenize a pool of prompts
    pool_size = len(gsm8k_data)  # use full training set
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
        "rewards": [], "kl": [], "importance": [], "accuracy": [],
    }

    running_correct = 0
    running_total = 0
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

        # Logging
        if ep % 10 == 0 or ep == episodes - 1:
            batch_acc = ep_correct / len(valid)
            print(f"  ep={ep:>3d}  r={stats['ppo/mean_reward']:.3f}  "
                  f"kl={stats['ppo/mean_kl']:.4f}  "
                  f"imp={stats['ti_ppo/mean_importance']:.3f}  "
                  f"batch_acc={batch_acc:.2f}  "
                  f"running_acc={accuracy:.4f}")

    elapsed = time.time() - start

    # Final accuracy
    n = len(history["rewards"])
    final_accuracy = history["accuracy"][-1] if n else 0.0

    print(f"\n  Done {elapsed:.0f}s | episodes={n} | "
          f"final_accuracy={final_accuracy:.4f} | "
          f"correct={running_correct}/{running_total}")

    # Build result
    result = {
        "method": method,
        "label": label,
        "batch_size": batch_size,
        "episodes": n,
        "final_accuracy": round(final_accuracy, 4),
        "history": {
            "rewards": history["rewards"],
            "kl": history["kl"],
            "accuracy": history["accuracy"],
        },
        "total_correct": running_correct,
        "total_samples": running_total,
        "time_seconds": round(elapsed, 1),
        "model": model_name,
    }

    # Cleanup
    del model, ref_model, trainer
    prompt_cache.clear()
    torch.cuda.empty_cache()

    return result


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Batch-size sweep for TI-PPO on GSM8K")
    parser.add_argument("--method", type=int, required=True,
                        help=f"Method index (0-{len(METHODS)-1}): "
                             + ", ".join(f"{i}={m[1]}" for i, m in enumerate(METHODS)))
    parser.add_argument("--batch_size", type=str, default="all",
                        help="Batch size (1,2,4,8) or 'all' to sweep all sizes")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU index to use")
    parser.add_argument("--episodes", type=int, default=100,
                        help="Number of training episodes per combo")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()

    if args.method < 0 or args.method >= len(METHODS):
        print(f"Error: method index {args.method} out of range [0, {len(METHODS)-1}]")
        print("Available methods:")
        for i, (m, l, _) in enumerate(METHODS):
            print(f"  [{i}] {l} ({m})")
        sys.exit(1)

    # Determine batch sizes to sweep
    if args.batch_size == "all":
        batch_sizes = BATCH_SIZES
    else:
        bs = int(args.batch_size)
        if bs not in BATCH_SIZES:
            print(f"Warning: batch_size={bs} not in standard set {BATCH_SIZES}, running anyway")
        batch_sizes = [bs]

    # Set seeds
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(f"cuda:{args.gpu}")
    method, label, extra_config = METHODS[args.method]

    print(f"=== Batch-Size Sweep: {label} ===")
    print(f"Method: {method} (index {args.method})")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Episodes: {args.episodes} | GPU: {device} | Seed: {args.seed}")

    # Load dataset
    gsm8k_data = load_gsm8k()

    # Probe model availability
    print(f"\nProbing model availability...")
    probe_model, probe_ref, tokenizer, model_name, lora_targets = load_model_and_tokenizer(device)
    del probe_model, probe_ref
    torch.cuda.empty_cache()

    # Run each batch size
    all_results = []
    for bs in batch_sizes:
        # Reset seed for each combo so results are comparable
        random.seed(args.seed)
        torch.manual_seed(args.seed)

        result = run_sweep_combo(
            method, label, extra_config, bs, tokenizer,
            gsm8k_data, device, model_name, lora_targets, args.episodes,
        )
        all_results.append(result)

        # Save individual result
        out_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            f"logs/sweep_batchsize/sweep_batchsize_results_{method}_{bs}.json",
        )
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Saved to {out_path}")

    # Print summary table
    if len(all_results) > 1:
        print(f"\n{'='*70}")
        print(f"  Batch-Size Sweep Summary: {label}")
        print(f"{'='*70}")
        print(f"{'Batch Size':<12} {'Episodes':<10} {'Final Acc':<12} "
              f"{'Correct':<12} {'Time (s)':<10}")
        print(f"{'-'*70}")
        for r in all_results:
            print(f"{r['batch_size']:<12} {r['episodes']:<10} "
                  f"{r['final_accuracy']:<12.4f} "
                  f"{r['total_correct']}/{r['total_samples']:<8} "
                  f"{r['time_seconds']:<10.1f}")
        print(f"{'='*70}")


if __name__ == "__main__":
    main()
