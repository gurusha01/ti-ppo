"""Benchmark all token-importance methods against baseline PPO.

Uses GPT-2 (124M) as the policy model and a synthetic reward function
(combining coherence, helpfulness keywords, and length penalty) to keep
GPU memory manageable and allow running all 6 methods sequentially.

Usage:
    python scripts/benchmark.py
    python scripts/benchmark.py --episodes 150 --gpu 7
"""

import argparse
import json
import os
import time
import torch
import torch.nn.functional as F
import math
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

from ti_ppo import TIPPOConfig, TIPPOTrainer, CausalLMWithValueHead


METHODS = [
    ("uniform", False, "PPO (baseline)"),
    ("hybrid", True, "TI-PPO Hybrid + Triplet"),
    ("hybrid", False, "TI-PPO Hybrid"),
    ("gradient", False, "TI-PPO Gradient"),
    ("attention", False, "TI-PPO Attention"),
    ("td_error", False, "TI-PPO TD-Error"),
]

# Prompts for training — mix of instruction-following tasks
PROMPTS = [
    "Explain what machine learning is in simple terms.",
    "What are the benefits of regular exercise?",
    "Write a short poem about the ocean.",
    "How does photosynthesis work?",
    "What is the difference between a list and a tuple in Python?",
    "Describe the water cycle in three sentences.",
    "What causes earthquakes?",
    "Explain how a computer works to a child.",
    "What are three tips for better sleep?",
    "Why is the sky blue?",
    "What is climate change and why does it matter?",
    "Explain the concept of supply and demand.",
    "What are the main differences between cats and dogs as pets?",
    "How do vaccines work?",
    "What is artificial intelligence?",
    "Describe the process of making bread.",
    "What are renewable energy sources?",
    "Explain gravity in simple terms.",
    "What are the benefits of reading books?",
    "How does the internet work?",
    "What is democracy?",
    "Explain how airplanes fly.",
    "What are the planets in our solar system?",
    "Why is biodiversity important?",
    "What is the greenhouse effect?",
    "How do computers store information?",
    "What causes seasons to change?",
    "Explain what DNA is.",
    "What are the basic principles of cooking?",
    "How does electricity work?",
    "What is evolution?",
    "Describe how a car engine works.",
]


def synthetic_reward(prompt: str, response: str, tokenizer) -> float:
    """Synthetic reward combining multiple quality signals.

    Components:
    1. Length: prefer moderate-length responses (30-80 tokens), penalize very short/long
    2. Coherence: low repetition ratio
    3. Informativeness: contains explanation-like words
    4. Fluency: ends with proper punctuation
    """
    tokens = tokenizer.encode(response)
    n_tokens = len(tokens)

    # 1. Length reward: bell curve centered at 50 tokens
    length_reward = math.exp(-0.5 * ((n_tokens - 50) / 25) ** 2)

    # 2. Repetition penalty: ratio of unique tokens
    if n_tokens > 0:
        unique_ratio = len(set(tokens)) / n_tokens
        rep_reward = min(unique_ratio * 1.5, 1.0)
    else:
        return -2.0  # empty response

    # 3. Informativeness: presence of explanation markers
    info_words = ["because", "therefore", "means", "example", "important",
                  "process", "first", "helps", "called", "when", "which"]
    response_lower = response.lower()
    info_count = sum(1 for w in info_words if w in response_lower)
    info_reward = min(info_count / 3.0, 1.0)

    # 4. Fluency: ends with sentence-ending punctuation
    stripped = response.strip()
    fluency_reward = 1.0 if stripped and stripped[-1] in ".!?" else 0.3

    # 5. No degenerate patterns
    if any(response.count(c * 5) > 0 for c in "abcdefghijklmnopqrstuvwxyz "):
        return -1.0

    total = 0.3 * length_reward + 0.25 * rep_reward + 0.25 * info_reward + 0.2 * fluency_reward
    # Scale to roughly [-1, 2] range
    return total * 3 - 1.0


class SyntheticRewardModel:
    """Wraps synthetic_reward to match the trainer.get_rewards interface."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.device = torch.device("cpu")


def run_method(method_name, use_triplet, label, model_name,
               prompts, tokenizer, episodes, batch_size, device):
    """Train one method and return metrics history."""
    print(f"\n{'='*60}")
    print(f"  Running: {label}")
    print(f"  Method: {method_name}, Triplet: {use_triplet}")
    print(f"{'='*60}\n")

    config = TIPPOConfig(
        model_name=model_name,
        importance_method=method_name,
        use_triplet_loss=use_triplet,
        lambda_blend=0.7,
        total_episodes=episodes,
        batch_size=batch_size,
        mini_batch_size=min(4, batch_size),
        ppo_epochs=2,
        max_new_tokens=64,
        max_prompt_length=64,
        learning_rate=5e-5,
        use_peft=True,
        lora_r=8,
        lora_alpha=16,
        clip_epsilon=0.2,
        gamma=1.0,
        lam=0.95,
    )

    # Fresh model for each method
    model = CausalLMWithValueHead.from_pretrained(
        model_name, torch_dtype=torch.float32, device_map={"": device}
    )
    lora_config = LoraConfig(
        r=config.lora_r, lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout, bias="none", task_type="CAUSAL_LM",
        target_modules=["c_attn", "c_proj"],
    )
    model.pretrained_model = get_peft_model(model.pretrained_model, lora_config)

    ref_model = CausalLMWithValueHead.from_pretrained(
        model_name, torch_dtype=torch.float32, device_map={"": device}
    )
    for p in ref_model.parameters():
        p.requires_grad = False

    trainer = TIPPOTrainer(
        config=config, model=model, ref_model=ref_model,
        tokenizer=tokenizer, reward_model=None,
    )

    gen_kwargs = {
        "max_new_tokens": config.max_new_tokens,
        "do_sample": True, "top_k": 50, "top_p": 0.95,
        "temperature": 0.8, "pad_token_id": tokenizer.pad_token_id,
    }

    # Tokenize prompts
    prompt_tokens = [
        tokenizer(p, truncation=True, max_length=config.max_prompt_length,
                  return_tensors="pt").input_ids.squeeze(0).to(device)
        for p in prompts
    ]

    history = {"rewards": [], "kl": [], "policy_loss": [], "importance_mean": [], "importance_std": []}
    start_time = time.time()

    for episode in range(episodes):
        # Sample a mini-batch of prompts
        indices = torch.randint(0, len(prompt_tokens), (batch_size,))
        query_tensors = [prompt_tokens[i] for i in indices]
        query_texts = [prompts[i] for i in indices]

        # Generate responses
        response_tensors = []
        for qt in query_tensors:
            with torch.no_grad():
                out = model.generate(input_ids=qt.unsqueeze(0), **gen_kwargs)
            resp = out[0, qt.shape[0]:]
            response_tensors.append(resp)

        # Filter empty
        valid_idx = [i for i, r in enumerate(response_tensors) if r.shape[0] > 0]
        if not valid_idx:
            continue
        query_tensors = [query_tensors[i] for i in valid_idx]
        response_tensors = [response_tensors[i] for i in valid_idx]
        query_texts_batch = [query_texts[i] for i in valid_idx]

        # Compute synthetic rewards
        rewards = []
        for qt_text, rt in zip(query_texts_batch, response_tensors):
            resp_text = tokenizer.decode(rt, skip_special_tokens=True)
            r = synthetic_reward(qt_text, resp_text, tokenizer)
            rewards.append(torch.tensor(r))

        try:
            stats = trainer.step(query_tensors, response_tensors, rewards)
        except Exception as e:
            print(f"  Step {episode} error: {e}")
            continue

        history["rewards"].append(stats["ppo/mean_reward"])
        history["kl"].append(stats["ppo/mean_kl"])
        history["policy_loss"].append(stats["ppo/policy_loss"])
        history["importance_mean"].append(stats["ti_ppo/mean_importance"])
        history["importance_std"].append(stats["ti_ppo/importance_std"])

        if episode % 20 == 0:
            # Show a sample response
            sample_resp = tokenizer.decode(response_tensors[0], skip_special_tokens=True)
            print(
                f"  [{label}] ep={episode:>4d}  "
                f"reward={stats['ppo/mean_reward']:.3f}  "
                f"kl={stats['ppo/mean_kl']:.4f}  "
                f"imp={stats['ti_ppo/mean_importance']:.3f}"
            )
            if episode % 40 == 0:
                print(f"    Sample: {query_texts_batch[0][:50]}...")
                print(f"    Response: {sample_resp[:100]}...")

    elapsed = time.time() - start_time

    # Cleanup
    del model, ref_model, trainer
    torch.cuda.empty_cache()

    # Summary stats
    n = len(history["rewards"])
    last_q = max(1, n // 4)
    summary = {
        "label": label,
        "method": method_name,
        "triplet": use_triplet,
        "episodes_completed": n,
        "time_seconds": round(elapsed, 1),
        "final_reward_mean": round(sum(history["rewards"][-last_q:]) / last_q, 4) if n else 0,
        "final_reward_max": round(max(history["rewards"][-last_q:]), 4) if n else 0,
        "overall_reward_mean": round(sum(history["rewards"]) / max(n, 1), 4),
        "first_quarter_reward": round(sum(history["rewards"][:last_q]) / last_q, 4) if n else 0,
        "final_kl": round(sum(history["kl"][-last_q:]) / last_q, 4) if n else 0,
        "final_importance_mean": round(sum(history["importance_mean"][-last_q:]) / last_q, 4) if n else 0,
    }

    print(f"\n  {label} finished in {elapsed:.0f}s ({n} episodes)")
    print(f"  Reward: first_25%={summary['first_quarter_reward']:.4f} -> last_25%={summary['final_reward_mean']:.4f}")
    print(f"  KL: {summary['final_kl']:.4f}")

    return summary, history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gpu", type=int, default=7)
    parser.add_argument("--output", type=str, default="benchmark_results.json")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"=== TI-PPO Benchmark ===")
    print(f"Model: {args.model_name}")
    print(f"Device: {device}")
    print(f"Episodes per method: {args.episodes}")
    print(f"Batch size: {args.batch_size}")
    print(f"Methods: {len(METHODS)}")
    print(f"Reward: Synthetic (coherence + informativeness + length + fluency)")
    print()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_results = []
    all_histories = {}

    for method_name, use_triplet, label in METHODS:
        summary, history = run_method(
            method_name, use_triplet, label,
            args.model_name, PROMPTS, tokenizer,
            args.episodes, args.batch_size, device,
        )
        all_results.append(summary)
        all_histories[label] = history

    # Print final comparison table
    print(f"\n\n{'='*90}")
    print(f"{'BENCHMARK RESULTS':^90}")
    print(f"{'='*90}")
    print(
        f"{'Method':<28} "
        f"{'Reward (1st Q)':<16} "
        f"{'Reward (last Q)':<16} "
        f"{'Improvement':<14} "
        f"{'KL':<10} "
        f"{'Time':<10}"
    )
    print(f"{'-'*90}")

    baseline_reward = None
    for r in sorted(all_results, key=lambda x: x["final_reward_mean"], reverse=True):
        if r["method"] == "uniform":
            baseline_reward = r["final_reward_mean"]

    for r in sorted(all_results, key=lambda x: x["final_reward_mean"], reverse=True):
        improvement = r["final_reward_mean"] - r["first_quarter_reward"]
        vs_baseline = ""
        if baseline_reward is not None and r["method"] != "uniform":
            diff = r["final_reward_mean"] - baseline_reward
            vs_baseline = f" ({'+' if diff >= 0 else ''}{diff:.3f} vs PPO)"

        print(
            f"{r['label']:<28} "
            f"{r['first_quarter_reward']:<16.4f} "
            f"{r['final_reward_mean']:<16.4f} "
            f"{'+' if improvement >= 0 else ''}{improvement:<13.4f} "
            f"{r['final_kl']:<10.4f} "
            f"{r['time_seconds']:<10.1f}s"
            f"{vs_baseline}"
        )
    print(f"{'='*90}")

    # Save results
    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), args.output)
    with open(output_path, "w") as f:
        json.dump({"results": all_results, "histories": all_histories}, f, indent=2)
    print(f"\nDetailed results saved to {output_path}")


if __name__ == "__main__":
    main()
