"""Benchmark v2: PPO-native importance methods vs paper's hybrid vs baseline.

Tests methods that use PPO-internal signals (advantages, entropy, KL)
which should be more aligned with the optimization objective.
"""

import json, os, time, math, torch, torch.nn.functional as F
from transformers import AutoTokenizer
from peft import LoraConfig, get_peft_model
from ti_ppo import TIPPOConfig, TIPPOTrainer, CausalLMWithValueHead


METHODS = [
    # (method, triplet, label)
    ("uniform",            False, "PPO baseline"),
    ("hybrid",             True,  "Paper: Hybrid+Triplet"),
    ("advantage",          False, "Ours: |Advantage|"),
    ("entropy",            False, "Ours: Entropy"),
    ("kl_guided",          False, "Ours: KL-Guided Adv"),
    ("adv_gaussian",       False, "Ours: Adv+Gaussian"),
    ("entropy_advantage",  False, "Ours: Entropy*Adv"),
]

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


def synthetic_reward(prompt, response, tokenizer):
    tokens = tokenizer.encode(response)
    n = len(tokens)
    if n == 0:
        return -2.0
    length_r = math.exp(-0.5 * ((n - 50) / 25) ** 2)
    unique_r = min(len(set(tokens)) / n * 1.5, 1.0)
    info_words = ["because", "therefore", "means", "example", "important",
                  "process", "first", "helps", "called", "when", "which"]
    info_r = min(sum(1 for w in info_words if w in response.lower()) / 3.0, 1.0)
    fluency_r = 1.0 if response.strip() and response.strip()[-1] in ".!?" else 0.3
    if any(response.count(c * 5) > 0 for c in "abcdefghijklmnopqrstuvwxyz "):
        return -1.0
    return (0.3 * length_r + 0.25 * unique_r + 0.25 * info_r + 0.2 * fluency_r) * 3 - 1.0


def run_method(method, use_triplet, label, tokenizer, prompt_tokens, device, episodes=100):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  method={method}, triplet={use_triplet}")
    print(f"{'='*60}")

    model = CausalLMWithValueHead.from_pretrained(
        "gpt2", torch_dtype=torch.float32, device_map={"": device}
    )
    lora_cfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
                          bias="none", task_type="CAUSAL_LM",
                          target_modules=["c_attn", "c_proj"])
    model.pretrained_model = get_peft_model(model.pretrained_model, lora_cfg)

    ref_model = CausalLMWithValueHead.from_pretrained(
        "gpt2", torch_dtype=torch.float32, device_map={"": device}
    )
    for p in ref_model.parameters():
        p.requires_grad = False

    config = TIPPOConfig(
        importance_method=method, use_triplet_loss=use_triplet,
        ppo_epochs=2, learning_rate=5e-5, lora_r=8, lora_alpha=16,
        max_new_tokens=64, clip_epsilon=0.2, lambda_blend=0.6,
    )
    trainer = TIPPOTrainer(config=config, model=model, ref_model=ref_model,
                           tokenizer=tokenizer)

    gen_kwargs = {"max_new_tokens": 64, "do_sample": True, "top_k": 50,
                  "top_p": 0.95, "temperature": 0.8,
                  "pad_token_id": tokenizer.pad_token_id}

    history = {"rewards": [], "kl": [], "importance": []}
    start = time.time()

    for ep in range(episodes):
        idx = torch.randint(0, len(prompt_tokens), (4,))
        qts = [prompt_tokens[i] for i in idx]
        rts = []
        for qt in qts:
            with torch.no_grad():
                out = model.generate(input_ids=qt.unsqueeze(0), **gen_kwargs)
            rts.append(out[0, qt.shape[0]:])

        valid = [(q, r, i) for q, r, i in zip(qts, rts, idx) if r.shape[0] > 0]
        if not valid:
            continue
        qts = [v[0] for v in valid]
        rts = [v[1] for v in valid]
        valid_idx = [v[2] for v in valid]

        rewards = []
        for j, rt in enumerate(rts):
            resp_text = tokenizer.decode(rt, skip_special_tokens=True)
            r = synthetic_reward(PROMPTS[valid_idx[j]], resp_text, tokenizer)
            rewards.append(torch.tensor(r))

        try:
            stats = trainer.step(qts, rts, rewards)
        except Exception as e:
            print(f"  ep={ep} error: {e}")
            continue

        history["rewards"].append(stats["ppo/mean_reward"])
        history["kl"].append(stats["ppo/mean_kl"])
        history["importance"].append(stats["ti_ppo/mean_importance"])

        if ep % 25 == 0:
            sample = tokenizer.decode(rts[0], skip_special_tokens=True)[:80]
            print(f"  ep={ep:>3d}  r={stats['ppo/mean_reward']:.3f}  "
                  f"kl={stats['ppo/mean_kl']:.4f}  "
                  f"imp={stats['ti_ppo/mean_importance']:.3f}")

    elapsed = time.time() - start
    del model, ref_model, trainer
    torch.cuda.empty_cache()

    n = len(history["rewards"])
    q = max(1, n // 4)
    result = {
        "label": label,
        "method": method,
        "triplet": use_triplet,
        "episodes": n,
        "time": round(elapsed, 1),
        "reward_first_q": round(sum(history["rewards"][:q]) / q, 4) if n else 0,
        "reward_last_q": round(sum(history["rewards"][-q:]) / q, 4) if n else 0,
        "reward_max": round(max(history["rewards"]), 4) if n else 0,
        "kl_last_q": round(sum(history["kl"][-q:]) / q, 4) if n else 0,
        "imp_last_q": round(sum(history["importance"][-q:]) / q, 4) if n else 0,
    }
    result["reward_improvement"] = round(result["reward_last_q"] - result["reward_first_q"], 4)
    # KL-efficiency: reward per unit KL
    if result["kl_last_q"] > 0.001:
        result["kl_efficiency"] = round(result["reward_last_q"] / result["kl_last_q"], 2)
    else:
        result["kl_efficiency"] = float("inf")

    print(f"\n  Done {elapsed:.0f}s | first_q={result['reward_first_q']:.4f} -> "
          f"last_q={result['reward_last_q']:.4f} | KL={result['kl_last_q']:.4f} | "
          f"KL-eff={result['kl_efficiency']}")
    return result, history


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    prompt_tokens = [
        tokenizer(p, truncation=True, max_length=64,
                  return_tensors="pt").input_ids.squeeze(0).to(device)
        for p in PROMPTS
    ]

    print(f"=== TI-PPO Benchmark v2 ===")
    print(f"Episodes: {args.episodes} | Methods: {len(METHODS)} | GPU: {device}")
    print(f"Focus: PPO-native importance methods vs paper's hybrid\n")

    results = []
    histories = {}

    for method, triplet, label in METHODS:
        r, h = run_method(method, triplet, label, tokenizer, prompt_tokens,
                          device, args.episodes)
        results.append(r)
        histories[label] = h

    # Final table
    baseline = next(r for r in results if r["method"] == "uniform")
    paper = next(r for r in results if r["method"] == "hybrid")

    print(f"\n\n{'='*100}")
    print(f"{'BENCHMARK v2 RESULTS':^100}")
    print(f"{'='*100}")
    print(f"{'Method':<25} {'Reward(1Q)':<12} {'Reward(4Q)':<12} {'Improve':<10} "
          f"{'KL':<10} {'KL-Eff':<10} {'vs PPO':<10} {'vs Paper':<10}")
    print(f"{'-'*100}")

    for r in sorted(results, key=lambda x: x["reward_last_q"], reverse=True):
        vs_ppo = r["reward_last_q"] - baseline["reward_last_q"]
        vs_paper = r["reward_last_q"] - paper["reward_last_q"]
        kl_eff = f"{r['kl_efficiency']:.1f}" if r['kl_efficiency'] != float('inf') else "inf"
        print(
            f"{r['label']:<25} "
            f"{r['reward_first_q']:<12.4f} "
            f"{r['reward_last_q']:<12.4f} "
            f"{'+' if r['reward_improvement']>=0 else ''}{r['reward_improvement']:<9.4f} "
            f"{r['kl_last_q']:<10.4f} "
            f"{kl_eff:<10} "
            f"{'+' if vs_ppo>=0 else ''}{vs_ppo:<9.4f} "
            f"{'+' if vs_paper>=0 else ''}{vs_paper:<9.4f}"
        )
    print(f"{'='*100}")
    print(f"\nKL-Efficiency = Reward / KL  (higher = better alignment per unit divergence)")

    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "benchmark_v2_results.json")
    with open(out, "w") as f:
        json.dump({"results": results, "histories": histories}, f, indent=2)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
