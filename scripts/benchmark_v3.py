"""Benchmark v3: Theoretically-derived optimal importance methods.

Tests 4 novel methods derived from optimization theory:
1. POTI (Pareto-Optimal Token Importance) - Lagrangian constrained optimization
2. Adaptive Phase - Entropy→Advantage annealing
3. SNR - Signal-to-Noise Ratio from PG variance reduction theory
4. Entropy-KL Lagrangian - Formalized entropy regularization with KL constraint

Compared against the best from v2: Entropy, |Advantage|, Paper Hybrid, PPO baseline.

Key question: Can we beat entropy on reward WHILE also beating the paper on KL-efficiency?
"""

import json, os, sys, time, math, copy
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from peft import LoraConfig, get_peft_model

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ti_ppo import TIPPOConfig, TIPPOTrainer, CausalLMWithValueHead


METHODS = [
    # (method, triplet, label)
    ("uniform",                False, "PPO baseline"),
    ("hybrid",                 True,  "Paper: Hybrid+Triplet"),
    ("entropy",                False, "v2 Best: Entropy"),
    ("advantage",              False, "v2 Best: |Advantage|"),
    # --- NEW Phase 2 methods ---
    ("pareto",                 False, "NEW: Pareto-Optimal (POTI)"),
    ("adaptive_phase",         False, "NEW: Adaptive Phase"),
    ("snr",                    False, "NEW: SNR (Var-Optimal)"),
    ("entropy_kl_lagrangian",  False, "NEW: Entropy-KL Lagrangian"),
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


def run_method(method, use_triplet, label, tokenizer, prompt_tokens, device, episodes=150):
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

    history = {"rewards": [], "kl": [], "importance": [], "policy_loss": [], "value_loss": []}
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
        history["policy_loss"].append(stats["ppo/policy_loss"])
        history["value_loss"].append(stats["ppo/value_loss"])

        if ep % 25 == 0:
            # Track dual variable for Lagrangian methods
            dual_info = ""
            if hasattr(trainer.scorer, 'lambda_dual'):
                dual_info = f"  λ={trainer.scorer.lambda_dual:.3f}"
            elif hasattr(trainer.scorer, 'mu_dual'):
                dual_info = f"  μ={trainer.scorer.mu_dual:.3f}"
            elif hasattr(trainer.scorer, 'step_count'):
                alpha = min(1.0, trainer.scorer.step_count / trainer.scorer.warmup_steps)
                dual_info = f"  α={alpha:.2f}"

            print(f"  ep={ep:>3d}  r={stats['ppo/mean_reward']:.3f}  "
                  f"kl={stats['ppo/mean_kl']:.4f}  "
                  f"imp={stats['ti_ppo/mean_importance']:.3f}{dual_info}")

    elapsed = time.time() - start
    del model, ref_model, trainer
    torch.cuda.empty_cache()

    n = len(history["rewards"])
    q = max(1, n // 4)

    # Compute quartile stats for detailed analysis
    def quartile_mean(lst, q_idx):
        start_idx = q_idx * q
        end_idx = min((q_idx + 1) * q, len(lst))
        if start_idx >= len(lst):
            return 0.0
        return sum(lst[start_idx:end_idx]) / max(1, end_idx - start_idx)

    result = {
        "label": label,
        "method": method,
        "triplet": use_triplet,
        "episodes": n,
        "time": round(elapsed, 1),
        "reward_q1": round(quartile_mean(history["rewards"], 0), 4),
        "reward_q2": round(quartile_mean(history["rewards"], 1), 4),
        "reward_q3": round(quartile_mean(history["rewards"], 2), 4),
        "reward_q4": round(quartile_mean(history["rewards"], 3), 4),
        "reward_max": round(max(history["rewards"]), 4) if n else 0,
        "kl_q1": round(quartile_mean(history["kl"], 0), 4),
        "kl_q4": round(quartile_mean(history["kl"], 3), 4),
        "imp_q4": round(quartile_mean(history["importance"], 3), 4),
    }

    result["reward_first_q"] = result["reward_q1"]
    result["reward_last_q"] = result["reward_q4"]
    result["kl_last_q"] = result["kl_q4"]
    result["reward_improvement"] = round(result["reward_q4"] - result["reward_q1"], 4)

    # KL-efficiency: |reward| / |KL| (use abs KL to handle negative KL)
    abs_kl = abs(result["kl_q4"])
    if abs_kl > 0.001:
        result["kl_efficiency"] = round(result["reward_q4"] / abs_kl, 2)
    else:
        result["kl_efficiency"] = float("inf")

    # Reward trajectory stability (lower = more stable)
    if n > 10:
        diffs = [abs(history["rewards"][i] - history["rewards"][i-1]) for i in range(1, n)]
        result["reward_volatility"] = round(sum(diffs) / len(diffs), 4)
    else:
        result["reward_volatility"] = 0.0

    print(f"\n  Done {elapsed:.0f}s | Q1={result['reward_q1']:.4f} Q2={result['reward_q2']:.4f} "
          f"Q3={result['reward_q3']:.4f} Q4={result['reward_q4']:.4f}")
    print(f"  KL: Q1={result['kl_q1']:.4f} Q4={result['kl_q4']:.4f} | "
          f"KL-eff={result['kl_efficiency']} | vol={result['reward_volatility']:.4f}")
    return result, history


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=150)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--methods", type=str, default=None,
                        help="Comma-separated method indices to run (e.g., 0,1,4,5)")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    prompt_tokens = [
        tokenizer(p, truncation=True, max_length=64,
                  return_tensors="pt").input_ids.squeeze(0).to(device)
        for p in PROMPTS
    ]

    methods_to_run = METHODS
    if args.methods:
        indices = [int(i) for i in args.methods.split(",")]
        methods_to_run = [METHODS[i] for i in indices]

    print(f"=== TI-PPO Benchmark v3: Theoretically-Optimal Methods ===")
    print(f"Episodes: {args.episodes} | Methods: {len(methods_to_run)} | GPU: {device}")
    print(f"Novel methods: POTI, Adaptive Phase, SNR, Entropy-KL Lagrangian\n")

    results = []
    histories = {}

    for method, triplet, label in methods_to_run:
        r, h = run_method(method, triplet, label, tokenizer, prompt_tokens,
                          device, args.episodes)
        results.append(r)
        histories[label] = h

    # Final analysis
    baseline = next((r for r in results if r["method"] == "uniform"), None)
    paper = next((r for r in results if r["method"] == "hybrid"), None)
    v2_best = next((r for r in results if r["method"] == "entropy" and "v2" in r["label"]), None)

    print(f"\n\n{'='*120}")
    print(f"{'BENCHMARK v3 RESULTS — THEORETICALLY-DERIVED METHODS':^120}")
    print(f"{'='*120}")
    print(f"{'Method':<30} {'R(Q1)':<9} {'R(Q2)':<9} {'R(Q3)':<9} {'R(Q4)':<9} "
          f"{'ΔR':<9} {'KL(Q4)':<10} {'KL-Eff':<9} {'Vol':<8} "
          f"{'vs PPO':<9} {'vs Paper':<9} {'vs v2Best':<9}")
    print(f"{'-'*120}")

    for r in sorted(results, key=lambda x: x["reward_q4"], reverse=True):
        vs_ppo = (r["reward_q4"] - baseline["reward_q4"]) if baseline else 0
        vs_paper = (r["reward_q4"] - paper["reward_q4"]) if paper else 0
        vs_v2 = (r["reward_q4"] - v2_best["reward_q4"]) if v2_best else 0
        kl_eff = f"{r['kl_efficiency']:.1f}" if r['kl_efficiency'] != float('inf') else "inf"
        print(
            f"{r['label']:<30} "
            f"{r['reward_q1']:<9.4f} "
            f"{r['reward_q2']:<9.4f} "
            f"{r['reward_q3']:<9.4f} "
            f"{r['reward_q4']:<9.4f} "
            f"{'+' if r['reward_improvement']>=0 else ''}{r['reward_improvement']:<8.4f} "
            f"{r['kl_q4']:<10.4f} "
            f"{kl_eff:<9} "
            f"{r['reward_volatility']:<8.4f} "
            f"{'+' if vs_ppo>=0 else ''}{vs_ppo:<8.4f} "
            f"{'+' if vs_paper>=0 else ''}{vs_paper:<8.4f} "
            f"{'+' if vs_v2>=0 else ''}{vs_v2:<8.4f}"
        )
    print(f"{'='*120}")

    # Pareto frontier analysis
    print(f"\n--- Pareto Frontier Analysis ---")
    print(f"A method is Pareto-dominated if another method beats it on BOTH reward AND |KL|.")
    for r in sorted(results, key=lambda x: x["reward_q4"], reverse=True):
        dominated_by = []
        for r2 in results:
            if r2["label"] == r["label"]:
                continue
            if r2["reward_q4"] >= r["reward_q4"] and abs(r2["kl_q4"]) <= abs(r["kl_q4"]):
                dominated_by.append(r2["label"])
        status = "PARETO-OPTIMAL" if not dominated_by else f"dominated by: {', '.join(dominated_by)}"
        print(f"  {r['label']:<30} R={r['reward_q4']:.4f}  |KL|={abs(r['kl_q4']):.4f}  → {status}")

    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "benchmark_v3_results.json")
    with open(out, "w") as f:
        json.dump({"results": results, "histories": histories}, f, indent=2,
                  default=lambda x: None if x == float('inf') else x)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
