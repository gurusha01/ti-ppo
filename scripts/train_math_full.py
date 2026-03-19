"""Full MATH training with LoRA checkpoint saving and loss logging.

Trains Qwen3-4B on all 7500 MATH examples for 3 epochs with effective
batch_size=32 (mini_batch=4, grad_accumulation=8). Saves LoRA adapters
every 100 steps.

Usage (parallel, one method per GPU):
    python scripts/train_math_full.py --method 0 --gpu 4 &
    python scripts/train_math_full.py --method 1 --gpu 5 &
    python scripts/train_math_full.py --method 2 --gpu 6 &
    python scripts/train_math_full.py --method 3 --gpu 7 &
"""

import json, os, sys, time, math, re, random
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from peft import LoraConfig, get_peft_model

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ti_ppo import TIPPOConfig, TIPPOTrainer, CausalLMWithValueHead

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# -----------------------------------------------------------------------
# Methods
# -----------------------------------------------------------------------
METHODS = [
    ("uniform", "PPO baseline", {}),
    ("entropy", "Entropy (fixed)", {}),
    ("aiti_entropy", "AITI-Entropy", {
        "aiti_epsilon_max": 1.0, "aiti_decay_steps": 200,
        "aiti_power": 1.0, "aiti_min_epsilon": 0.0}),
    ("moai_entropy_mono", "MOAI-Entropy", {
        "moai_ema_decay": 0.90, "moai_warmup_steps": 5}),
]

# -----------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------
MATH_SUBJECTS = ['algebra', 'counting_and_probability', 'geometry',
                 'intermediate_algebra', 'number_theory', 'prealgebra', 'precalculus']

def load_math_train():
    from datasets import load_dataset
    examples = []
    for subject in MATH_SUBJECTS:
        ds = load_dataset("EleutherAI/hendrycks_math", subject, split="train", trust_remote_code=True)
        for row in ds:
            answer = extract_boxed(row["solution"])
            if answer is not None:
                examples.append({"problem": row["problem"], "answer": answer, "subject": subject})
    print(f"Loaded {len(examples)} MATH train examples")
    return examples

def load_math_test():
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test", trust_remote_code=True)
    examples = []
    for row in ds:
        examples.append({"problem": row["problem"], "answer": row["answer"].strip()})
    print(f"Loaded {len(examples)} MATH-500 test examples")
    return examples

def extract_boxed(text):
    matches = re.findall(r'\\boxed\{([^}]+)\}', text)
    if matches:
        return matches[-1].strip()
    return None

def normalize_answer(s):
    if s is None: return None
    s = s.strip().strip("$")
    s = re.sub(r"\\(?:text|mathrm|textbf|mbox)\{([^}]*)\}", r"\1", s)
    s = s.replace("\\left", "").replace("\\right", "")
    s = re.sub(r"\\[,;:!]", "", s)
    s = s.replace("\\tfrac", "\\frac").replace("\\dfrac", "\\frac")
    s = s.strip()
    return s

def answers_match(pred, gt):
    if pred is None: return False
    p, g = normalize_answer(pred), normalize_answer(gt)
    if p == g: return True
    # Try numeric
    try:
        pf, gf = float(p), float(g)
        if abs(pf - gf) < 0.01: return True
        if gf != 0 and abs(pf - gf) / abs(gf) < 0.001: return True
    except: pass
    # Try fraction eval
    for s in [p, g]:
        m = re.match(r"\\frac\{([\-\d]+)\}\{([\-\d]+)\}", s)
        if m:
            try:
                val = float(m.group(1)) / float(m.group(2))
                other = g if s == p else p
                try:
                    if abs(val - float(other)) < 0.01: return True
                except: pass
            except: pass
    return False

def math_reward(response, gt_answer):
    pred = extract_boxed(response)
    if pred is None:
        # Fallback
        m = re.search(r"(?:answer\s+is|answer[:=])\s*(.+?)(?:\.|$)", response, re.IGNORECASE)
        if m: pred = m.group(1).strip()
    if answers_match(pred, gt_answer):
        return 1.0
    return -1.0

def format_prompt(problem, tokenizer):
    messages = [{"role": "user", "content": f"Solve this math problem. Show your work step by step, then put your final answer in \\boxed{{}}.\n\n{problem}"}]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    method, label, extra = METHODS[args.method]
    device = torch.device(f"cuda:{args.gpu}")
    MODEL = "Qwen/Qwen3-4B-Instruct-2507"

    effective_batch_size = 32
    mini_batch_size = 4  # actual batch per PPO step
    grad_accum_steps = effective_batch_size // mini_batch_size  # 8

    # Load data
    train_data = load_math_train()
    test_data = load_math_test()

    total_steps = (len(train_data) * args.epochs) // effective_batch_size
    print(f"\n{'='*70}")
    print(f"  {label} | {MODEL}")
    print(f"  {len(train_data)} train | {args.epochs} epochs | bs={effective_batch_size} (mini={mini_batch_size} x accum={grad_accum_steps})")
    print(f"  Total steps: {total_steps} | Save every 100 steps")
    print(f"{'='*70}")

    # Load model
    print(f"\nLoading {MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = CausalLMWithValueHead.from_pretrained(MODEL, dtype=torch.float16, device_map={"": device}, trust_remote_code=True)
    lora_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM", target_modules=["q_proj", "v_proj"])
    model.pretrained_model = get_peft_model(model.pretrained_model, lora_cfg)

    ref_model = CausalLMWithValueHead.from_pretrained(MODEL, dtype=torch.float16, device_map={"": device}, trust_remote_code=True)
    for p in ref_model.parameters():
        p.requires_grad = False

    # Configure
    config = TIPPOConfig(
        importance_method=method, use_triplet_loss=False,
        ppo_epochs=4, learning_rate=5e-5, max_new_tokens=512,
        clip_epsilon=0.2, lambda_blend=0.6,
    )
    for k, v in extra.items():
        setattr(config, k, v)

    trainer = TIPPOTrainer(config=config, model=model, ref_model=ref_model, tokenizer=tokenizer)

    gen_kwargs = {
        "max_new_tokens": 512, "do_sample": True, "top_k": 50,
        "top_p": 0.95, "temperature": 0.7, "pad_token_id": tokenizer.pad_token_id,
    }

    # Output dirs
    method_tag = f"{args.method}_{method}"
    ckpt_dir = os.path.join(PROJECT_ROOT, "checkpoints", method_tag)
    os.makedirs(ckpt_dir, exist_ok=True)
    log_dir = os.path.join(PROJECT_ROOT, "logs", "training")
    os.makedirs(log_dir, exist_ok=True)

    # Training loop
    history = {"step": [], "policy_loss": [], "value_loss": [], "kl": [],
               "reward": [], "accuracy": [], "importance": [], "epoch": []}

    # Build epoch-shuffled indices
    all_indices = []
    for epoch in range(args.epochs):
        idx = list(range(len(train_data)))
        random.shuffle(idx)
        all_indices.extend([(epoch, i) for i in idx])

    global_step = 0
    running_correct = 0
    running_total = 0
    start_time = time.time()

    prompt_cache = {}
    def get_tokens(idx):
        if idx not in prompt_cache:
            p = format_prompt(train_data[idx]["problem"], tokenizer)
            t = tokenizer(p, truncation=True, max_length=256, return_tensors="pt").input_ids.squeeze(0).to(device)
            prompt_cache[idx] = t
        return prompt_cache[idx]

    # Process in chunks of effective_batch_size
    for batch_start in range(0, len(all_indices), effective_batch_size):
        batch_indices = all_indices[batch_start:batch_start + effective_batch_size]
        if len(batch_indices) < effective_batch_size:
            break  # skip incomplete last batch

        current_epoch = batch_indices[0][0]
        global_step += 1

        # Process mini-batches with gradient accumulation
        step_rewards = []
        step_correct = 0
        step_total = 0
        step_policy_loss = 0
        step_value_loss = 0
        step_kl = 0
        step_importance = 0

        for accum_idx in range(grad_accum_steps):
            mb_start = accum_idx * mini_batch_size
            mb_indices = batch_indices[mb_start:mb_start + mini_batch_size]

            # Get prompts and generate
            qts = [get_tokens(idx) for _, idx in mb_indices]
            rts = []
            for qt in qts:
                with torch.no_grad():
                    out = model.generate(input_ids=qt.unsqueeze(0), **gen_kwargs)
                rts.append(out[0, qt.shape[0]:])

            # Filter empty
            valid = [(q, r, idx) for (q, r, (_, idx)) in zip(qts, rts, mb_indices) if r.shape[0] > 0]
            if not valid:
                continue
            qts_v = [v[0] for v in valid]
            rts_v = [v[1] for v in valid]
            idx_v = [v[2] for v in valid]

            # Score
            rewards = []
            for j, rt in enumerate(rts_v):
                resp = tokenizer.decode(rt, skip_special_tokens=True)
                r = math_reward(resp, train_data[idx_v[j]]["answer"])
                rewards.append(torch.tensor(r))
                if r > 0:
                    step_correct += 1
                step_total += 1
            step_rewards.extend([r.item() for r in rewards])

            # PPO step
            try:
                stats = trainer.step(qts_v, rts_v, rewards)
                step_policy_loss += stats["ppo/policy_loss"]
                step_value_loss += stats["ppo/value_loss"]
                step_kl += stats["ppo/mean_kl"]
                step_importance += stats["ti_ppo/mean_importance"]
            except Exception as e:
                print(f"  step={global_step} accum={accum_idx} error: {e}")
                continue

        running_correct += step_correct
        running_total += step_total
        acc = running_correct / max(1, running_total)
        mean_reward = sum(step_rewards) / max(1, len(step_rewards))

        # Log
        history["step"].append(global_step)
        history["policy_loss"].append(step_policy_loss / grad_accum_steps)
        history["value_loss"].append(step_value_loss / grad_accum_steps)
        history["kl"].append(step_kl / grad_accum_steps)
        history["reward"].append(mean_reward)
        history["accuracy"].append(acc)
        history["importance"].append(step_importance / grad_accum_steps)
        history["epoch"].append(current_epoch)

        if global_step % 10 == 0 or global_step == 1:
            elapsed = time.time() - start_time
            print(f"  step={global_step:>4d}/{total_steps}  ep={current_epoch}  "
                  f"ploss={step_policy_loss/grad_accum_steps:.4f}  "
                  f"vloss={step_value_loss/grad_accum_steps:.4f}  "
                  f"kl={step_kl/grad_accum_steps:.4f}  "
                  f"r={mean_reward:.3f}  acc={acc:.4f}  "
                  f"imp={step_importance/grad_accum_steps:.3f}  "
                  f"[{elapsed:.0f}s]", flush=True)

        # Save checkpoint every 100 steps
        if global_step % 100 == 0:
            ckpt_path = os.path.join(ckpt_dir, f"step_{global_step}")
            model.pretrained_model.save_pretrained(ckpt_path)
            print(f"  >> Saved LoRA checkpoint to {ckpt_path}", flush=True)

    # Save final checkpoint
    final_path = os.path.join(ckpt_dir, f"step_{global_step}_final")
    model.pretrained_model.save_pretrained(final_path)
    print(f"\nSaved final LoRA to {final_path}")

    elapsed = time.time() - start_time
    print(f"\nTraining done in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"Final train accuracy: {acc:.4f} ({running_correct}/{running_total})")

    # Evaluate on test
    print(f"\nEvaluating on {len(test_data)} MATH-500 test examples (greedy)...")
    test_correct = 0
    for i, ex in enumerate(test_data):
        p = format_prompt(ex["problem"], tokenizer)
        t = tokenizer(p, truncation=True, max_length=256, return_tensors="pt").input_ids.to(device)
        a = torch.ones_like(t)
        with torch.no_grad():
            out = model.generate(input_ids=t, attention_mask=a, max_new_tokens=512, do_sample=False)
        resp = tokenizer.decode(out[0, t.shape[1]:], skip_special_tokens=True)
        if answers_match(extract_boxed(resp), ex["answer"]):
            test_correct += 1
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/500: {test_correct}/{i+1} = {test_correct/(i+1):.4f}", flush=True)

    test_acc = test_correct / len(test_data)
    print(f"\nFINAL TEST ACCURACY: {test_acc:.4f} ({test_correct}/{len(test_data)})")

    # Save everything
    output = {
        "label": label, "method": method, "model": MODEL,
        "epochs": args.epochs, "effective_batch_size": effective_batch_size,
        "total_steps": global_step, "final_train_acc": acc,
        "test_accuracy": test_acc, "test_correct": test_correct,
        "time_seconds": elapsed,
        "hyperparams": {
            "lr": 5e-5, "ppo_epochs": 4, "clip_epsilon": 0.2,
            "vf_coef": 0.1, "gamma": 1.0, "lam": 0.95,
            "lora_r": 16, "lora_alpha": 32, "max_new_tokens": 512,
        },
        "history": history,
    }
    out_path = os.path.join(log_dir, f"train_math_full_{method_tag}.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    main()
