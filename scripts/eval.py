"""Evaluate a TI-PPO trained model on generation quality.

Usage:
    python scripts/eval.py --model_path checkpoints/ --prompts_file prompts.txt
    python scripts/eval.py --model_path checkpoints/  # uses default prompts
"""

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from peft import PeftModel


DEFAULT_PROMPTS = [
    "Explain quantum computing in simple terms.",
    "What are the ethical concerns around AI?",
    "Write a Python function to find the longest common subsequence.",
    "How would you help someone who is feeling anxious?",
    "What is the capital of France and why is it historically significant?",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--base_model", type=str, default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--reward_model", type=str, default="weqweasdas/RM-Gemma-2B")
    parser.add_argument("--prompts_file", type=str, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    args = parser.parse_args()

    # Load prompts
    if args.prompts_file:
        with open(args.prompts_file) as f:
            prompts = [line.strip() for line in f if line.strip()]
    else:
        prompts = DEFAULT_PROMPTS

    # Load model
    print(f"Loading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map="auto"
    )
    try:
        model = PeftModel.from_pretrained(base_model, args.model_path)
        print("Loaded LoRA adapter.")
    except Exception:
        model = base_model
        print("Loaded full model (no adapter found).")

    model.eval()

    # Load reward model for scoring
    print(f"Loading reward model: {args.reward_model}")
    rm = AutoModelForSequenceClassification.from_pretrained(
        args.reward_model, torch_dtype=torch.bfloat16, device_map="auto", num_labels=1
    )
    rm.eval()

    # Generate and score
    print(f"\n{'='*60}")
    print(f"Evaluating on {len(prompts)} prompts")
    print(f"{'='*60}\n")

    all_scores = []
    for i, prompt in enumerate(prompts):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id,
            )

        response = tokenizer.decode(output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        # Score with reward model
        full_text = prompt + response
        rm_inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=512).to(rm.device)
        with torch.no_grad():
            rm_out = rm(**rm_inputs)
            score = rm_out.logits[0, 0].item() if rm_out.logits.dim() > 1 else rm_out.logits[0].item()

        all_scores.append(score)

        print(f"[Prompt {i+1}] {prompt}")
        print(f"[Response] {response[:300]}{'...' if len(response) > 300 else ''}")
        print(f"[Reward Score] {score:.4f}")
        print(f"{'-'*60}\n")

    print(f"{'='*60}")
    print(f"Average reward score: {sum(all_scores)/len(all_scores):.4f}")
    print(f"Min: {min(all_scores):.4f}  Max: {max(all_scores):.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
