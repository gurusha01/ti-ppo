"""Compare different token importance methods on the same data.

Runs a quick benchmark showing how each scoring method distributes
importance weights, useful for understanding method behavior before
committing to a full training run.

Usage:
    python scripts/compare_methods.py --model_name meta-llama/Llama-3.2-1B
"""

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from ti_ppo.token_importance import (
    GradientImportance,
    GaussianPrior,
    HybridImportance,
    AttentionImportance,
    _UniformScorer,
)


SAMPLE_PROMPTS = [
    "The patient should seek immediate medical attention if symptoms persist.",
    "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
    "The capital of France is Paris, which is known for the Eiffel Tower.",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.float32, device_map="auto"
    )

    scorers = {
        "Uniform": _UniformScorer(),
        "Gaussian Prior": GaussianPrior(sigma_scale=4.0),
        "Gradient Attribution": GradientImportance(),
        "Attention-based": AttentionImportance(),
        "Hybrid (lambda=0.7)": HybridImportance(lambda_blend=0.7),
    }

    for prompt in SAMPLE_PROMPTS:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])

        print(f"\n{'='*80}")
        print(f"Prompt: {prompt}")
        print(f"Tokens: {len(tokens)}")
        print(f"{'='*80}")

        for name, scorer in scorers.items():
            kwargs = dict(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
            if name not in ("Uniform", "Gaussian Prior"):
                kwargs["model"] = model

            try:
                weights = scorer.score(**kwargs)
                w = weights[0].cpu().numpy()

                # Show top-5 and bottom-5 tokens by importance
                indices = w.argsort()
                top5 = indices[-5:][::-1]
                bot5 = indices[:5]

                print(f"\n  {name}:")
                print(f"    Mean={w.mean():.3f}  Std={w.std():.3f}  Min={w.min():.3f}  Max={w.max():.3f}")
                print(f"    Top-5: {', '.join(f'{tokens[i]}({w[i]:.2f})' for i in top5)}")
                print(f"    Bot-5: {', '.join(f'{tokens[i]}({w[i]:.2f})' for i in bot5)}")
            except Exception as e:
                print(f"\n  {name}: ERROR - {e}")

    print()


if __name__ == "__main__":
    main()
