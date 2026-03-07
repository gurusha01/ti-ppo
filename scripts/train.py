"""Training script for TI-PPO.

Usage:
    python scripts/train.py                          # defaults (hybrid importance)
    python scripts/train.py --importance_method attention   # simpler alternative
    python scripts/train.py --importance_method uniform     # ablation: standard PPO
"""

import argparse
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from trl import AutoModelForCausalLMWithValueHead
from peft import LoraConfig

from ti_ppo import TIPPOConfig, TIPPOTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train TI-PPO")
    # Allow overriding any TIPPOConfig field
    for field_name, field_obj in TIPPOConfig.__dataclass_fields__.items():
        ftype = field_obj.type
        if ftype == "bool":
            parser.add_argument(f"--{field_name}", type=lambda x: x.lower() == "true", default=field_obj.default)
        elif ftype == "float":
            parser.add_argument(f"--{field_name}", type=float, default=field_obj.default)
        elif ftype == "int":
            parser.add_argument(f"--{field_name}", type=int, default=field_obj.default)
        elif ftype == "str" or "Literal" in str(ftype):
            parser.add_argument(f"--{field_name}", type=str, default=field_obj.default)
    return parser.parse_args()


def build_dataset(dataset_name, tokenizer, max_prompt_length, split="train"):
    """Load and tokenize a preference dataset, extracting prompts."""
    ds = load_dataset(dataset_name, split=split)

    def extract_prompt(example):
        # hh-rlhf format: "chosen" and "rejected" are full conversations
        # Extract the human turn as the prompt
        text = example.get("chosen", example.get("prompt", ""))
        if "\n\nHuman:" in text:
            prompt = text.split("\n\nAssistant:")[0] + "\n\nAssistant:"
        else:
            prompt = text[:200]

        tokens = tokenizer(
            prompt,
            truncation=True,
            max_length=max_prompt_length,
            padding=False,
            return_tensors=None,
        )
        tokens["query"] = prompt
        return tokens

    ds = ds.map(extract_prompt, remove_columns=ds.column_names)
    ds.set_format(type="torch")
    return ds


def collator(data):
    return {key: [d[key] for d in data] for key in data[0]}


def main():
    args = parse_args()
    config = TIPPOConfig(**{k: v for k, v in vars(args).items() if hasattr(TIPPOConfig, k)})

    print(f"=== TI-PPO Training ===")
    print(f"Model:       {config.model_name}")
    print(f"Importance:  {config.importance_method}")
    print(f"Triplet:     {config.use_triplet_loss}")
    print(f"Lambda:      {config.lambda_blend}")
    print(f"Episodes:    {config.total_episodes}")
    print()

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Policy model with value head
    lora_config = None
    if config.use_peft:
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )

    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        config.model_name,
        peft_config=lora_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Reference model (frozen copy)
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Reward model
    print(f"Loading reward model: {config.reward_model_name}")
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        config.reward_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        num_labels=1,
    )
    reward_tokenizer = AutoTokenizer.from_pretrained(config.reward_model_name)

    # Dataset
    print(f"Loading dataset: {config.dataset_name}")
    dataset = build_dataset(config.dataset_name, tokenizer, config.max_prompt_length)

    # Trainer
    trainer = TIPPOTrainer(
        config=config,
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        reward_model=reward_model,
        reward_tokenizer=reward_tokenizer,
    )

    # Training loop
    print(f"\nStarting training for {config.total_episodes} episodes...")
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collator,
    )

    generation_kwargs = {
        "max_new_tokens": config.max_new_tokens,
        "do_sample": True,
        "top_k": 50,
        "top_p": 0.95,
        "temperature": 0.7,
        "pad_token_id": tokenizer.pad_token_id,
    }

    episode = 0
    for epoch in range(100):  # outer loop; we break on total_episodes
        for batch in dataloader:
            if episode >= config.total_episodes:
                break

            query_tensors = [torch.tensor(ids) for ids in batch["input_ids"]]

            # Generate responses
            response_tensors = trainer.ppo_trainer.generate(
                query_tensors, return_prompt=False, **generation_kwargs
            )

            # Decode for logging
            batch_responses = [tokenizer.decode(r.squeeze(), skip_special_tokens=True) for r in response_tensors]

            # Get rewards from reward model
            rewards = trainer.get_rewards(query_tensors, response_tensors)

            # TI-PPO step
            stats = trainer.step(query_tensors, response_tensors, rewards)

            # Logging
            if episode % 10 == 0:
                mean_reward = torch.stack(rewards).mean().item()
                print(
                    f"[Episode {episode:>5d}] "
                    f"reward={mean_reward:.3f}  "
                    f"kl={stats.get('ppo/mean_kl', 0):.3f}  "
                    f"importance_mean={stats.get('ti_ppo/mean_importance', 0):.3f}  "
                    f"importance_std={stats.get('ti_ppo/importance_std', 0):.3f}"
                )

            episode += 1

        if episode >= config.total_episodes:
            break

    # Save
    print(f"\nSaving model to {config.output_dir}/")
    model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
