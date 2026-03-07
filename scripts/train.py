"""Training script for TI-PPO.

Usage:
    python scripts/train.py                                  # defaults (hybrid importance)
    python scripts/train.py --importance_method attention     # simpler alternative
    python scripts/train.py --importance_method uniform       # ablation: standard PPO
"""

import argparse
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model

from ti_ppo import TIPPOConfig, TIPPOTrainer, CausalLMWithValueHead


def parse_args():
    parser = argparse.ArgumentParser(description="Train TI-PPO")
    for field_name, field_obj in TIPPOConfig.__dataclass_fields__.items():
        ftype = field_obj.type
        if ftype == "bool":
            parser.add_argument(
                f"--{field_name}",
                type=lambda x: x.lower() == "true",
                default=field_obj.default,
            )
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
    model = CausalLMWithValueHead.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    if config.use_peft:
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model.pretrained_model = get_peft_model(model.pretrained_model, lora_config)
        model.pretrained_model.print_trainable_parameters()

    # Reference model (frozen copy, no value head needed)
    ref_model = CausalLMWithValueHead.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    for p in ref_model.parameters():
        p.requires_grad = False

    # Reward model
    print(f"Loading reward model: {config.reward_model_name}")
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        config.reward_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        num_labels=1,
    )
    reward_model.eval()
    for p in reward_model.parameters():
        p.requires_grad = False

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
    for epoch in range(100):
        for batch in dataloader:
            if episode >= config.total_episodes:
                break

            query_tensors = [ids.to(model.device) for ids in batch["input_ids"]]

            # Generate responses
            response_tensors = []
            for qt in query_tensors:
                out = model.generate(
                    input_ids=qt.unsqueeze(0), **generation_kwargs
                )
                resp = out[0, qt.shape[0] :]
                response_tensors.append(resp)

            # Get rewards from reward model
            rewards = trainer.get_rewards(query_tensors, response_tensors)

            # Filter out empty responses
            valid = [(q, r, s) for q, r, s in zip(query_tensors, response_tensors, rewards) if r.shape[0] > 0]
            if not valid:
                continue
            query_tensors, response_tensors, rewards = zip(*valid)
            query_tensors, response_tensors, rewards = list(query_tensors), list(response_tensors), list(rewards)

            # TI-PPO step
            stats = trainer.step(query_tensors, response_tensors, rewards)

            if episode % 10 == 0:
                print(
                    f"[Episode {episode:>5d}] "
                    f"reward={stats['ppo/mean_reward']:.3f}  "
                    f"kl={stats['ppo/mean_kl']:.4f}  "
                    f"policy_loss={stats['ppo/policy_loss']:.4f}  "
                    f"importance={stats['ti_ppo/mean_importance']:.3f}"
                )

            episode += 1

        if episode >= config.total_episodes:
            break

    # Save
    print(f"\nSaving model to {config.output_dir}/")
    if config.use_peft:
        model.pretrained_model.save_pretrained(config.output_dir)
    else:
        model.pretrained_model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
