from dataclasses import dataclass, field
from typing import Literal


@dataclass
class TIPPOConfig:
    """Configuration for TI-PPO training."""

    # Model
    model_name: str = "meta-llama/Llama-3.2-1B"
    reward_model_name: str = "weqweasdas/RM-Gemma-2B"
    use_peft: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    # Token importance
    importance_method: Literal[
        "hybrid", "gradient", "attention", "td_error", "reward_model", "uniform"
    ] = "hybrid"
    lambda_blend: float = 0.7  # blend between gradient and gaussian prior
    gaussian_sigma_scale: float = 4.0  # sigma = seq_len / this value
    importance_update_freq: int = 10  # recompute gradient importance every N steps
    importance_ema_decay: float = 0.9  # EMA smoothing for importance scores

    # Triplet loss
    use_triplet_loss: bool = True
    triplet_margin: float = 1.0
    triplet_gamma: float = 0.1  # weight of triplet loss in total objective

    # PPO
    ppo_epochs: int = 4
    mini_batch_size: int = 4
    batch_size: int = 16
    learning_rate: float = 1.41e-5
    max_grad_norm: float = 1.0
    clip_epsilon: float = 0.2
    vf_coef: float = 0.1
    gamma: float = 1.0
    lam: float = 0.95
    kl_penalty: str = "kl"
    target_kl: float = 6.0
    max_new_tokens: int = 128

    # Training
    total_episodes: int = 1000
    seed: int = 42
    output_dir: str = "checkpoints"
    log_with: str = "wandb"
    project_name: str = "ti-ppo"

    # Data
    dataset_name: str = "Anthropic/hh-rlhf"
    max_prompt_length: int = 256
