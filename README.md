# TI-PPO: Token-Importance Guided Proximal Policy Optimization

An adaptation of [Token-Importance Guided Direct Preference Optimization (TI-DPO)](https://arxiv.org/abs/2505.19653) to PPO-based RLHF. Instead of treating all tokens equally during policy optimization, TI-PPO weights the PPO objective by per-token importance scores, focusing learning on tokens that matter most for alignment.

## Core Idea

Standard PPO-RLHF applies the clipped surrogate objective uniformly across all tokens:

```
L_PPO = E[ min(r(t) * A(t), clip(r(t)) * A(t)) ]
```

TI-PPO adds token importance weighting:

```
L_TI-PPO = E[ w(t) * min(r(t) * A(t), clip(r(t)) * A(t)) ]
```

where `w(t)` is a per-token importance score. This reduces gradient variance by downweighting noisy/irrelevant tokens and concentrating the KL budget on critical ones.

## Token Importance Methods

### Hybrid (paper method, default)
Convex combination of gradient attribution and Gaussian prior, from TI-DPO:
- **Gradient attribution**: L1-norm of gradients w.r.t. token embeddings — captures each token's influence on model prediction
- **Gaussian prior**: Bell curve centered at mid-sequence — counteracts the "Lost-in-the-Middle" attention bias
- `W = lambda * gradient + (1 - lambda) * gaussian`

### Simpler Alternatives
| Method | How it works | Cost |
|---|---|---|
| `attention` | Average attention received per token across layers/heads | 1 forward pass |
| `td_error` | Absolute TD-error from value function — tokens where critic is most surprised | Free (uses existing values) |
| `reward_model` | Leave-one-out reward perturbation — measures per-token reward contribution | T forward passes (expensive) |
| `uniform` | All tokens weighted equally (standard PPO baseline) | Free |

### Triplet Loss (optional)
An auxiliary loss that pushes model outputs closer to preferred responses and farther from rejected ones in hidden-state space, providing structured guidance beyond scalar rewards.

## Project Structure

```
291K/
├── ti_ppo/
│   ├── __init__.py
│   ├── config.py              # TIPPOConfig dataclass
│   ├── token_importance.py    # All importance scoring methods
│   └── trainer.py             # TIPPOTrainer (wraps trl PPOTrainer)
├── scripts/
│   ├── train.py               # Main training script
│   ├── eval.py                # Evaluate trained model
│   └── compare_methods.py     # Compare importance methods side-by-side
├── pyproject.toml
└── README.md
```

## Setup

```bash
cd 291K
uv venv .venv
source .venv/bin/activate
uv pip install -e .
```

## Usage

### Train with hybrid importance (default)
```bash
python scripts/train.py \
    --model_name meta-llama/Llama-3.2-1B \
    --importance_method hybrid \
    --lambda_blend 0.7 \
    --total_episodes 1000 \
    --batch_size 8
```

### Train with attention-based importance (cheaper)
```bash
python scripts/train.py \
    --importance_method attention \
    --total_episodes 1000
```

### Train standard PPO (ablation baseline)
```bash
python scripts/train.py \
    --importance_method uniform \
    --use_triplet_loss false
```

### Compare importance methods
```bash
python scripts/compare_methods.py --model_name meta-llama/Llama-3.2-1B
```

### Evaluate
```bash
python scripts/eval.py --model_path checkpoints/ --base_model meta-llama/Llama-3.2-1B
```

## Key Configuration

| Parameter | Default | Description |
|---|---|---|
| `importance_method` | `hybrid` | Scoring method: hybrid, gradient, attention, td_error, reward_model, uniform |
| `lambda_blend` | `0.7` | Weight for gradient vs gaussian in hybrid mode |
| `use_triplet_loss` | `true` | Enable triplet loss auxiliary objective |
| `triplet_gamma` | `0.1` | Weight of triplet loss in total objective |
| `importance_update_freq` | `10` | Recompute gradient importance every N steps |
| `importance_ema_decay` | `0.9` | EMA smoothing for importance score stability |
| `use_peft` | `true` | Use LoRA for memory-efficient training |

## Why PPO benefits more than DPO

PPO already operates at the token level (per-token policy gradients) and suffers from high variance in gradient estimates. Token importance weighting directly reduces this variance — the theoretical guarantees from TI-DPO (variance reduction, tighter loss bounds) transfer naturally and arguably have a larger impact in the PPO setting.

## Reference

```bibtex
@article{yang2025tidpo,
  title={Token-Importance Guided Direct Preference Optimization},
  author={Yang, Ning and others},
  journal={arXiv preprint arXiv:2505.19653},
  year={2025}
}
```
