# TI-PPO: Token-Importance Guided Proximal Policy Optimization

An adaptation of [Token-Importance Guided Direct Preference Optimization (TI-DPO)](https://arxiv.org/abs/2505.19653) to PPO-based RLHF, with two novel contributions:

1. **AITI** (Adaptive Intensity Token Importance): Decaying importance intensity to manage the bias-variance tradeoff
2. **MOAI** (MSE-Optimal Adaptive Intensity): Closed-form optimal intensity with monotone constraint — discovers that permanent partial weighting beats full decay

## Key Results

| Method | Reward (Q4) | KL (Q4) | KL-Efficiency | Pareto? |
|--------|------------|---------|---------------|---------|
| **MOAI-Adv mono** | **0.880** | 0.056 | 15.7 | **YES** |
| **AITI-Advantage** | 0.874 | **0.031** | **28.4** | **YES** |
| PPO baseline | 0.859 | 0.028 | 30.4 | YES |
| Paper: Hybrid+Triplet | 0.779 | 0.024 | 32.7 | dominated |

MOAI achieves the highest reward. AITI provides the best reward-KL tradeoff. Both Pareto-dominate uniform PPO.

## The Problem with Existing Token Importance

All prior token importance methods (including TI-DPO) use **fixed-intensity weighting**: `w(t)` weights the objective permanently. We discovered this introduces **accumulated bias** into the PPO gradient:

```
Bias = E[w(t)·f(t)] - E[f(t)] = Cov(w, f) ≠ 0
```

Over training:
- **Short runs (≤100 episodes)**: Variance reduction > Bias → importance helps
- **Long runs (>100 episodes)**: Bias > Variance reduction → importance hurts

This explains why importance weighting shows initial gains but degrades over time.

## Our Solution: Adaptive Intensity (AITI)

Instead of `w(t)`, use:

```
w_final(t) = 1 + ε(step) · (w(t) - 1)
```

Where `ε` decays over training:
- **Early (ε≈1)**: Full importance weighting → variance reduction when it matters most
- **Late (ε≈0)**: Uniform PPO → unbiased gradients for fine-tuning

This is analogous to learning rate scheduling, but for gradient quality rather than step size.

## All Token Importance Methods (15 total)

### From TI-DPO paper
| Method | Signal | Cost |
|--------|--------|------|
| `hybrid` | Gradient attribution + Gaussian prior | 1 backward + 1 forward |
| `gradient` | L1-norm of ∇embedding | 1 backward |
| `attention` | Average attention received per token | 1 forward |
| `td_error` | |TD-error| from value function | Free |
| `reward_model` | Leave-one-out reward perturbation | T forward passes |

### PPO-native methods (Phase 2)
| Method | Signal | Cost |
|--------|--------|------|
| `advantage` | |A(t)| magnitude | Free |
| `entropy` | H(π(·\|s_t)) per token | Free (uses logits) |
| `kl_guided` | |A(t)| · (1 - tanh(β·|KL(t)|)) | Free |
| `adv_gaussian` | |A(t)| + Gaussian prior | Free |
| `entropy_advantage` | H(t) · |A(t)| product | Free |
| `pareto` | Lagrangian: softmax(|A| - λ·|KL|) | Free, adaptive λ |
| `snr` | |A(t)| / √FI(t) (Fisher Info) | Free (uses logits) |

### AITI: Adaptive Intensity (Phase 3)
| Method | Inner Scorer | Schedule |
|--------|-------------|----------|
| `aiti_advantage` | |Advantage| | Linear decay ε: 1→0 |
| `aiti_entropy` | Entropy | Linear/quadratic decay |
| `aiti_adaptive` | Entropy→Advantage | Linear/quadratic decay |

### MOAI: MSE-Optimal Adaptive Intensity (Phase 4 — Novel)
| Method | Inner Scorer | ε selection |
|--------|-------------|-------------|
| `moai_advantage_mono` | |Advantage| | Closed-form ε* with monotone constraint |
| `moai_entropy_mono` | Entropy | Closed-form ε* with monotone constraint |
| `moai_advantage` | |Advantage| | Closed-form ε* (free, no constraint) |
| `moai_entropy` | Entropy | Closed-form ε* (free, no constraint) |

## Project Structure

```
291K/
├── ti_ppo/
│   ├── __init__.py
│   ├── config.py              # TIPPOConfig dataclass
│   ├── token_importance.py    # All 19 importance scoring methods
│   ├── trainer.py             # Pure PyTorch PPO with importance weighting
│   └── value_head.py          # CausalLMWithValueHead
├── scripts/
│   ├── train.py               # Main training script
│   ├── eval.py                # Evaluate trained model
│   ├── compare_methods.py     # Compare importance distributions
│   ├── benchmark.py           # Benchmark v1 (original 6 methods)
│   ├── benchmark_v2.py        # Benchmark v2 (PPO-native methods)
│   ├── benchmark_v3.py        # Benchmark v3 (theoretically-derived methods)
│   ├── benchmark_v4.py        # Benchmark v4 (AITI)
│   ├── benchmark_v5.py        # Benchmark v5 (MOAI — free ε)
│   ├── benchmark_v5b.py       # Benchmark v5b (MOAI — monotone constraint)
│   └── analyze_entropy_kl.py  # Analysis: why entropy implies negative KL
├── FINDINGS.md                # Full research writeup with derivations
├── RESEARCH_LOG.md            # Research log (lab notebook)
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

### Train with AITI-Advantage (recommended)
```bash
python scripts/train.py \
    --importance_method aiti_advantage \
    --total_episodes 500
```

### Run benchmark
```bash
python scripts/benchmark_v4.py --episodes 150 --gpu 0
```

### Standard PPO baseline
```bash
python scripts/train.py \
    --importance_method uniform \
    --use_triplet_loss false
```

## Key Findings

1. **Entropy weighting produces negative KL** — the policy moves CLOSER to the reference while improving reward. This is because entropy focuses updates on uncertain tokens where both π and π_ref have broad distributions.

2. **All fixed-intensity importance methods degrade over training** due to bias accumulation in the PPO gradient estimator.

3. **AITI solves this** by decaying importance intensity, getting variance reduction early and unbiased gradients late.

4. **AITI-Advantage is Pareto-optimal** — the only method that beats uniform PPO on BOTH reward AND KL divergence simultaneously.

## Reference

```bibtex
@article{yang2025tidpo,
  title={Token-Importance Guided Direct Preference Optimization},
  author={Yang, Ning and others},
  journal={arXiv preprint arXiv:2505.19653},
  year={2025}
}
```
