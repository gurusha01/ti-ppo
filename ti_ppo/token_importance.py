"""Token importance scoring methods for TI-PPO.

Implements the hybrid weighting mechanism from TI-DPO (gradient attribution +
Gaussian prior) adapted for PPO, plus simpler alternatives: attention-based,
TD-error-based, and reward-model-based importance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from abc import ABC, abstractmethod


class TokenImportanceScorer(ABC):
    """Base class for token importance scoring."""

    @abstractmethod
    def score(self, **kwargs) -> torch.Tensor:
        """Return per-token importance weights of shape (batch, seq_len) in [0, 1]."""
        ...


# ---------------------------------------------------------------------------
# 1. Gradient Attribution (from TI-DPO Section 3.2)
# ---------------------------------------------------------------------------

class GradientImportance(TokenImportanceScorer):
    """Compute importance via L1-norm of gradients w.r.t. token embeddings.

    I_i = ||nabla_{e_i} L_target||_1 , normalized to [0, 1].
    """

    @torch.enable_grad()
    def score(self, model, input_ids, attention_mask=None, **kwargs) -> torch.Tensor:
        embeddings = model.get_input_embeddings()
        embeds = embeddings(input_ids)  # (B, T, D)
        embeds = embeds.detach().requires_grad_(True)

        # Forward through the model with embeddings directly
        outputs = model(inputs_embeds=embeds, attention_mask=attention_mask)
        logits = outputs.logits  # (B, T, V)

        # Target: max logit at the last real token per sequence
        if attention_mask is not None:
            last_idx = attention_mask.sum(dim=1) - 1  # (B,)
        else:
            last_idx = torch.full(
                (input_ids.shape[0],), input_ids.shape[1] - 1, device=input_ids.device
            )

        batch_idx = torch.arange(logits.size(0), device=logits.device)
        last_logits = logits[batch_idx, last_idx]  # (B, V)
        target = last_logits.max(dim=-1).values.sum()

        target.backward()

        # L1 norm per token
        grad = embeds.grad  # (B, T, D)
        importance = grad.abs().sum(dim=-1)  # (B, T)

        # Normalize to [0, 1] per sequence
        importance = _min_max_normalize(importance, attention_mask)
        return importance.detach()


# ---------------------------------------------------------------------------
# 2. Gaussian Prior (from TI-DPO Section 3.2)
# ---------------------------------------------------------------------------

class GaussianPrior(TokenImportanceScorer):
    """Gaussian prior centered at the middle of the sequence.

    Counteracts "Lost-in-the-Middle" bias.
    P_prior(t) = exp(-0.5 * ((t - mu) / sigma)^2)
    mu = (T-1)/2, sigma = T / sigma_scale
    """

    def __init__(self, sigma_scale: float = 4.0):
        self.sigma_scale = sigma_scale

    def score(self, input_ids, attention_mask=None, **kwargs) -> torch.Tensor:
        B, T = input_ids.shape
        device = input_ids.device

        positions = torch.arange(T, device=device, dtype=torch.float32)

        if attention_mask is not None:
            seq_lens = attention_mask.sum(dim=1, keepdim=True).float()  # (B, 1)
        else:
            seq_lens = torch.full((B, 1), T, device=device, dtype=torch.float32)

        mu = (seq_lens - 1) / 2  # (B, 1)
        sigma = seq_lens / self.sigma_scale  # (B, 1)

        prior = torch.exp(-0.5 * ((positions.unsqueeze(0) - mu) / sigma) ** 2)  # (B, T)

        if attention_mask is not None:
            prior = prior * attention_mask.float()

        return prior


# ---------------------------------------------------------------------------
# 3. Hybrid = lambda * Gradient + (1 - lambda) * Gaussian (TI-DPO Eq. 5)
# ---------------------------------------------------------------------------

class HybridImportance(TokenImportanceScorer):
    """Hybrid weighting: convex combination of gradient attribution and Gaussian prior."""

    def __init__(self, lambda_blend: float = 0.7, sigma_scale: float = 4.0):
        self.lambda_blend = lambda_blend
        self.gradient_scorer = GradientImportance()
        self.gaussian_scorer = GaussianPrior(sigma_scale=sigma_scale)

    def score(self, model, input_ids, attention_mask=None, **kwargs) -> torch.Tensor:
        grad_scores = self.gradient_scorer.score(
            model=model, input_ids=input_ids, attention_mask=attention_mask
        )
        gauss_scores = self.gaussian_scorer.score(
            input_ids=input_ids, attention_mask=attention_mask
        )
        weights = self.lambda_blend * grad_scores + (1 - self.lambda_blend) * gauss_scores
        return weights


# ---------------------------------------------------------------------------
# Simpler alternatives
# ---------------------------------------------------------------------------

class AttentionImportance(TokenImportanceScorer):
    """Use average attention weight received by each token as importance proxy.

    Cheap and architecture-aware: tokens the model attends to more are weighted higher.
    """

    @torch.no_grad()
    def score(self, model, input_ids, attention_mask=None, **kwargs) -> torch.Tensor:
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
        )
        # attentions: tuple of (B, num_heads, T, T) per layer
        # Average over layers and heads, then sum over query positions
        # -> how much total attention each key token receives
        attn_stack = torch.stack(outputs.attentions, dim=0)  # (L, B, H, T, T)
        avg_attn = attn_stack.mean(dim=(0, 2))  # (B, T_query, T_key)
        importance = avg_attn.sum(dim=1)  # (B, T_key) — total attention received

        importance = _min_max_normalize(importance, attention_mask)
        return importance


class TDErrorImportance(TokenImportanceScorer):
    """Use |TD-error| from the value function as importance.

    Tokens where the value function is most surprised are most important.
    Requires precomputed values and rewards.
    """

    def __init__(self, gamma: float = 1.0, lam: float = 0.95):
        self.gamma = gamma
        self.lam = lam

    def score(self, values, rewards, attention_mask=None, **kwargs) -> torch.Tensor:
        """
        Args:
            values: (B, T) value estimates from the critic
            rewards: (B, T) per-token rewards (usually 0 except last token)
        """
        B, T = values.shape
        td_errors = torch.zeros_like(values)

        for t in reversed(range(T - 1)):
            td_errors[:, t] = (
                rewards[:, t] + self.gamma * values[:, t + 1] - values[:, t]
            ).abs()

        # Last token: just the reward residual
        td_errors[:, -1] = (rewards[:, -1] - values[:, -1]).abs()

        importance = _min_max_normalize(td_errors, attention_mask)
        return importance


class RewardModelImportance(TokenImportanceScorer):
    """Compute importance via per-token reward model score differences.

    For each token position, measure how much removing it changes the
    reward model's score (leave-one-out). Approximated via a single
    forward pass with causal masking perturbation.
    """

    @torch.no_grad()
    def score(
        self, reward_model, input_ids, attention_mask=None, **kwargs
    ) -> torch.Tensor:
        B, T = input_ids.shape
        device = input_ids.device

        # Baseline reward
        base_output = reward_model(input_ids=input_ids, attention_mask=attention_mask)
        if hasattr(base_output, "logits"):
            base_score = base_output.logits.squeeze(-1)  # (B,) or (B, T)
        else:
            base_score = base_output[0].squeeze(-1)

        if base_score.dim() == 1:
            base_score = base_score.unsqueeze(1).expand(B, T)

        # Approximate leave-one-out via embedding perturbation:
        # Replace each token embedding with zeros and measure reward change
        embeddings = reward_model.get_input_embeddings()
        embeds = embeddings(input_ids)  # (B, T, D)

        importance = torch.zeros(B, T, device=device)
        for t in range(T):
            perturbed = embeds.clone()
            perturbed[:, t, :] = 0.0
            out = reward_model(inputs_embeds=perturbed, attention_mask=attention_mask)
            if hasattr(out, "logits"):
                perturbed_score = out.logits.squeeze(-1)
            else:
                perturbed_score = out[0].squeeze(-1)

            if perturbed_score.dim() == 2:
                perturbed_score = perturbed_score[:, -1]
            if base_score.dim() == 2:
                diff = (base_score[:, -1] - perturbed_score).abs()
            else:
                diff = (base_score - perturbed_score).abs()
            importance[:, t] = diff

        importance = _min_max_normalize(importance, attention_mask)
        return importance


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _min_max_normalize(x: torch.Tensor, mask=None) -> torch.Tensor:
    """Normalize tensor to [0, 1] per row, respecting mask."""
    if mask is not None:
        x = x.masked_fill(~mask.bool(), float("-inf"))

    x_min = x.min(dim=-1, keepdim=True).values
    x_max = x.max(dim=-1, keepdim=True).values
    denom = (x_max - x_min).clamp(min=1e-8)
    out = (x - x_min) / denom

    if mask is not None:
        out = out * mask.float()
    return out


def build_scorer(config) -> TokenImportanceScorer:
    """Factory to build a scorer from config."""
    method = config.importance_method
    if method == "hybrid":
        return HybridImportance(
            lambda_blend=config.lambda_blend,
            sigma_scale=config.gaussian_sigma_scale,
        )
    elif method == "gradient":
        return GradientImportance()
    elif method == "attention":
        return AttentionImportance()
    elif method == "td_error":
        return TDErrorImportance(gamma=config.gamma, lam=config.lam)
    elif method == "reward_model":
        return RewardModelImportance()
    elif method == "uniform":
        return _UniformScorer()
    else:
        raise ValueError(f"Unknown importance method: {method}")


class _UniformScorer(TokenImportanceScorer):
    """Baseline: all tokens weighted equally."""

    def score(self, input_ids, attention_mask=None, **kwargs) -> torch.Tensor:
        weights = torch.ones_like(input_ids, dtype=torch.float32)
        if attention_mask is not None:
            weights = weights * attention_mask.float()
        return weights
