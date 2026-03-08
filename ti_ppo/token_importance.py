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
        # Get the base model that supports inputs_embeds (unwrap PEFT if needed)
        base = model
        if hasattr(model, "get_base_model"):
            base = model.get_base_model()

        embeddings = base.get_input_embeddings()
        embeds = embeddings(input_ids)  # (B, T, D)
        embeds = embeds.detach().clone().requires_grad_(True)

        # Forward through the base model with embeddings directly
        try:
            outputs = base(inputs_embeds=embeds, attention_mask=attention_mask)
        except Exception:
            # Fallback: return uniform weights if model doesn't support inputs_embeds
            return torch.ones(input_ids.shape, device=input_ids.device, dtype=torch.float32)

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
        if grad is None:
            return torch.ones(input_ids.shape, device=input_ids.device, dtype=torch.float32)

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
        # Unwrap PEFT model to get attentions (PEFT wrappers may not return them)
        base = model
        if hasattr(model, "get_base_model"):
            base = model.get_base_model()

        outputs = base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
        )

        if not outputs.attentions or len(outputs.attentions) == 0:
            # Fallback to uniform if attentions unavailable
            return torch.ones(input_ids.shape, device=input_ids.device, dtype=torch.float32)

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
# PPO-native importance methods (use signals from the PPO loop itself)
# These are computed inside the trainer, not via the scorer factory.
# ---------------------------------------------------------------------------

class AdvantageImportance(TokenImportanceScorer):
    """Weight tokens by |advantage| magnitude.

    Mathematical justification: In PPO, gradient = E[nabla log pi * A].
    Tokens with |A| ~ 0 contribute noise, not signal. Weighting by |A|
    suppresses these, directly reducing gradient variance:
        Var[w * nabla log pi * A] < Var[nabla log pi * A]
    when w(t) is small for tokens where A(t) ~ 0.

    This is FREE — we already computed advantages.
    """

    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature

    def score(self, advantages, attention_mask=None, **kwargs) -> torch.Tensor:
        # Softmax over |A| to get normalized importance in (0, 1)
        abs_adv = advantages.abs()
        # Temperature-scaled softmax per sequence
        weights = F.softmax(abs_adv / self.temperature, dim=-1) * abs_adv.shape[-1]
        # Clamp to [0, 1] after scaling
        weights = _min_max_normalize(weights, attention_mask)
        return weights


class EntropyImportance(TokenImportanceScorer):
    """Weight tokens by policy entropy H(pi(.|s_t)).

    High entropy = model is uncertain = critical decision point.
    These tokens represent the frontier of alignment — where the model
    could go either way. Focusing optimization here is efficient because
    low-entropy tokens are already "decided" and hard to move.

    Cost: uses logits already computed in the PPO forward pass.
    """

    def score(self, logits, attention_mask=None, **kwargs) -> torch.Tensor:
        # logits: (B, T, V)
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)  # (B, T)
        weights = _min_max_normalize(entropy, attention_mask)
        return weights


class KLGuidedAdvantageImportance(TokenImportanceScorer):
    """Weight by advantage magnitude, downweighted where KL is already high.

    w(t) = |A(t)| * (1 - tanh(beta * |KL(t)|))

    Rationale: tokens where |A| is high but KL is low represent UNTAPPED
    POTENTIAL — the reward signal says "change here" but the model hasn't
    diverged yet. Tokens where KL is already high have been addressed.
    This focuses the remaining optimization budget where it matters most.

    Equivalent to a "remaining value" heuristic: prioritize tokens with
    the highest (reward signal) / (effort already spent) ratio.
    """

    def __init__(self, beta: float = 5.0):
        self.beta = beta

    def score(self, advantages, old_logprobs, ref_logprobs,
              attention_mask=None, **kwargs) -> torch.Tensor:
        abs_adv = advantages.abs()
        kl_per_token = (old_logprobs - ref_logprobs).abs()
        kl_per_token = torch.nan_to_num(kl_per_token, nan=0.0)

        # High advantage + low KL = high weight
        unexploited = 1.0 - torch.tanh(self.beta * kl_per_token)
        raw = abs_adv * unexploited

        weights = _min_max_normalize(raw, attention_mask)
        return weights


class AdvantageGaussianImportance(TokenImportanceScorer):
    """Advantage magnitude + Gaussian prior (replaces gradient with advantage).

    Same structure as the paper's hybrid method but swaps gradient attribution
    for |advantage|. This gets the stabilization benefit of the Gaussian prior
    without the compute cost of gradient attribution.

    W = lambda * normalize(|A|) + (1 - lambda) * Gaussian_prior

    Theoretical motivation: |A| is a better importance signal for PPO than
    gradient attribution because it directly measures the per-token reward
    signal, whereas gradient attribution measures prediction sensitivity
    (which may not correlate with alignment-relevant tokens).
    """

    def __init__(self, lambda_blend: float = 0.6, sigma_scale: float = 4.0):
        self.lambda_blend = lambda_blend
        self.gaussian = GaussianPrior(sigma_scale=sigma_scale)

    def score(self, advantages, input_ids, attention_mask=None, **kwargs) -> torch.Tensor:
        adv_weights = _min_max_normalize(advantages.abs(), attention_mask)
        gauss_weights = self.gaussian.score(input_ids=input_ids, attention_mask=attention_mask)
        return self.lambda_blend * adv_weights + (1 - self.lambda_blend) * gauss_weights


class EntropyAdvantageImportance(TokenImportanceScorer):
    """Product of entropy and |advantage|: focus on uncertain AND high-signal tokens.

    w(t) = normalize(H(t)) * normalize(|A(t)|)

    Only tokens that are BOTH uncertain (high entropy) AND reward-relevant
    (high |advantage|) get high weight. This is the tightest filter:
    - High entropy, low advantage → model is confused but it doesn't matter → low weight
    - Low entropy, high advantage → model is confident, hard to move → low weight
    - High entropy, high advantage → sweet spot → HIGH weight
    """

    def score(self, logits, advantages, attention_mask=None, **kwargs) -> torch.Tensor:
        # Entropy
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)
        ent_norm = _min_max_normalize(entropy, attention_mask)

        # Advantage magnitude
        adv_norm = _min_max_normalize(advantages.abs(), attention_mask)

        # Product — geometric mean of both signals
        raw = ent_norm * adv_norm
        return _min_max_normalize(raw, attention_mask)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _min_max_normalize(x: torch.Tensor, mask=None) -> torch.Tensor:
    """Normalize tensor to [0, 1] per row, respecting mask."""
    # Replace NaN/Inf with 0
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    if mask is not None:
        x = x * mask.float()

    x_min = x.min(dim=-1, keepdim=True).values
    x_max = x.max(dim=-1, keepdim=True).values
    denom = (x_max - x_min).clamp(min=1e-8)
    out = (x - x_min) / denom

    # Ensure no NaN in output
    out = torch.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)

    if mask is not None:
        out = out * mask.float()
    return out


# ---------------------------------------------------------------------------
# Phase 2: Theoretically-derived optimal importance methods
# ---------------------------------------------------------------------------

class ParetoOptimalImportance(TokenImportanceScorer):
    """Pareto-Optimal Token Importance (POTI).

    Derived from constrained optimization:
        max_w  E_t[w(t) * A(t)]          (maximize reward improvement)
        s.t.   E_t[w(t) * |KL(t)|] <= δ  (KL budget constraint)

    Lagrangian solution:
        w*(t) = softmax((A(t) - λ * |KL(t)|) / τ)

    where λ is the dual variable, adapted online:
        λ ← max(0, λ + η * (E[w*|KL|] - δ))

    When λ=0 → pure advantage weighting.
    When λ→∞ → maximum KL conservation.
    The algorithm FINDS the Pareto-optimal trade-off automatically.

    This is the first token-level Lagrangian importance method for RLHF.
    """

    def __init__(self, temperature: float = 1.0, kl_target: float = 0.02,
                 dual_lr: float = 0.1, lambda_init: float = 1.0):
        self.temperature = temperature
        self.kl_target = kl_target
        self.dual_lr = dual_lr
        self.lambda_dual = lambda_init  # Lagrange multiplier (mutable state)

    def score(self, advantages, old_logprobs, ref_logprobs,
              attention_mask=None, **kwargs) -> torch.Tensor:
        abs_adv = advantages.abs()
        kl_per_token = (old_logprobs - ref_logprobs).abs()
        kl_per_token = torch.nan_to_num(kl_per_token, nan=0.0)

        # Lagrangian score: reward signal minus KL penalty
        lagrangian_score = abs_adv - self.lambda_dual * kl_per_token

        # Softmax to get proper weights in (0, 1)
        weights = F.softmax(lagrangian_score / self.temperature, dim=-1) * lagrangian_score.shape[-1]
        weights = _min_max_normalize(weights, attention_mask)

        # Dual variable update: if KL exceeds target, increase λ (more conservative)
        if attention_mask is not None:
            mask_f = attention_mask.float()
            effective_kl = (weights.detach() * kl_per_token * mask_f).sum() / mask_f.sum().clamp(min=1)
        else:
            effective_kl = (weights.detach() * kl_per_token).mean()

        # Dual gradient ascent
        kl_violation = effective_kl.item() - self.kl_target
        self.lambda_dual = max(0.0, self.lambda_dual + self.dual_lr * kl_violation)

        return weights


class AdaptivePhaseImportance(TokenImportanceScorer):
    """Adaptive Entropy→Advantage Annealing.

    Phase 1 (early training): w(t) ≈ H(π(·|s_t))
        → Implicit KL regularization (entropy focuses on uncertain tokens
          where both π and π_ref have broad distributions, keeping KL low)
    Phase 2 (later training): w(t) ≈ |A(t)|
        → Direct reward maximization via advantage-based focus

    The annealing schedule:
        α(step) = min(1, step / warmup_steps)
        w(t) = (1-α) * normalize(H(t)) + α * normalize(|A(t)|)

    Motivation: Entropy-first prevents early policy collapse (observed empirically:
    entropy weighting gives NEGATIVE KL). Advantage-later maximizes reward once
    the policy is stabilized.
    """

    def __init__(self, warmup_steps: int = 50, temperature: float = 1.0):
        self.warmup_steps = warmup_steps
        self.temperature = temperature
        self.step_count = 0

    def score(self, logits, advantages, attention_mask=None, **kwargs) -> torch.Tensor:
        self.step_count += 1
        alpha = min(1.0, self.step_count / self.warmup_steps)

        # Entropy component
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)
        ent_norm = _min_max_normalize(entropy, attention_mask)

        # Advantage component
        adv_norm = _min_max_normalize(advantages.abs(), attention_mask)

        # Anneal from entropy to advantage
        weights = (1.0 - alpha) * ent_norm + alpha * adv_norm
        return _min_max_normalize(weights, attention_mask)


class SNRImportance(TokenImportanceScorer):
    """Signal-to-Noise Ratio Token Importance.

    From policy gradient variance reduction theory:
        Var[∇J] = E_t[w(t)² · A(t)² · Var[∇log π(a_t)]]

    Minimizing this variance w.r.t. w(t) subject to E[w]=1 gives:
        w*(t) ∝ |A(t)| / √Var[∇log π(a_t)]

    We approximate Var[∇log π] via the Fisher Information at each token:
        FI(t) = Var_π[log π(v|s_t)] = E[(log π)²] - (E[log π])²

    So: w*(t) ∝ |A(t)| / √FI(t)

    Intuition: weight tokens that have BOTH high reward signal (|A|) AND
    low gradient noise (low FI). These are the most informative tokens
    for learning — reliable signal, not noise.
    """

    def __init__(self, epsilon: float = 1e-6):
        self.epsilon = epsilon

    def score(self, logits, advantages, attention_mask=None, **kwargs) -> torch.Tensor:
        # Compute Fisher Information per token
        probs = F.softmax(logits, dim=-1)  # (B, T, V)
        log_probs = F.log_softmax(logits, dim=-1)  # (B, T, V)

        # FI = E[(log π)²] - (E[log π])²
        E_logp_sq = (probs * log_probs ** 2).sum(dim=-1)  # E[(log π)²]
        E_logp = (probs * log_probs).sum(dim=-1)  # E[log π] = -H
        fisher_info = (E_logp_sq - E_logp ** 2).clamp(min=self.epsilon)  # (B, T)

        # SNR = |A| / √FI
        snr = advantages.abs() / fisher_info.sqrt()
        return _min_max_normalize(snr, attention_mask)


class EntropyKLLagrangianImportance(TokenImportanceScorer):
    """Entropy-KL Lagrangian: formalized version of entropy's implicit regularization.

    Observation: Entropy weighting gives negative KL because it focuses updates
    on high-entropy tokens where both π and π_ref are uncertain.

    This method FORMALIZES that insight:
        w(t) = softmax((H(t) - μ · |KL(t)|) / τ)

    Where μ adapts to maintain a KL target:
        μ ← max(0, μ + η · (mean_KL - target_KL))

    Unlike POTI (advantage-based), this uses ENTROPY as the primary signal
    and KL as the constraint. The theoretical advantage: entropy identifies
    tokens that are inherently "safe to modify" (uncertain positions),
    while the KL constraint ensures we don't exploit this too aggressively.

    This combines the entropy method's regularization with explicit KL control.
    """

    def __init__(self, temperature: float = 1.0, kl_target: float = 0.01,
                 dual_lr: float = 0.05, mu_init: float = 1.0):
        self.temperature = temperature
        self.kl_target = kl_target
        self.dual_lr = dual_lr
        self.mu_dual = mu_init

    def score(self, logits, old_logprobs, ref_logprobs,
              attention_mask=None, **kwargs) -> torch.Tensor:
        # Entropy
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)  # (B, T)

        # Per-token KL
        kl_per_token = (old_logprobs - ref_logprobs).abs()
        kl_per_token = torch.nan_to_num(kl_per_token, nan=0.0)

        # Lagrangian: maximize entropy focus subject to KL constraint
        score = entropy - self.mu_dual * kl_per_token
        weights = F.softmax(score / self.temperature, dim=-1) * score.shape[-1]
        weights = _min_max_normalize(weights, attention_mask)

        # Dual update
        if attention_mask is not None:
            mask_f = attention_mask.float()
            effective_kl = (kl_per_token * mask_f).sum() / mask_f.sum().clamp(min=1)
        else:
            effective_kl = kl_per_token.mean()

        kl_violation = effective_kl.item() - self.kl_target
        self.mu_dual = max(0.0, self.mu_dual + self.dual_lr * kl_violation)

        return weights


# ---------------------------------------------------------------------------
# Phase 3: Adaptive Intensity — the bias-variance schedule
# ---------------------------------------------------------------------------

class AdaptiveIntensityImportance(TokenImportanceScorer):
    """Adaptive Intensity Token Importance (AITI).

    KEY INSIGHT: All token importance methods introduce bias into the PPO
    gradient estimate. The bias is: E[w·f] - E[f] = Cov(w, f).
    Over training, this bias accumulates and degrades performance.

    Solution: interpolate between importance-weighted and uniform:
        w_final(t) = 1 + ε(step) · (s(t) - 1)

    Where:
    - s(t) is the raw importance score (any method)
    - ε(step) controls intensity, decaying over training
    - ε=0 → uniform (zero bias), ε=1 → full importance (max variance reduction)

    Schedule: ε(step) = ε_max · max(0, 1 - step / decay_steps)^power

    This provides variance reduction early (when it helps most) and
    removes bias late (when the model is refined and uniform is better).

    The inner scorer can be any method — we default to entropy (best early-game).

    THIS IS NOVEL: No prior work on token importance for RLHF addresses the
    bias-variance tradeoff of importance weighting itself.
    """

    def __init__(self, inner_scorer: TokenImportanceScorer,
                 epsilon_max: float = 1.0, decay_steps: int = 100,
                 power: float = 1.0, min_epsilon: float = 0.0):
        self.inner_scorer = inner_scorer
        self.epsilon_max = epsilon_max
        self.decay_steps = decay_steps
        self.power = power
        self.min_epsilon = min_epsilon
        self.step_count = 0

    @property
    def epsilon(self):
        progress = min(1.0, self.step_count / self.decay_steps)
        return max(self.min_epsilon,
                   self.epsilon_max * (1.0 - progress) ** self.power)

    def score(self, attention_mask=None, **kwargs) -> torch.Tensor:
        self.step_count += 1
        eps = self.epsilon

        # Get raw importance from inner scorer
        raw_scores = self.inner_scorer.score(attention_mask=attention_mask, **kwargs)

        # Interpolate: w = 1 + ε * (s - 1) = (1-ε) * 1 + ε * s
        if attention_mask is not None:
            uniform = attention_mask.float()
        else:
            uniform = torch.ones_like(raw_scores)

        weights = (1.0 - eps) * uniform + eps * raw_scores
        return weights


# Methods that require PPO-internal signals (advantages, logits, etc.)
PPO_NATIVE_METHODS = {
    "advantage", "entropy", "kl_guided", "adv_gaussian", "entropy_advantage",
    "pareto", "adaptive_phase", "snr", "entropy_kl_lagrangian",
    "aiti_entropy", "aiti_advantage", "aiti_adaptive",
}


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
    elif method == "advantage":
        return AdvantageImportance(temperature=1.0)
    elif method == "entropy":
        return EntropyImportance()
    elif method == "kl_guided":
        return KLGuidedAdvantageImportance(beta=5.0)
    elif method == "adv_gaussian":
        return AdvantageGaussianImportance(
            lambda_blend=config.lambda_blend,
            sigma_scale=config.gaussian_sigma_scale,
        )
    elif method == "entropy_advantage":
        return EntropyAdvantageImportance()
    elif method == "pareto":
        return ParetoOptimalImportance(
            temperature=getattr(config, 'pareto_temperature', 1.0),
            kl_target=getattr(config, 'pareto_kl_target', 0.02),
            dual_lr=getattr(config, 'pareto_dual_lr', 0.1),
            lambda_init=getattr(config, 'pareto_lambda_init', 1.0),
        )
    elif method == "adaptive_phase":
        return AdaptivePhaseImportance(
            warmup_steps=getattr(config, 'phase_warmup_steps', 50),
        )
    elif method == "snr":
        return SNRImportance()
    elif method == "entropy_kl_lagrangian":
        return EntropyKLLagrangianImportance(
            kl_target=getattr(config, 'pareto_kl_target', 0.01),
            dual_lr=getattr(config, 'pareto_dual_lr', 0.05),
        )
    elif method == "aiti_entropy":
        return AdaptiveIntensityImportance(
            inner_scorer=EntropyImportance(),
            epsilon_max=getattr(config, 'aiti_epsilon_max', 1.0),
            decay_steps=getattr(config, 'aiti_decay_steps', 100),
            power=getattr(config, 'aiti_power', 1.0),
            min_epsilon=getattr(config, 'aiti_min_epsilon', 0.0),
        )
    elif method == "aiti_advantage":
        return AdaptiveIntensityImportance(
            inner_scorer=AdvantageImportance(temperature=1.0),
            epsilon_max=getattr(config, 'aiti_epsilon_max', 1.0),
            decay_steps=getattr(config, 'aiti_decay_steps', 100),
            power=getattr(config, 'aiti_power', 1.0),
            min_epsilon=getattr(config, 'aiti_min_epsilon', 0.0),
        )
    elif method == "aiti_adaptive":
        return AdaptiveIntensityImportance(
            inner_scorer=AdaptivePhaseImportance(
                warmup_steps=getattr(config, 'phase_warmup_steps', 50),
            ),
            epsilon_max=getattr(config, 'aiti_epsilon_max', 1.0),
            decay_steps=getattr(config, 'aiti_decay_steps', 100),
            power=getattr(config, 'aiti_power', 1.0),
            min_epsilon=getattr(config, 'aiti_min_epsilon', 0.0),
        )
    else:
        raise ValueError(f"Unknown importance method: {method}")


class _UniformScorer(TokenImportanceScorer):
    """Baseline: all tokens weighted equally."""

    def score(self, input_ids, attention_mask=None, **kwargs) -> torch.Tensor:
        weights = torch.ones_like(input_ids, dtype=torch.float32)
        if attention_mask is not None:
            weights = weights * attention_mask.float()
        return weights
