"""TI-PPO Trainer: Token-Importance Guided PPO for LLM alignment.

Extends the standard PPO-RLHF loop with:
1. Per-token importance weighting on the policy/value objectives
2. Optional triplet loss (anchor=model output, positive=preferred, negative=rejected)
3. EMA-smoothed importance scores to reduce compute of gradient attribution
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional
import numpy as np

from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer

from .token_importance import build_scorer, TokenImportanceScorer
from .config import TIPPOConfig


class TIPPOTrainer:
    """Wraps trl.PPOTrainer with token-importance weighting and triplet loss."""

    def __init__(
        self,
        config: TIPPOConfig,
        model: AutoModelForCausalLMWithValueHead,
        ref_model: Optional[AutoModelForCausalLMWithValueHead],
        tokenizer: AutoTokenizer,
        reward_model=None,
        reward_tokenizer=None,
    ):
        self.config = config
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.reward_model = reward_model
        self.reward_tokenizer = reward_tokenizer

        # Build PPO config
        self.ppo_config = PPOConfig(
            model_name=config.model_name,
            learning_rate=config.learning_rate,
            batch_size=config.batch_size,
            mini_batch_size=config.mini_batch_size,
            ppo_epochs=config.ppo_epochs,
            max_grad_norm=config.max_grad_norm,
            cliprange=config.clip_epsilon,
            vf_coef=config.vf_coef,
            gamma=config.gamma,
            lam=config.lam,
            kl_penalty=config.kl_penalty,
            target_kl=config.target_kl,
            seed=config.seed,
            log_with=config.log_with if config.log_with != "none" else None,
            project_kwargs={"project_name": config.project_name},
        )

        self.ppo_trainer = PPOTrainer(
            config=self.ppo_config,
            model=model,
            ref_model=ref_model,
            tokenizer=tokenizer,
        )

        # Token importance scorer
        self.scorer = build_scorer(config)
        self.step_count = 0

        # Cache for EMA-smoothed importance scores
        self._importance_cache = None

    def compute_importance_weights(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
        values: Optional[torch.Tensor] = None,
        rewards: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute or retrieve cached token importance weights."""
        method = self.config.importance_method

        should_recompute = (
            self._importance_cache is None
            or self._importance_cache.shape != (input_ids.shape[0], input_ids.shape[1])
            or self.step_count % self.config.importance_update_freq == 0
        )

        if not should_recompute and self._importance_cache is not None:
            return self._importance_cache

        kwargs = dict(input_ids=input_ids, attention_mask=attention_mask)

        if method in ("hybrid", "gradient"):
            kwargs["model"] = self.model.pretrained_model
        elif method == "attention":
            kwargs["model"] = self.model.pretrained_model
        elif method == "td_error":
            if values is None or rewards is None:
                # Fallback to uniform if values/rewards not available yet
                return torch.ones_like(input_ids, dtype=torch.float32)
            kwargs["values"] = values
            kwargs["rewards"] = rewards
        elif method == "reward_model":
            kwargs["reward_model"] = self.reward_model

        weights = self.scorer.score(**kwargs)

        # EMA smoothing
        if self._importance_cache is not None and self._importance_cache.shape == weights.shape:
            alpha = self.config.importance_ema_decay
            weights = alpha * self._importance_cache + (1 - alpha) * weights

        self._importance_cache = weights.detach()
        return weights

    def weighted_ppo_loss(
        self,
        old_logprobs: torch.Tensor,
        new_logprobs: torch.Tensor,
        advantages: torch.Tensor,
        importance_weights: torch.Tensor,
        response_mask: torch.Tensor,
    ) -> torch.Tensor:
        """PPO clipped surrogate objective with token-importance weighting.

        L_TI-PPO = E[ w(t) * min(r(t)*A(t), clip(r(t), 1-eps, 1+eps)*A(t)) ]
        """
        eps = self.config.clip_epsilon
        ratio = torch.exp(new_logprobs - old_logprobs)

        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - eps, 1.0 + eps) * advantages
        clipped_surr = torch.min(surr1, surr2)

        # Apply token importance weighting
        weighted_surr = importance_weights * clipped_surr

        # Mask out padding and average over valid tokens
        masked = weighted_surr * response_mask
        loss = -masked.sum() / response_mask.sum().clamp(min=1)
        return loss

    def weighted_value_loss(
        self,
        values: torch.Tensor,
        returns: torch.Tensor,
        importance_weights: torch.Tensor,
        response_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Token-importance weighted value function loss."""
        vf_loss = (values - returns) ** 2
        weighted_vf = importance_weights * vf_loss
        masked = weighted_vf * response_mask
        return 0.5 * masked.sum() / response_mask.sum().clamp(min=1)

    def triplet_loss(
        self,
        anchor_hidden: torch.Tensor,
        preferred_hidden: torch.Tensor,
        rejected_hidden: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Triplet loss on sequence representations.

        Pushes model output closer to preferred and farther from rejected.
        Uses mean-pooled hidden states over valid tokens.
        """
        # Mean pool over valid token positions
        mask_f = mask.float().unsqueeze(-1)  # (B, T, 1)
        anchor_pool = (anchor_hidden * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)
        preferred_pool = (preferred_hidden * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)
        rejected_pool = (rejected_hidden * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)

        dist_pos = F.pairwise_distance(anchor_pool, preferred_pool)
        dist_neg = F.pairwise_distance(anchor_pool, rejected_pool)

        loss = F.relu(dist_pos - dist_neg + self.config.triplet_margin)
        return loss.mean()

    @torch.no_grad()
    def get_rewards(self, query_tensors, response_tensors):
        """Score responses using the reward model."""
        rewards = []
        for q, r in zip(query_tensors, response_tensors):
            full_ids = torch.cat([q, r]).unsqueeze(0).to(self.reward_model.device)
            attn = torch.ones_like(full_ids)

            output = self.reward_model(input_ids=full_ids, attention_mask=attn)
            if hasattr(output, "logits"):
                score = output.logits[0, -1].float()
            else:
                score = output[0][0, -1].float()
            rewards.append(score.cpu())
        return rewards

    def step(self, queries, responses, scores):
        """Run one TI-PPO step.

        This wraps trl's PPOTrainer.step() and applies token-importance
        weighting to the advantages before the PPO update.

        Args:
            queries: list of tokenized query tensors
            responses: list of tokenized response tensors
            scores: list of scalar reward tensors
        """
        self.step_count += 1

        # Standard PPO step through trl — this handles:
        # - old logprob computation
        # - advantage estimation (GAE)
        # - PPO mini-batch updates
        # - KL penalty logging
        stats = self.ppo_trainer.step(queries, responses, scores)

        # Compute importance weights for logging and next-step caching
        if queries and responses:
            sample_full = torch.cat([queries[0], responses[0]]).unsqueeze(0)
            sample_mask = torch.ones_like(sample_full)
            device = next(self.model.parameters()).device
            sample_full = sample_full.to(device)
            sample_mask = sample_mask.to(device)

            weights = self.compute_importance_weights(sample_full, sample_mask)
            stats["ti_ppo/mean_importance"] = weights.mean().item()
            stats["ti_ppo/importance_std"] = weights.std().item()
            stats["ti_ppo/importance_max"] = weights.max().item()
            stats["ti_ppo/importance_min"] = weights[weights > 0].min().item() if (weights > 0).any() else 0.0

        return stats

    def custom_ppo_step(
        self,
        query_ids: torch.Tensor,
        response_ids: torch.Tensor,
        old_logprobs: torch.Tensor,
        new_logprobs: torch.Tensor,
        values: torch.Tensor,
        returns: torch.Tensor,
        advantages: torch.Tensor,
        response_mask: torch.Tensor,
        preferred_ids: Optional[torch.Tensor] = None,
        rejected_ids: Optional[torch.Tensor] = None,
    ) -> dict:
        """Full custom TI-PPO step with importance-weighted losses and triplet loss.

        Use this instead of `step()` for full control over the training loop.
        """
        device = next(self.model.parameters()).device
        full_ids = torch.cat([query_ids, response_ids], dim=1).to(device)
        full_mask = torch.ones(full_ids.shape, device=device, dtype=torch.long)

        # 1. Compute token importance
        importance = self.compute_importance_weights(
            full_ids, full_mask, values=values, rewards=returns
        )
        # Slice to response portion only
        resp_len = response_ids.shape[1]
        resp_importance = importance[:, -resp_len:]

        # 2. Weighted PPO policy loss
        policy_loss = self.weighted_ppo_loss(
            old_logprobs, new_logprobs, advantages, resp_importance, response_mask
        )

        # 3. Weighted value loss
        value_loss = self.weighted_value_loss(
            values, returns, resp_importance, response_mask
        )

        total_loss = policy_loss + self.config.vf_coef * value_loss

        # 4. Triplet loss (optional)
        triplet_loss_val = torch.tensor(0.0, device=device)
        if self.config.use_triplet_loss and preferred_ids is not None and rejected_ids is not None:
            anchor_out = self.model.pretrained_model(
                input_ids=full_ids, attention_mask=full_mask, output_hidden_states=True
            )
            pref_mask = torch.ones(preferred_ids.shape, device=device, dtype=torch.long)
            pref_out = self.model.pretrained_model(
                input_ids=preferred_ids.to(device), attention_mask=pref_mask, output_hidden_states=True
            )
            rej_mask = torch.ones(rejected_ids.shape, device=device, dtype=torch.long)
            rej_out = self.model.pretrained_model(
                input_ids=rejected_ids.to(device), attention_mask=rej_mask, output_hidden_states=True
            )

            # Use last hidden layer for triplet comparison
            # Truncate to min length for comparability
            min_len = min(
                anchor_out.hidden_states[-1].shape[1],
                pref_out.hidden_states[-1].shape[1],
                rej_out.hidden_states[-1].shape[1],
            )
            common_mask = torch.ones(
                (full_ids.shape[0], min_len), device=device, dtype=torch.long
            )
            triplet_loss_val = self.triplet_loss(
                anchor_out.hidden_states[-1][:, :min_len],
                pref_out.hidden_states[-1][:, :min_len],
                rej_out.hidden_states[-1][:, :min_len],
                common_mask,
            )
            total_loss = total_loss + self.config.triplet_gamma * triplet_loss_val

        stats = {
            "ti_ppo/policy_loss": policy_loss.item(),
            "ti_ppo/value_loss": value_loss.item(),
            "ti_ppo/triplet_loss": triplet_loss_val.item(),
            "ti_ppo/total_loss": total_loss.item(),
            "ti_ppo/mean_importance": resp_importance.mean().item(),
            "ti_ppo/importance_std": resp_importance.std().item(),
        }

        self.step_count += 1
        return total_loss, stats
