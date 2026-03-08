"""TI-PPO Trainer: Token-Importance Guided PPO for LLM alignment.

Pure PyTorch implementation of PPO-RLHF with:
1. Per-token importance weighting on the policy/value objectives
2. Optional triplet loss (anchor=model output, positive=preferred, negative=rejected)
3. EMA-smoothed importance scores to reduce compute of gradient attribution
4. GAE advantage estimation
5. KL penalty against reference model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math

from .token_importance import build_scorer, TokenImportanceScorer, PPO_NATIVE_METHODS
from .config import TIPPOConfig


class TIPPOTrainer:
    """Full PPO-RLHF trainer with token-importance weighting."""

    def __init__(
        self,
        config: TIPPOConfig,
        model,
        ref_model,
        tokenizer,
        reward_model=None,
        optimizer=None,
    ):
        self.config = config
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.reward_model = reward_model

        self.device = next(model.parameters()).device
        self.scorer = build_scorer(config)
        self.step_count = 0
        self._importance_cache = None

        # Optimizer: only train policy model params
        if optimizer is None:
            self.optimizer = torch.optim.AdamW(
                model.parameters(), lr=config.learning_rate
            )
        else:
            self.optimizer = optimizer

    # ------------------------------------------------------------------
    # Token importance
    # ------------------------------------------------------------------

    def compute_importance_weights(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        values: Optional[torch.Tensor] = None,
        rewards: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        method = self.config.importance_method

        should_recompute = (
            self._importance_cache is None
            or self._importance_cache.shape != input_ids.shape
            or self.step_count % self.config.importance_update_freq == 0
        )

        if not should_recompute and self._importance_cache is not None:
            return self._importance_cache

        kwargs = dict(input_ids=input_ids, attention_mask=attention_mask)

        if method in ("hybrid", "gradient", "attention"):
            kwargs["model"] = self.model.pretrained_model
        elif method == "td_error":
            if values is None or rewards is None:
                return torch.ones_like(input_ids, dtype=torch.float32)
            kwargs["values"] = values
            kwargs["rewards"] = rewards
        elif method == "reward_model":
            kwargs["reward_model"] = self.reward_model

        weights = self.scorer.score(**kwargs)

        # Sanitize: no NaN/Inf, clamp to [0, 1]
        weights = torch.nan_to_num(weights, nan=1.0, posinf=1.0, neginf=0.0)
        weights = weights.clamp(0.0, 1.0)

        # EMA smoothing
        if self._importance_cache is not None and self._importance_cache.shape == weights.shape:
            alpha = self.config.importance_ema_decay
            weights = alpha * self._importance_cache + (1 - alpha) * weights

        self._importance_cache = weights.detach()
        return weights

    # ------------------------------------------------------------------
    # Core PPO components
    # ------------------------------------------------------------------

    @staticmethod
    def compute_logprobs(logits, labels):
        """Per-token log probabilities of the chosen actions."""
        logprobs = F.log_softmax(logits, dim=-1)
        return torch.gather(logprobs, 2, labels.unsqueeze(-1)).squeeze(-1)

    def compute_gae(self, rewards_per_token, values, mask):
        """Generalized Advantage Estimation.

        Args:
            rewards_per_token: (B, T) per-token rewards
            values: (B, T) value estimates
            mask: (B, T) valid token mask
        Returns:
            advantages: (B, T)
            returns: (B, T)
        """
        B, T = values.shape
        gamma = self.config.gamma
        lam = self.config.lam

        advantages = torch.zeros_like(values)
        last_gae = torch.zeros(B, device=values.device)

        for t in reversed(range(T)):
            next_val = values[:, t + 1] if t + 1 < T else torch.zeros(B, device=values.device)
            delta = rewards_per_token[:, t] + gamma * next_val - values[:, t]
            last_gae = delta + gamma * lam * last_gae
            last_gae = last_gae * mask[:, t]
            advantages[:, t] = last_gae

        returns = advantages + values
        return advantages, returns

    def assign_token_rewards(self, scores, response_lens, max_resp_len):
        """Distribute scalar reward to the last token of each response.

        Args:
            scores: list of scalar reward tensors (one per sample)
            response_lens: list of int response lengths
            max_resp_len: int, padded response length
        Returns:
            rewards_per_token: (B, max_resp_len)
        """
        B = len(scores)
        rewards = torch.zeros(B, max_resp_len, device=self.device)
        for i, (s, rlen) in enumerate(zip(scores, response_lens)):
            if rlen > 0:
                rewards[i, rlen - 1] = s.to(self.device)
        return rewards

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

    # ------------------------------------------------------------------
    # PPO losses
    # ------------------------------------------------------------------

    def weighted_ppo_loss(self, old_logprobs, new_logprobs, advantages, importance, mask):
        """PPO clipped surrogate with token-importance weighting."""
        eps = self.config.clip_epsilon
        ratio = torch.exp(new_logprobs - old_logprobs)

        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - eps, 1.0 + eps) * advantages
        clipped = torch.min(surr1, surr2)

        weighted = importance * clipped * mask
        return -weighted.sum() / mask.sum().clamp(min=1)

    def weighted_value_loss(self, values, returns, importance, mask):
        """Token-importance weighted value function loss."""
        vf_loss = (values - returns) ** 2
        weighted = importance * vf_loss * mask
        return 0.5 * weighted.sum() / mask.sum().clamp(min=1)

    def kl_penalty(self, new_logprobs, ref_logprobs, mask):
        """Per-token KL divergence from reference model."""
        kl = new_logprobs - ref_logprobs  # approximate KL
        return (kl * mask).sum() / mask.sum().clamp(min=1)

    def triplet_loss(self, anchor_hidden, preferred_hidden, rejected_hidden, mask):
        """Triplet loss on mean-pooled hidden states."""
        mask_f = mask.float().unsqueeze(-1)
        anchor_pool = (anchor_hidden * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)
        preferred_pool = (preferred_hidden * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)
        rejected_pool = (rejected_hidden * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)

        dist_pos = F.pairwise_distance(anchor_pool, preferred_pool)
        dist_neg = F.pairwise_distance(anchor_pool, rejected_pool)

        return F.relu(dist_pos - dist_neg + self.config.triplet_margin).mean()

    # ------------------------------------------------------------------
    # Main training step
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _collect_rollout(self, query_tensors, response_tensors):
        """Compute old logprobs, ref logprobs, and values for the rollout."""
        old_logprobs_list = []
        ref_logprobs_list = []
        values_list = []

        for q, r in zip(query_tensors, response_tensors):
            full = torch.cat([q, r]).unsqueeze(0).to(self.device)
            resp_start = q.shape[0]
            resp_len = r.shape[0]

            # Policy model forward
            logits, vals = self.model(full)
            lp = self.compute_logprobs(
                logits[:, resp_start - 1 : resp_start + resp_len - 1],
                full[:, resp_start : resp_start + resp_len],
            )
            old_logprobs_list.append(lp.squeeze(0))
            values_list.append(vals[:, resp_start : resp_start + resp_len].squeeze(0))

            # Reference model forward
            ref_out = self.ref_model.pretrained_model(full)
            ref_logits = ref_out.logits
            ref_lp = self.compute_logprobs(
                ref_logits[:, resp_start - 1 : resp_start + resp_len - 1],
                full[:, resp_start : resp_start + resp_len],
            )
            ref_logprobs_list.append(ref_lp.squeeze(0))

        return old_logprobs_list, ref_logprobs_list, values_list

    def _pad_tensors(self, tensor_list, pad_value=0.0):
        """Pad list of 1D tensors to same length and stack."""
        max_len = max(t.shape[0] for t in tensor_list)
        padded = []
        masks = []
        for t in tensor_list:
            pad_len = max_len - t.shape[0]
            padded.append(F.pad(t, (0, pad_len), value=pad_value))
            m = torch.ones(max_len, device=t.device)
            if pad_len > 0:
                m[-pad_len:] = 0
            masks.append(m)
        return torch.stack(padded), torch.stack(masks)

    def step(self, query_tensors, response_tensors, scores):
        """Run one full TI-PPO update.

        Args:
            query_tensors: list of (query_len,) token id tensors
            response_tensors: list of (resp_len,) token id tensors
            scores: list of scalar reward tensors
        Returns:
            stats: dict of training statistics
        """
        self.step_count += 1
        B = len(query_tensors)

        # 1. Collect rollout data
        old_logprobs_list, ref_logprobs_list, values_list = self._collect_rollout(
            query_tensors, response_tensors
        )
        response_lens = [r.shape[0] for r in response_tensors]

        # 2. Pad everything to same length
        old_logprobs, resp_mask = self._pad_tensors(old_logprobs_list)
        ref_logprobs, _ = self._pad_tensors(ref_logprobs_list)
        values, _ = self._pad_tensors(values_list)
        max_resp_len = old_logprobs.shape[1]

        # 3. Build per-token rewards (scalar reward at last response token)
        rewards_per_token = self.assign_token_rewards(scores, response_lens, max_resp_len)

        # 4. KL penalty as reward shaping
        kl_per_token = old_logprobs - ref_logprobs
        kl_per_token = torch.nan_to_num(kl_per_token, nan=0.0, posinf=0.0, neginf=0.0)
        rewards_per_token = rewards_per_token - 0.1 * kl_per_token

        # 5. GAE
        advantages, returns = self.compute_gae(rewards_per_token, values.detach(), resp_mask)

        # Normalize advantages
        adv_mean = (advantages * resp_mask).sum() / resp_mask.sum()
        adv_std = ((advantages - adv_mean).pow(2) * resp_mask).sum() / resp_mask.sum()
        advantages = (advantages - adv_mean) / (adv_std.sqrt() + 1e-8)

        # 6. Compute token importance weights
        method = self.config.importance_method

        if method in PPO_NATIVE_METHODS:
            # PPO-native methods use signals from the rollout itself
            # Collect response-level logits for entropy-based methods
            resp_logits = None
            if method in ("entropy", "entropy_advantage"):
                resp_logits_list = []
                with torch.no_grad():
                    for q, r in zip(query_tensors, response_tensors):
                        full = torch.cat([q, r]).unsqueeze(0).to(self.device)
                        resp_start = q.shape[0]
                        resp_len_i = r.shape[0]
                        logits_out, _ = self.model(full)
                        resp_logits_list.append(
                            logits_out[0, resp_start:resp_start + resp_len_i]
                        )
                # Pad logits to same length
                max_rlen = max(l.shape[0] for l in resp_logits_list)
                V = resp_logits_list[0].shape[-1]
                padded_logits = []
                for l in resp_logits_list:
                    pad_len = max_rlen - l.shape[0]
                    if pad_len > 0:
                        padded_logits.append(F.pad(l, (0, 0, 0, pad_len)))
                    else:
                        padded_logits.append(l)
                resp_logits = torch.stack(padded_logits)  # (B, T, V)

            # Build kwargs for the scorer
            scorer_kwargs = dict(attention_mask=resp_mask)
            if method == "advantage":
                scorer_kwargs["advantages"] = advantages.detach()
            elif method == "entropy":
                scorer_kwargs["logits"] = resp_logits
            elif method == "kl_guided":
                scorer_kwargs["advantages"] = advantages.detach()
                scorer_kwargs["old_logprobs"] = old_logprobs.detach()
                scorer_kwargs["ref_logprobs"] = ref_logprobs.detach()
            elif method == "adv_gaussian":
                scorer_kwargs["advantages"] = advantages.detach()
                # Need response-level input_ids for Gaussian prior
                resp_ids_list = [r.to(self.device) for r in response_tensors]
                resp_ids_padded, _ = self._pad_tensors(resp_ids_list, pad_value=0)
                scorer_kwargs["input_ids"] = resp_ids_padded.long()
            elif method == "entropy_advantage":
                scorer_kwargs["logits"] = resp_logits
                scorer_kwargs["advantages"] = advantages.detach()

            resp_importance = self.scorer.score(**scorer_kwargs)
            resp_importance = torch.nan_to_num(resp_importance, nan=1.0, posinf=1.0, neginf=0.0)
            resp_importance = resp_importance.clamp(0.0, 1.0)

        elif method == "td_error":
            # TD-Error operates on response-level values/rewards directly
            resp_importance = self.compute_importance_weights(
                old_logprobs.long(), resp_mask.long(),
                values=values.detach(), rewards=rewards_per_token,
            )
            if resp_importance.shape != resp_mask.shape:
                resp_importance = resp_importance[:, :resp_mask.shape[1]]
        else:
            # External methods (hybrid, gradient, attention, uniform)
            full_ids_list = [
                torch.cat([q, r]).to(self.device)
                for q, r in zip(query_tensors, response_tensors)
            ]
            full_ids_padded, full_mask = self._pad_tensors(full_ids_list, pad_value=0)
            full_ids_padded = full_ids_padded.long()

            importance = self.compute_importance_weights(
                full_ids_padded, full_mask.long(),
            )
            # Slice to response portion
            resp_importance_list = []
            for i, q in enumerate(query_tensors):
                qlen = q.shape[0]
                rlen = response_lens[i]
                imp = importance[i, qlen : qlen + rlen]
                resp_importance_list.append(imp)
            resp_importance, _ = self._pad_tensors(resp_importance_list, pad_value=0)

        # 7. PPO mini-batch updates
        total_policy_loss = 0
        total_value_loss = 0
        total_kl = 0

        for epoch in range(self.config.ppo_epochs):
            # Recompute logprobs and values under current policy
            new_logprobs_list = []
            new_values_list = []
            for q, r in zip(query_tensors, response_tensors):
                full = torch.cat([q, r]).unsqueeze(0).to(self.device)
                resp_start = q.shape[0]
                resp_len = r.shape[0]

                logits, vals = self.model(full)
                lp = self.compute_logprobs(
                    logits[:, resp_start - 1 : resp_start + resp_len - 1],
                    full[:, resp_start : resp_start + resp_len],
                )
                new_logprobs_list.append(lp.squeeze(0))
                new_values_list.append(vals[:, resp_start : resp_start + resp_len].squeeze(0))

            new_logprobs, _ = self._pad_tensors(new_logprobs_list)
            new_values, _ = self._pad_tensors(new_values_list)

            # Losses
            policy_loss = self.weighted_ppo_loss(
                old_logprobs.detach(), new_logprobs, advantages.detach(),
                resp_importance.detach(), resp_mask,
            )
            value_loss = self.weighted_value_loss(
                new_values, returns.detach(), resp_importance.detach(), resp_mask,
            )
            kl = self.kl_penalty(new_logprobs, ref_logprobs.detach(), resp_mask)

            loss = policy_loss + self.config.vf_coef * value_loss

            # Skip update if loss is NaN to prevent model corruption
            if torch.isnan(loss) or torch.isinf(loss):
                continue

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_kl += kl.item()

        n_epochs = self.config.ppo_epochs
        stats = {
            "ppo/policy_loss": total_policy_loss / n_epochs,
            "ppo/value_loss": total_value_loss / n_epochs,
            "ppo/mean_kl": total_kl / n_epochs,
            "ppo/mean_reward": torch.stack(scores).mean().item(),
            "ti_ppo/mean_importance": resp_importance[resp_mask.bool()].mean().item(),
            "ti_ppo/importance_std": resp_importance[resp_mask.bool()].std().item(),
        }
        return stats
