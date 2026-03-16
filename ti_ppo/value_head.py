"""Causal LM with a scalar value head for PPO."""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM


class CausalLMWithValueHead(nn.Module):
    """Wraps a causal LM with a linear value head on top of the last hidden state."""

    def __init__(self, pretrained_model):
        super().__init__()
        self.pretrained_model = pretrained_model
        hidden_size = pretrained_model.config.hidden_size
        # Place value head on same device and dtype as the model
        model_param = next(pretrained_model.parameters())
        model_device = model_param.device
        model_dtype = model_param.dtype
        self.value_head = nn.Linear(hidden_size, 1, bias=False,
                                    device=model_device, dtype=model_dtype)
        self.value_head.weight.data.zero_()

    @classmethod
    def from_pretrained(cls, model_name_or_path, **kwargs):
        base = AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
        return cls(base)

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None, **kwargs):
        outputs = self.pretrained_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=True,
            **kwargs,
        )
        hidden = outputs.hidden_states[-1]  # (B, T, D)
        # Ensure dtype matches value head (e.g., fp16 model with fp32 LoRA grads)
        hidden = hidden.to(self.value_head.weight.dtype)
        values = self.value_head(hidden).squeeze(-1)  # (B, T)
        return outputs.logits, values

    def generate(self, **kwargs):
        return self.pretrained_model.generate(**kwargs)

    @property
    def device(self):
        return next(self.pretrained_model.parameters()).device

    @property
    def config(self):
        return self.pretrained_model.config
