"""The looped LM family:

    embedding -> prelude (P layers) -> core (C layers) x n_loops -> coda (K layers) -> RMSNorm -> head

The core is a module list applied in a loop, so weight sharing across loops needs no
restructuring. Only n_loops = 1 is implemented: the model is then P + C + K distinct layers.
The head is tied to the input embedding.
"""
import math

import torch
from torch import nn

from .config import ModelConfig
from .layers import Block, RMSNorm, RotaryEmbedding


class LoopedLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        if config.n_loops != 1:
            raise NotImplementedError(f"n_loops = {config.n_loops}: only n_loops = 1 is implemented")
        self.config = c = config

        def stack(n):
            return nn.ModuleList(Block(c.width, c.num_heads, c.head_dim, c.mlp_hidden, c.norm_eps) for _ in range(n))
        self.embed = nn.Embedding(c.vocab_size, c.width)
        self.prelude = stack(c.prelude_layers)
        self.core = stack(c.core_layers)
        self.coda = stack(c.coda_layers)
        self.final_norm = RMSNorm(c.width, c.norm_eps)
        self.rotary = RotaryEmbedding(c.head_dim, c.max_positions, c.rope_base)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Normal(0, init_std) everywhere; attention and MLP output projections scaled by
        1 / sqrt(2 * num_layers); norm gains 1."""
        std, out_std = self.config.init_std, self.config.init_std / math.sqrt(2 * self.config.num_layers)
        for name, p in self.named_parameters():
            if name.endswith("norm.weight"):
                nn.init.ones_(p)
            elif name.endswith("o_proj.weight") or name.endswith("down_proj.weight"):
                nn.init.normal_(p, 0.0, out_std)
            else:
                nn.init.normal_(p, 0.0, std)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens [B, S] -> final hidden states [B, S, d] (after the final norm). Use head() for logits."""
        S = tokens.shape[1]
        if S > self.config.max_positions:
            raise ValueError(f"sequence of {S} tokens, more than max_positions = {self.config.max_positions}")
        cos, sin = self.rotary(S)
        x = self.embed(tokens)
        for block in self.prelude:
            x = block(x, cos, sin)
        for _ in range(self.config.n_loops):
            for block in self.core:
                x = block(x, cos, sin)
        for block in self.coda:
            x = block(x, cos, sin)
        return self.final_norm(x)

    def head(self, hidden: torch.Tensor) -> torch.Tensor:
        """Logits from hidden states (any leading shape), with the tied embedding."""
        return hidden @ self.embed.weight.to(hidden.dtype).t()

    def logits(self, tokens: torch.Tensor) -> torch.Tensor:
        """Full logits [B, S, V] (tests and inspection; training computes the head at loss positions only)."""
        return self.head(self(tokens))

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
