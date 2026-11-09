"""Auxiliary halting heads.

Auxiliary heads predict the same target as the model's own halting head
(`q_head`: "is the current prediction fully correct?") but are strictly
observers: they are trained alongside the model, yet

- they only ever receive DETACHED copies of the model state (the model builds
  `AuxHaltInputs` from `.detach()`ed tensors in one place, so no head type can
  send gradients into the backbone, the embeddings or `q_head`);
- their logits are never used for ACT halting decisions;
- they are created after all other parameters, under `torch.random.fork_rng`,
  so enabling them does not change the random numbers the backbone sees.

Adding a head type: subclass `AuxHaltHead`, decorate it with
`@register_aux_halt_head("<type>")`, and enable it from the arch config:

    aux_halt_heads:
      - type: detached_linear          # `name` defaults to the type
      - type: my_head
        name: my_head_2l
        num_layers: 2                  # extra keys are passed to the head as `head_config`
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Type

import torch
from torch import nn
from pydantic import BaseModel, ConfigDict

from models.layers import CastedLinear, CosSin


class AuxHaltHeadConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: str
    name: Optional[str] = None

    @property
    def head_name(self) -> str:
        return self.name if self.name is not None else self.type


@dataclass
class AuxHaltInputs:
    """Everything an auxiliary head may look at. All tensors are detached.

    z_H, z_L: [B, puzzle_emb_len + seq_len, hidden_size] full answer / latent state
              after the current ACT step (prefix positions included), in forward_dtype.
    input_embeddings: [B, puzzle_emb_len + seq_len, hidden_size] scaled input embeddings.
    puzzle_emb_len: number of puzzle-embedding prefix positions.
    cos_sin: rotary tables of the backbone (None unless pos_encodings == "rope").
    """
    z_H: torch.Tensor
    z_L: torch.Tensor
    input_embeddings: torch.Tensor
    puzzle_emb_len: int
    cos_sin: Optional[CosSin]


class AuxHaltHead(nn.Module):
    """Base class. `forward` returns the halt logit, float32, shape [B]."""

    def __init__(self, model_config, head_config: AuxHaltHeadConfig) -> None:
        super().__init__()
        self.model_config = model_config
        self.head_config = head_config

    def forward(self, inputs: AuxHaltInputs) -> torch.Tensor:
        raise NotImplementedError


AUX_HALT_HEADS: Dict[str, Type[AuxHaltHead]] = {}


def register_aux_halt_head(type_name: str):
    def decorator(cls: Type[AuxHaltHead]) -> Type[AuxHaltHead]:
        if type_name in AUX_HALT_HEADS:
            raise ValueError(f"Auxiliary halt head type '{type_name}' registered twice")
        AUX_HALT_HEADS[type_name] = cls
        return cls
    return decorator


@register_aux_halt_head("detached_linear")
class DetachedLinearHaltHead(AuxHaltHead):
    """Twin of the model's q_head (halt output only), reading a detached z_H[:, 0].

    Same layer (CastedLinear, bias), same input position, same init (zero weight, bias -5),
    same casting (weight cast to the activation dtype, logit cast to float32).
    """

    def __init__(self, model_config, head_config: AuxHaltHeadConfig) -> None:
        super().__init__(model_config, head_config)
        self.linear = CastedLinear(model_config.hidden_size, 1, bias=True, zero_init=True)
        with torch.no_grad():
            self.linear.bias.fill_(-5)  # type: ignore

    def forward(self, inputs: AuxHaltInputs) -> torch.Tensor:
        return self.linear(inputs.z_H[:, 0]).to(torch.float32)[..., 0]


def build_aux_halt_heads(model_config, head_configs: List[AuxHaltHeadConfig]) -> nn.ModuleDict:
    """Create the heads without disturbing the global RNG state (CPU and current CUDA device)."""
    names = [cfg.head_name for cfg in head_configs]
    if len(set(names)) != len(names):
        raise ValueError(f"Auxiliary halt head names must be unique, got {names}")
    for cfg in head_configs:
        if cfg.type not in AUX_HALT_HEADS:
            raise ValueError(f"Unknown auxiliary halt head type '{cfg.type}'. Known: {sorted(AUX_HALT_HEADS)}")

    cuda_devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=cuda_devices):
        return nn.ModuleDict({cfg.head_name: AUX_HALT_HEADS[cfg.type](model_config, cfg) for cfg in head_configs})
