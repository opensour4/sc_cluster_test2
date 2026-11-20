"""Tensor contracts shared by training and reverse diffusion."""
from dataclasses import dataclass, field

import torch


@dataclass
class SequenceBatch:
    token_ids: torch.Tensor
    condition_prefix_length: int
    seq_length: int
    metadata: dict


@dataclass
class ModelInput:
    noisy_token_ids: torch.Tensor
    metadata: dict


@dataclass
class ModelOutput:
    logits: torch.Tensor
    metadata: dict


@dataclass
class LossInputContext:
    token_ids: torch.Tensor
    condition_prefix_length: int
    seq_length: int
    mask_indices: torch.Tensor
    p_mask: torch.Tensor


@dataclass
class LossResult:
    loss: torch.Tensor
    metrics: dict
    metadata: dict = field(default_factory=dict)
