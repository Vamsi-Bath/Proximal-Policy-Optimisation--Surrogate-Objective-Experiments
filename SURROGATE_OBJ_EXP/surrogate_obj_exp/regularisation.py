from __future__ import annotations

import torch
from torch import nn


def l1_penalty(model: nn.Module, coeff: float) -> torch.Tensor:
    if coeff <= 0:
        return torch.tensor(0.0, device=next(model.parameters()).device)
    return coeff * sum(p.abs().sum() for p in model.parameters() if p.requires_grad and p.dim() > 1)


def l2_penalty(model: nn.Module, coeff: float) -> torch.Tensor:
    if coeff <= 0:
        return torch.tensor(0.0, device=next(model.parameters()).device)
    return coeff * sum((p ** 2).sum() for p in model.parameters() if p.requires_grad and p.dim() > 1)


@torch.no_grad()
def shrink_and_perturb(model: nn.Module, shrink: float = 0.95, perturb_std: float = 1e-3) -> None:
    """Applies theta <- shrink * theta + N(0, perturb_std^2)."""
    for p in model.parameters():
        if p.requires_grad:
            p.mul_(shrink)
            if perturb_std > 0:
                p.add_(torch.randn_like(p) * perturb_std)
