from __future__ import annotations

import torch

from .ppo_base import PPOBase


class PPOClip(PPOBase):
    """PPO with clipped surrogate objective."""
    def policy_objective(self, ratio, advantages, approx_kl):
        unclipped = ratio * advantages
        clipped = torch.clamp(ratio, 1.0 - self.cfg.clip_coef, 1.0 + self.cfg.clip_coef) * advantages
        return -torch.min(unclipped, clipped).mean()
