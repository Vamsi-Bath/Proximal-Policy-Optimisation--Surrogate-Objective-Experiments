from __future__ import annotations

from .ppo_base import PPOBase


class PPONoClip(PPOBase):
    """No clipped surrogate and no KL penalty; still logs KL and can early-stop by target KL."""
    def policy_objective(self, ratio, advantages, approx_kl):
        return -(ratio * advantages).mean()
