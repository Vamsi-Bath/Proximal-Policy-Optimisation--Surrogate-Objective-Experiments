from __future__ import annotations


from .ppo_base import PPOBase


class PPOKL(PPOBase):
    """PPO with no clipped surrogate; uses likelihood-ratio objective plus KL penalty."""
    def policy_objective(self, ratio, advantages, approx_kl):
        surrogate = ratio * advantages
        return -surrogate.mean() + self.kl_coef * approx_kl
