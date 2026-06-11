from __future__ import annotations

import math
from typing import Tuple

import torch
from torch import nn
from torch.distributions import Normal, Independent


class MLPActorCritic(nn.Module):
    """Gaussian actor + scalar critic for continuous MuJoCo control."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes=(64, 64), log_std_init=-0.5):
        super().__init__()
        layers = []
        last = obs_dim
        for h in hidden_sizes:
            layers += [nn.Linear(last, h), nn.Tanh()]
            last = h
        self.shared = nn.Sequential(*layers)
        self.actor_mean = nn.Linear(last, act_dim)
        self.critic = nn.Linear(last, 1)
        self.log_std = nn.Parameter(torch.ones(act_dim) * log_std_init)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
                nn.init.constant_(module.bias, 0.0)
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)

    def forward(self, obs: torch.Tensor):
        z = self.shared(obs)
        mean = self.actor_mean(z)
        value = self.critic(z).squeeze(-1)
        return mean, self.log_std.exp().expand_as(mean), value

    def distribution(self, obs: torch.Tensor) -> Independent:
        mean, std, _ = self.forward(obs)
        return Independent(Normal(mean, std), 1)

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        z = self.shared(obs)
        return self.critic(z).squeeze(-1)

    @torch.no_grad()
    def act(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self.distribution(obs)
        action = dist.sample()
        logp = dist.log_prob(action)
        value = self.value(obs)
        return action, logp, value

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        dist = self.distribution(obs)
        logp = dist.log_prob(actions)
        entropy = dist.entropy()
        value = self.value(obs)
        return logp, entropy, value, dist
