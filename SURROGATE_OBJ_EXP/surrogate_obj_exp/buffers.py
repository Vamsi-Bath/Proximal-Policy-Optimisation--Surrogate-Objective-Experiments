from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch


@dataclass
class RolloutBatch:
    obs: torch.Tensor
    actions: torch.Tensor
    logp: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    values: torch.Tensor


class RolloutBuffer:
    def __init__(self, obs_dim: int, act_dim: int, size: int, gamma: float, gae_lambda: float):
        self.obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.actions = np.zeros((size, act_dim), dtype=np.float32)
        self.rewards = np.zeros(size, dtype=np.float32)
        self.dones = np.zeros(size, dtype=np.float32)
        self.values = np.zeros(size, dtype=np.float32)
        self.logp = np.zeros(size, dtype=np.float32)
        self.advantages = np.zeros(size, dtype=np.float32)
        self.returns = np.zeros(size, dtype=np.float32)
        self.ptr = 0
        self.size = size
        self.gamma = gamma
        self.gae_lambda = gae_lambda

    def add(self, obs, action, reward, done, value, logp):
        assert self.ptr < self.size
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.values[self.ptr] = value
        self.logp[self.ptr] = logp
        self.ptr += 1

    def compute_returns_advantages(self, last_value: float, last_done: float) -> None:
        adv = 0.0
        for t in reversed(range(self.size)):
            if t == self.size - 1:
                next_nonterminal = 1.0 - last_done
                next_value = last_value
            else:
                next_nonterminal = 1.0 - self.dones[t + 1]
                next_value = self.values[t + 1]
            delta = self.rewards[t] + self.gamma * next_value * next_nonterminal - self.values[t]
            adv = delta + self.gamma * self.gae_lambda * next_nonterminal * adv
            self.advantages[t] = adv
        self.returns = self.advantages + self.values

    def to_torch(self, device: str) -> RolloutBatch:
        adv = self.advantages.copy()
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return RolloutBatch(
            obs=torch.as_tensor(self.obs, dtype=torch.float32, device=device),
            actions=torch.as_tensor(self.actions, dtype=torch.float32, device=device),
            logp=torch.as_tensor(self.logp, dtype=torch.float32, device=device),
            advantages=torch.as_tensor(adv, dtype=torch.float32, device=device),
            returns=torch.as_tensor(self.returns, dtype=torch.float32, device=device),
            values=torch.as_tensor(self.values, dtype=torch.float32, device=device),
        )

    def minibatches(self, batch: RolloutBatch, minibatch_size: int):
        n = batch.obs.shape[0]
        idxs = np.random.permutation(n)
        for start in range(0, n, minibatch_size):
            mb = idxs[start:start + minibatch_size]
            yield RolloutBatch(
                obs=batch.obs[mb], actions=batch.actions[mb], logp=batch.logp[mb],
                advantages=batch.advantages[mb], returns=batch.returns[mb], values=batch.values[mb]
            )

    def as_np_metrics(self) -> Dict[str, np.ndarray]:
        return {'values': self.values.copy(), 'returns': self.returns.copy()}
