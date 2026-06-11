from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch.nn.utils import clip_grad_norm_

from .buffers import RolloutBatch
from .regularisation import l1_penalty, l2_penalty
from .utils import explained_variance


@dataclass
class PPOConfig:
    objective: str
    lr: float = 3e-4
    epochs: int = 10
    minibatch_size: int = 256
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    target_kl: float = 0.02
    kl_coef: float = 1.0
    kl_adapt: bool = True
    vf_coef: float = 0.5
    ent_coef: float = 0.0
    max_grad_norm: float = 0.5
    l1_coeff: float = 0.0
    l2_coeff: float = 0.0
    weight_decay: float = 0.0


class PPOBase:
    def __init__(self, model, cfg: PPOConfig):
        self.model = model
        self.cfg = cfg
        self.optimiser = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.kl_coef = cfg.kl_coef

    def policy_objective(self, ratio, advantages, approx_kl):
        raise NotImplementedError

    def update(self, buffer, device: str) -> Dict[str, float]:
        batch = buffer.to_torch(device)
        metrics_accum = []
        stop_early = False
        for _epoch in range(self.cfg.epochs):
            for mb in buffer.minibatches(batch, self.cfg.minibatch_size):
                new_logp, entropy, new_values, _dist = self.model.evaluate_actions(mb.obs, mb.actions)
                log_ratio = new_logp - mb.logp
                ratio = log_ratio.exp()
                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean()
                    old_approx_kl = (mb.logp - new_logp).mean()
                    clip_fraction = ((ratio - 1.0).abs() > self.cfg.clip_coef).float().mean()

                policy_loss = self.policy_objective(ratio, mb.advantages, approx_kl)
                value_loss = 0.5 * ((new_values - mb.returns) ** 2).mean()
                entropy_loss = entropy.mean()
                reg_loss = l1_penalty(self.model, self.cfg.l1_coeff) + l2_penalty(self.model, self.cfg.l2_coeff)
                loss = policy_loss + self.cfg.vf_coef * value_loss - self.cfg.ent_coef * entropy_loss + reg_loss

                self.optimiser.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                self.optimiser.step()

                metrics_accum.append({
                    'loss': float(loss.detach().cpu()),
                    'policy_loss': float(policy_loss.detach().cpu()),
                    'value_loss': float(value_loss.detach().cpu()),
                    'entropy': float(entropy_loss.detach().cpu()),
                    'approx_kl': float(approx_kl.detach().cpu()),
                    'old_approx_kl': float(old_approx_kl.detach().cpu()),
                    'clip_fraction': float(clip_fraction.detach().cpu()),
                    'grad_norm': float(grad_norm.detach().cpu() if hasattr(grad_norm, 'detach') else grad_norm),
                    'kl_coef': float(self.kl_coef),
                })

                if self.cfg.target_kl and approx_kl > 1.5 * self.cfg.target_kl:
                    stop_early = True
                    break
            if stop_early:
                break

        # Adaptive KL coefficient for PPO-KL; harmless no-op for subclasses that ignore it.
        last_kl = metrics_accum[-1]['approx_kl'] if metrics_accum else 0.0
        if self.cfg.kl_adapt and self.cfg.objective == 'kl':
            if last_kl > 1.5 * self.cfg.target_kl:
                self.kl_coef *= 2.0
            elif last_kl < self.cfg.target_kl / 1.5:
                self.kl_coef *= 0.5

        out = {k: sum(m[k] for m in metrics_accum) / max(1, len(metrics_accum)) for k in metrics_accum[0]} if metrics_accum else {}
        np_metrics = buffer.as_np_metrics()
        out['explained_variance'] = explained_variance(np_metrics['values'], np_metrics['returns'])
        out['early_stop_kl'] = float(stop_early)
        out['num_minibatches'] = float(len(metrics_accum))
        return out
