from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import trange

from .buffers import RolloutBuffer
from .csv_logger import CSVLogger
from .envs import extract_xy, make_env, reset_env, step_env
from .metrics import coverage_metrics
from .models import MLPActorCritic
from .ppo_base import PPOConfig
from .ppo_clip import PPOClip
from .ppo_kl import PPOKL
from .ppo_no_clip import PPONoClip
from .regularisation import shrink_and_perturb
from .utils import ensure_dir, load_yaml, set_seed, safe_mean, safe_std


def build_agent(model, cfg: PPOConfig):
    if cfg.objective == 'clip':
        return PPOClip(model, cfg)
    if cfg.objective == 'kl':
        return PPOKL(model, cfg)
    if cfg.objective == 'noclip':
        return PPONoClip(model, cfg)
    raise ValueError(f'Unknown objective: {cfg.objective}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    raw_cfg = load_yaml(args.config)

    seed = int(raw_cfg.get('seed', 0))
    set_seed(seed)
    device = raw_cfg.get('device', 'auto')
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    env_id = raw_cfg.get('env_id', 'HalfCheetah-v5')
    run_name = raw_cfg.get('run_name', f'{env_id}_{raw_cfg["ppo"]["objective"]}_seed{seed}_{int(time.time())}')
    out_dir = ensure_dir(Path(raw_cfg.get('out_dir', 'results')) / run_name)

    with open(out_dir / 'config_resolved.json', 'w', encoding='utf-8') as f:
        json.dump(raw_cfg, f, indent=2)

    env = make_env(env_id, seed=seed)
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))
    action_low = torch.as_tensor(env.action_space.low, dtype=torch.float32, device=device)
    action_high = torch.as_tensor(env.action_space.high, dtype=torch.float32, device=device)

    model_cfg = raw_cfg.get('model', {})
    model = MLPActorCritic(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_sizes=tuple(model_cfg.get('hidden_sizes', [64, 64])),
        log_std_init=float(model_cfg.get('log_std_init', -0.5)),
    ).to(device)

    ppo_cfg = PPOConfig(**raw_cfg['ppo'])
    agent = build_agent(model, ppo_cfg)

    total_timesteps = int(raw_cfg.get('total_timesteps', 200_000))
    rollout_steps = int(raw_cfg.get('rollout_steps', 2048))
    updates = total_timesteps // rollout_steps
    save_every = int(raw_cfg.get('save_every', 25))
    coverage_bins = int(raw_cfg.get('coverage_bins', 50))

    sp_cfg = raw_cfg.get('shrink_perturb', {})
    sp_enabled = bool(sp_cfg.get('enabled', False))
    sp_every = int(sp_cfg.get('every_updates', 50))
    sp_shrink = float(sp_cfg.get('shrink', 0.95))
    sp_std = float(sp_cfg.get('perturb_std', 1e-3))

    train_log = CSVLogger(out_dir / 'train_metrics.csv')
    episode_log = CSVLogger(out_dir / 'episodes.csv')
    xy_log = CSVLogger(out_dir / 'xy_trajectory.csv')

    obs = reset_env(env, seed=seed)
    global_step = 0
    episode_return = 0.0
    episode_length = 0
    ep_returns, ep_lengths = [], []
    all_xys = []

    for update in trange(1, updates + 1, desc=run_name):
        buffer = RolloutBuffer(obs_dim, act_dim, rollout_steps, ppo_cfg.gamma, ppo_cfg.gae_lambda)
        update_returns, update_lengths = [], []

        for step in range(rollout_steps):
            global_step += 1
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action_t, logp_t, value_t = model.act(obs_t)
                clipped_action_t = torch.clamp(action_t, action_low, action_high)
            action = clipped_action_t.squeeze(0).cpu().numpy()
            raw_action = action_t.squeeze(0).cpu().numpy()
            next_obs, reward, done, info = step_env(env, action)
            x, y = extract_xy(next_obs, info)
            all_xys.append((x, y))
            xy_log.log({'global_step': global_step, 'update': update, 'x': x, 'y': y, 'episode_length': episode_length})

            buffer.add(obs, raw_action, reward, float(done), value_t.item(), logp_t.item())
            episode_return += reward
            episode_length += 1
            obs = next_obs

            if done:
                ep_returns.append(episode_return)
                ep_lengths.append(episode_length)
                update_returns.append(episode_return)
                update_lengths.append(episode_length)
                episode_log.log({
                    'global_step': global_step, 'update': update,
                    'episode_return': episode_return,
                    'episode_length': episode_length,
                    'objective': ppo_cfg.objective,
                    'env_id': env_id,
                    'seed': seed,
                })
                obs = reset_env(env)
                episode_return = 0.0
                episode_length = 0

        with torch.no_grad():
            last_obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            last_value = float(model.value(last_obs_t).item())
        buffer.compute_returns_advantages(last_value, float(done))
        update_metrics = agent.update(buffer, device)

        if sp_enabled and update % sp_every == 0:
            shrink_and_perturb(model, shrink=sp_shrink, perturb_std=sp_std)
            sp_applied = 1.0
        else:
            sp_applied = 0.0

        cov = coverage_metrics(all_xys[-rollout_steps:], bins=coverage_bins)
        row = {
            'global_step': global_step,
            'update': update,
            'env_id': env_id,
            'seed': seed,
            'objective': ppo_cfg.objective,
            'l1_coeff': ppo_cfg.l1_coeff,
            'l2_coeff': ppo_cfg.l2_coeff,
            'weight_decay': ppo_cfg.weight_decay,
            'shrink_perturb_enabled': float(sp_enabled),
            'shrink_perturb_applied': sp_applied,
            'episodic_return_mean': safe_mean(update_returns if update_returns else ep_returns[-10:]),
            'episodic_return_std': safe_std(update_returns if update_returns else ep_returns[-10:]),
            'episodic_return_min': float(np.min(update_returns)) if update_returns else float('nan'),
            'episodic_return_max': float(np.max(update_returns)) if update_returns else float('nan'),
            'episodic_length_mean': safe_mean(update_lengths if update_lengths else ep_lengths[-10:]),
            **update_metrics,
            **cov,
        }
        train_log.log(row)

        if update % save_every == 0 or update == updates:
            torch.save(model.state_dict(), out_dir / f'model_update_{update}.pt')

    train_log.close(); episode_log.close(); xy_log.close(); env.close()
    print(f'Done. Results in: {out_dir}')


if __name__ == '__main__':
    main()
