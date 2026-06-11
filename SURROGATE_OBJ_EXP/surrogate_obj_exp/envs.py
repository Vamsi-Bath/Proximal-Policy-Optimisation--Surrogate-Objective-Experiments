from __future__ import annotations

from typing import Any, Dict, Tuple

import gymnasium as gym
import numpy as np


def make_env(env_id: str, seed: int, render_mode: str | None = None):
    env = gym.make(env_id, render_mode=render_mode)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env


def reset_env(env, seed: int | None = None) -> np.ndarray:
    obs, _info = env.reset(seed=seed)
    return np.asarray(obs, dtype=np.float32)


def step_env(env, action) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
    obs, reward, terminated, truncated, info = env.step(action)
    done = bool(terminated or truncated)
    return np.asarray(obs, dtype=np.float32), float(reward), done, info


def extract_xy(obs: np.ndarray, info: Dict[str, Any]) -> tuple[float, float]:
    # MuJoCo v4/v5 envs often expose x_position and y_position in info.
    if 'x_position' in info and 'y_position' in info:
        return float(info['x_position']), float(info['y_position'])
    if 'x_position' in info:
        return float(info['x_position']), 0.0
    if len(obs) >= 2:
        return float(obs[0]), float(obs[1])
    if len(obs) == 1:
        return float(obs[0]), 0.0
    return 0.0, 0.0
