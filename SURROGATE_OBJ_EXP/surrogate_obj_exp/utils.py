from __future__ import annotations

import os
import random
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_yaml(path: str | os.PathLike) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def ensure_dir(path: str | os.PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def flatten_dict(d: Dict[str, Any], prefix: str = '') -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in d.items():
        key = f'{prefix}.{k}' if prefix else str(k)
        if isinstance(v, dict):
            out.update(flatten_dict(v, key))
        elif is_dataclass(v):
            out.update(flatten_dict(asdict(v), key))
        else:
            out[key] = v
    return out


def explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    var_y = np.var(y_true)
    if var_y < 1e-12:
        return float('nan')
    return float(1.0 - np.var(y_true - y_pred) / var_y)


def safe_mean(xs):
    return float(np.mean(xs)) if len(xs) else float('nan')


def safe_std(xs):
    return float(np.std(xs)) if len(xs) else float('nan')
