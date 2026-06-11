# PPO surrogate objective experiments for MuJoCo continuous-control environments.

1. **PPO-Clip**: clipped surrogate objective.
2. **PPO-KL**: no clipped surrogate; adaptive KL-penalty objective plus target-KL stopping.
3. **PPO-NoClip**: no clipped surrogate; plain likelihood-ratio policy gradient with optional target-KL stopping.

It also supports ablations for:

- L1 regularisation
- L2 regularisation / AdamW-style weight decay
- Shrink-and-perturb resets
- Surrogate objective choice
- KL target, clip range, entropy bonus, GAE lambda, seeds and environments

Default environment: `HalfCheetah-v5`.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

MuJoCo environments are provided through Gymnasium's `gymnasium[mujoco]` extra.

## Run one experiment

```bash
python -m surrogate_obj_exp.train --config configs/halfcheetah_clip.yaml
python -m surrogate_obj_exp.train --config configs/halfcheetah_kl.yaml
python -m surrogate_obj_exp.train --config configs/halfcheetah_noclip.yaml
```

## Run ablations

```bash
bash scripts/run_ablation.sh
```

## Plot results

```bash
python -m surrogate_obj_exp.plot_results --log-dir results --out-dir plots
```

This generates:

- learning curves
- approximate KL curves
- entropy curves
- value-loss curves
- explained-variance curves
- clip-fraction curves where applicable
- XY state coverage trajectory plots
- aggregate CSV summaries

## Key metrics logged

Per update:

- episodic return mean/std/min/max
- episodic length mean
- policy loss
- value loss
- entropy
- approximate KL
- clip fraction
- explained variance
- gradient norm
- action mean/std
- objective name
- regularisation settings
- shrink-and-perturb settings
- state coverage proxy metrics

For MuJoCo XY coverage, the logger uses `info["x_position"]`/`info["y_position"]` when available, otherwise it falls back to observation dimensions `obs[0]` and `obs[1]` as a generic 2D projection.

## Repository layout

```text
SURROGATE_OBJ_EXP/
  configs/
  scripts/
  surrogate_obj_exp/
    buffers.py
    csv_logger.py
    envs.py
    metrics.py
    models.py
    ppo_clip.py
    ppo_kl.py
    ppo_no_clip.py
    regularisation.py
    train.py
    plot_results.py
    utils.py
```

## Notes

- These are research scripts, not heavily optimised distributed-training code.
- The default settings are intentionally modest so they are easier to run on a single machine.
- For publication-quality comparisons, run multiple seeds and report confidence intervals.
