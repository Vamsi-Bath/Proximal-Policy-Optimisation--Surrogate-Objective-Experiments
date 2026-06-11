#!/usr/bin/env bash
set -euo pipefail
python -m surrogate_obj_exp.train --config configs/halfcheetah_clip.yaml
python -m surrogate_obj_exp.train --config configs/halfcheetah_kl.yaml
python -m surrogate_obj_exp.train --config configs/halfcheetah_noclip.yaml
python -m surrogate_obj_exp.plot_results --log-dir results --out-dir plots
