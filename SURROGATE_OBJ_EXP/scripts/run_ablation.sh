#!/usr/bin/env bash
set -euo pipefail

# Generates temporary configs for objective x regularisation x shrink-and-perturb ablations.
# Adjust ENVS, SEEDS, TOTAL_TIMESTEPS for larger studies.

ENVS=("HalfCheetah-v5")
OBJECTIVES=("clip" "kl" "noclip")
SEEDS=(0 1 2)
L1S=(0.0 1e-6 1e-5)
L2S=(0.0 1e-5 1e-4)
SHRINKS=("false" "true")
TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS:-200000}
TMP_DIR="configs/generated_ablation"
mkdir -p "$TMP_DIR"

for ENV_ID in "${ENVS[@]}"; do
  for OBJ in "${OBJECTIVES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
      for L1 in "${L1S[@]}"; do
        for L2 in "${L2S[@]}"; do
          for SP in "${SHRINKS[@]}"; do
            RUN="${ENV_ID}_${OBJ}_seed${SEED}_l1${L1}_l2${L2}_sp${SP}"
            CFG="$TMP_DIR/${RUN}.yaml"
            cat > "$CFG" <<YAML
run_name: $RUN
env_id: $ENV_ID
seed: $SEED
device: auto
out_dir: results
total_timesteps: $TOTAL_TIMESTEPS
rollout_steps: 2048
save_every: 999999
coverage_bins: 50
model:
  hidden_sizes: [64, 64]
  log_std_init: -0.5
ppo:
  objective: $OBJ
  lr: 0.0003
  epochs: 10
  minibatch_size: 256
  gamma: 0.99
  gae_lambda: 0.95
  clip_coef: 0.2
  target_kl: 0.02
  kl_coef: 1.0
  kl_adapt: true
  vf_coef: 0.5
  ent_coef: 0.0
  max_grad_norm: 0.5
  l1_coeff: $L1
  l2_coeff: $L2
  weight_decay: 0.0
shrink_perturb:
  enabled: $SP
  every_updates: 50
  shrink: 0.95
  perturb_std: 0.001
YAML
            python -m surrogate_obj_exp.train --config "$CFG"
          done
        done
      done
    done
  done
done

python -m surrogate_obj_exp.plot_results --log-dir results --out-dir plots
