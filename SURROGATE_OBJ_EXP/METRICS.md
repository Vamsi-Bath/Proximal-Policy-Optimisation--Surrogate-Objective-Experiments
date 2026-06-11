# Metrics for comparing surrogate objectives and ablations

Use at least 3-10 seeds per setting. The most important metric is sample efficiency: return versus environment steps, not just final return.

## Primary performance metrics

- `episodic_return_mean`: average episodic return collected during the update window.
- `episodic_return_std`: instability across episodes in the same update window.
- `episodic_return_min` and `episodic_return_max`: worst and best local episode returns.
- Area under the learning curve: compute from `episodic_return_mean` versus `global_step`.
- Final average return: average of the last 10-20% of updates.
- Time-to-threshold: first `global_step` at which a method reaches a target return.

## PPO stability metrics

- `approx_kl`: PPO trust-region proxy. High values mean the new policy moved far from the data-collection policy.
- `old_approx_kl`: simple old-logp minus new-logp estimator, useful as a sanity check.
- `clip_fraction`: fraction of samples outside the clip interval. Most relevant for PPO-Clip, but still logged for diagnostics in all variants.
- `early_stop_kl`: whether the update stopped because KL exceeded the threshold.
- `entropy`: exploration proxy. Very low entropy can indicate premature policy collapse.
- `grad_norm`: optimisation stability proxy.

## Critic metrics

- `value_loss`: critic mean-squared error scaled by 0.5.
- `explained_variance`: how much return variance the value function explains. Values near 1 are good; values near or below 0 indicate poor critic fit.

## State-coverage metrics

- `xy_coverage_bins`: number of occupied bins in the 2D trajectory histogram.
- `xy_path_length`: total path length in XY projection during the rollout window.
- `xy_bbox_area`: bounding-box area of the XY trajectory.
- `xy_trajectory.csv`: raw x/y points for plotting state coverage.

For 2D MuJoCo locomotion tasks such as HalfCheetah, y may be zero or a projection fallback because movement is mostly along x. For Ant-like tasks, x/y coverage is more meaningful.

## Ablation metrics

Compare each regularisation setting by:

- final return mean ± standard deviation across seeds
- AUC return mean ± standard deviation across seeds
- average KL and KL violation frequency
- entropy collapse frequency
- explained variance
- state coverage
- robustness across seeds

## Recommended comparison table columns

```text
objective, env_id, seed, l1_coeff, l2_coeff, weight_decay,
shrink_perturb_enabled, final_return_mean, auc_return,
time_to_threshold, mean_kl, kl_violation_rate, final_entropy,
final_explained_variance, xy_coverage_bins
```
