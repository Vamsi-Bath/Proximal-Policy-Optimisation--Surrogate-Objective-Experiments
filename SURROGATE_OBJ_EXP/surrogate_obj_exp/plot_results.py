from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def read_all(log_dir: Path):
    rows = []
    for csv_path in log_dir.glob('*/train_metrics.csv'):
        df = pd.read_csv(csv_path)
        df['run'] = csv_path.parent.name
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def plot_metric(df: pd.DataFrame, metric: str, out_dir: Path):
    if metric not in df.columns or df.empty:
        return
    plt.figure(figsize=(10, 6))
    for (objective, run), g in df.groupby(['objective', 'run']):
        g = g.sort_values('global_step')
        plt.plot(g['global_step'], g[metric], alpha=0.35, label=f'{objective}:{run}')
    mean_df = df.groupby(['objective', 'global_step'], as_index=False)[metric].mean()
    for objective, g in mean_df.groupby('objective'):
        g = g.sort_values('global_step')
        plt.plot(g['global_step'], g[metric], linewidth=2.5, label=f'{objective} mean')
    plt.xlabel('Environment steps')
    plt.ylabel(metric)
    plt.title(metric.replace('_', ' ').title())
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / f'{metric}.png', dpi=160)
    plt.close()


def plot_xy(log_dir: Path, out_dir: Path):
    for xy_path in log_dir.glob('*/xy_trajectory.csv'):
        df = pd.read_csv(xy_path)
        if df.empty or not {'x', 'y'}.issubset(df.columns):
            continue
        # Downsample for readable files.
        if len(df) > 30_000:
            df = df.iloc[:: max(1, len(df) // 30_000)]
        plt.figure(figsize=(8, 6))
        plt.plot(df['x'], df['y'], linewidth=0.7)
        plt.xlabel('x position / projection')
        plt.ylabel('y position / projection')
        plt.title(f'XY state coverage: {xy_path.parent.name}')
        plt.tight_layout()
        plt.savefig(out_dir / f'xy_trajectory_{xy_path.parent.name}.png', dpi=160)
        plt.close()


def write_summary(df: pd.DataFrame, out_dir: Path):
    if df.empty:
        return
    metrics = [
        'episodic_return_mean', 'episodic_length_mean', 'approx_kl', 'entropy',
        'value_loss', 'policy_loss', 'explained_variance', 'clip_fraction',
        'grad_norm', 'xy_coverage_bins', 'xy_path_length', 'xy_bbox_area'
    ]
    available = [m for m in metrics if m in df.columns]
    summary = df.groupby(['objective', 'l1_coeff', 'l2_coeff', 'weight_decay', 'shrink_perturb_enabled'])[available].agg(['mean', 'std', 'max', 'min'])
    summary.to_csv(out_dir / 'aggregate_summary.csv')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-dir', default='results')
    parser.add_argument('--out-dir', default='plots')
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = read_all(log_dir)
    if df.empty:
        print(f'No train_metrics.csv files found under {log_dir}')
        return
    df.to_csv(out_dir / 'all_train_metrics.csv', index=False)

    for metric in [
        'episodic_return_mean', 'episodic_length_mean', 'approx_kl', 'old_approx_kl',
        'entropy', 'policy_loss', 'value_loss', 'explained_variance', 'clip_fraction',
        'grad_norm', 'xy_coverage_bins', 'xy_path_length', 'xy_bbox_area'
    ]:
        plot_metric(df, metric, out_dir)
    plot_xy(log_dir, out_dir)
    write_summary(df, out_dir)
    print(f'Plots and summaries written to {out_dir}')


if __name__ == '__main__':
    main()
