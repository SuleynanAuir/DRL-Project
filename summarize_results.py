from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np

from utils.general import load_from_csv

METRICS = [
    ("Training Rewards", "training_rewards"),
    ("Eval Rewards", "eval_rewards"),
    ("Max Q", "max_q"),
    ("Loss", "loss"),
]


def load_runs(results_dir: Path, env_name: str) -> tuple[np.ndarray, dict[str, np.ndarray], list[Path]]:
    run_dirs = sorted(path for path in results_dir.glob(f"{env_name}_seed*") if path.is_dir())
    if not run_dirs:
        raise FileNotFoundError(f"No runs found for {env_name} in {results_dir}")

    per_metric = {metric: [] for metric, _ in METRICS}
    all_timesteps = []
    for run_dir in run_dirs:
        data = load_from_csv(str(run_dir / "log.csv"))
        all_timesteps.append(data["Timestep"])
        for metric, _ in METRICS:
            per_metric[metric].append(data[metric])

    min_len = min(len(ts) for ts in all_timesteps)
    timesteps = all_timesteps[0][:min_len]
    stacked = {
        metric: np.stack([value[:min_len] for value in values], axis=0)
        for metric, values in per_metric.items()
    }
    return timesteps, stacked, run_dirs


def plot_metric(output_dir: Path, env_name: str, timesteps: np.ndarray, values: np.ndarray, label: str, slug: str) -> None:
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    plt.figure()
    plt.plot(timesteps, mean, label='mean')
    plt.fill_between(timesteps, mean - std, mean + std, alpha=0.2, label='±1 std')
    plt.xlabel('Timesteps')
    plt.ylabel(label)
    plt.title(f'{env_name} {label}')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f'{env_name}_{slug}.png')
    plt.close()


def summarize_env(results_dir: Path, env_name: str) -> None:
    output_dir = results_dir / 'summary'
    output_dir.mkdir(parents=True, exist_ok=True)
    timesteps, stacked, run_dirs = load_runs(results_dir, env_name)

    print(f'ENV {env_name}')
    print('  runs:')
    for run_dir in run_dirs:
        print(f'    - {run_dir}')

    for metric, slug in METRICS:
        values = stacked[metric]
        plot_metric(output_dir, env_name, timesteps, values, metric, slug)
        final_mean = float(values[:, -1].mean())
        final_std = float(values[:, -1].std())
        best_per_seed = values.max(axis=1) if metric != 'Loss' else values.min(axis=1)
        best_mean = float(best_per_seed.mean())
        best_std = float(best_per_seed.std())
        tag = 'best(min)' if metric == 'Loss' else 'best(max)'
        print(f'  {metric}: final={final_mean:.4f}±{final_std:.4f}, {tag}={best_mean:.4f}±{best_std:.4f}')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', default='results_tuned')
    parser.add_argument('--envs', nargs='+', default=['CartPole-v0', 'Acrobot-v1'])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    for env_name in args.envs:
        summarize_env(results_dir, env_name)


if __name__ == '__main__':
    main()
