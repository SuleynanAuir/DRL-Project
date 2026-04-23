from __future__ import annotations

import argparse
from pathlib import Path
import re

import csv
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np


METRICS = [
    ("Training Rewards", "training_rewards"),
    ("Eval Rewards", "eval_rewards"),
    ("Max Q", "max_q"),
    ("Loss", "loss"),
]


def load_csv(path: Path) -> dict[str, np.ndarray]:
    with path.open() as csv_file:
        reader = csv.DictReader(csv_file)
        data = {name: [] for name in reader.fieldnames}
        for row in reader:
            for name in reader.fieldnames:
                data[name].append(float(row[name]))
    data["Timestep"] = np.asarray(data["Timestep"], dtype=np.int64)
    float_keys = ["Training Rewards", "Eval Rewards", "Max Q", "Loss", "Entropy", "KL"]
    for key in float_keys:
        if key in data:
            data[key] = np.asarray(data[key], dtype=np.float32)
    return data


def load_runs(results_dir: Path, env_name: str):
    run_dirs = sorted(path for path in results_dir.glob(f"{env_name}_seed*") if path.is_dir())
    if not run_dirs:
        raise FileNotFoundError(f"No runs found for {env_name} in {results_dir}")

    all_timesteps = []
    per_metric = {metric: [] for metric, _ in METRICS}

    for run_dir in run_dirs:
        data = load_csv(run_dir / "log.csv")
        all_timesteps.append(data["Timestep"])
        for metric, _ in METRICS:
            per_metric[metric].append(data[metric])

    min_len = min(len(item) for item in all_timesteps)
    timesteps = all_timesteps[0][:min_len]
    stacked = {
        metric: np.stack([values[:min_len] for values in series], axis=0)
        for metric, series in per_metric.items()
    }
    return timesteps, stacked, run_dirs


def extract_seed_from_dir(run_dir: Path) -> str:
    match = re.search(r"seed(\d+)", run_dir.name)
    if match:
        return match.group(1)
    return run_dir.name


def plot_metric(output_dir: Path, env_name: str, timesteps: np.ndarray, values: np.ndarray, label: str, slug: str) -> None:
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    plt.figure()
    plt.plot(timesteps, mean, label="mean")
    plt.fill_between(timesteps, mean - std, mean + std, alpha=0.2, label="±1 std")
    plt.xlabel("Timesteps")
    plt.ylabel(label)
    plt.title(f"{env_name} {label}")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{env_name}_{slug}.png")
    plt.close()


def plot_metric_overlay(output_dir: Path, env_name: str, run_dirs: list[Path], timesteps: np.ndarray, values: np.ndarray, label: str, slug: str) -> None:
    plt.figure()
    for idx, run_dir in enumerate(run_dirs):
        seed = extract_seed_from_dir(run_dir)
        plt.plot(timesteps, values[idx], label=f"seed {seed}")
    plt.xlabel("Timesteps")
    plt.ylabel(label)
    plt.title(f"{env_name} {label} (Per Seed)")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{env_name}_{slug}_per_seed.png")
    plt.close()


def summarize_env(results_dir: Path, env_name: str) -> None:
    output_dir = results_dir / "summary"
    output_dir.mkdir(parents=True, exist_ok=True)
    timesteps, stacked, run_dirs = load_runs(results_dir, env_name)

    print(f"ENV {env_name}")
    print("  runs:")
    for run_dir in run_dirs:
        print(f"    - {run_dir}")

    for metric, slug in METRICS:
        values = stacked[metric]
        plot_metric(output_dir, env_name, timesteps, values, metric, slug)
        plot_metric_overlay(output_dir, env_name, run_dirs, timesteps, values, metric, slug)
        final_mean = float(values[:, -1].mean())
        final_std = float(values[:, -1].std())
        if metric == "Loss":
            best_per_seed = values.min(axis=1)
            tag = "best(min)"
        else:
            best_per_seed = values.max(axis=1)
            tag = "best(max)"
        best_mean = float(best_per_seed.mean())
        best_std = float(best_per_seed.std())
        print(f"  {metric}: final={final_mean:.4f}±{final_std:.4f}, {tag}={best_mean:.4f}±{best_std:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="ppo_results")
    parser.add_argument("--envs", nargs="+", default=["CartPole-v0", "Acrobot-v1"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    for env_name in args.envs:
        summarize_env(results_dir, env_name)


if __name__ == "__main__":
    main()
