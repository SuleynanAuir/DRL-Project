from __future__ import annotations

import argparse
import os
from pathlib import Path

try:
    import gymnasium as gym
except ImportError:
    import gym

from main import config, main, seed_all
from utils.general import csv_plot, load_from_csv


ENV_CONFIGS = {
    "CartPole-v0": {
        "num_timesteps": 14000,
        "learning_start": 500,
        "learning_freq": 2,
        "target_update_freq": 100,
        "lr_begin": 2e-4,
        "lr_end": 5e-5,
        "lr_nsteps": 14000,
        "eps_begin": 1.0,
        "eps_end": 0.05,
        "eps_nsteps": 9000,
        "batch_size": 128,
        "gamma": 0.99,
        "tau": 0.003,
        "eval_freq": 500,
        "log_freq": 500,
        "num_episodes_eval": 60,
        "train_reward_window": 6,
        "clip_val": 2.0,
        "target_q_clip": 50.0,
        "normalize_loss": True,
        "high": 1.0,
    },
    "Acrobot-v1": {
        "num_timesteps": 60000,
        "learning_start": 3000,
        "learning_freq": 2,
        "target_update_freq": 100,
        "lr_begin": 1e-4,
        "lr_end": 3e-5,
        "lr_nsteps": 60000,
        "eps_begin": 1.0,
        "eps_end": 0.05,
        "eps_nsteps": 50000,
        "batch_size": 256,
        "gamma": 0.99,
        "tau": 0.001,
        "eval_freq": 1500,
        "log_freq": 1500,
        "num_episodes_eval": 60,
        "train_reward_window": 6,
        "clip_val": 3.0,
        "target_q_clip": 100.0,
        "normalize_loss": True,
        "high": 1.0,
    },
}


def configure_run(env_name: str, seed: int, base_dir: Path) -> Path:
    run_dir = base_dir / f"{env_name}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    for key, value in ENV_CONFIGS[env_name].items():
        setattr(config, key, value)

    config.output_path = str(run_dir) + os.sep
    config.plot_dir = str(run_dir / "rewards.pdf")
    config.train_plot_dir = str(run_dir / "during_training_rewards.pdf")
    config.model_dir = str(run_dir / "model.weights")
    config.csv_dir = str(run_dir / "log.csv")
    return run_dir


def summarize_run(log_path: Path) -> dict[str, float]:
    data = load_from_csv(str(log_path))
    return {
        "final_training_reward": float(data["Training Rewards"][-1]),
        "best_training_reward": float(data["Training Rewards"].max()),
        "final_eval_reward": float(data["Eval Rewards"][-1]),
        "best_eval_reward": float(data["Eval Rewards"].max()),
        "final_max_q": float(data["Max Q"][-1]),
        "peak_max_q": float(data["Max Q"].max()),
        "final_loss": float(data["Loss"][-1]),
        "min_loss": float(data["Loss"].min()),
    }


def run_once(env_name: str, seed: int, results_dir: Path, double: bool = False) -> tuple[Path, dict[str, float]]:
    run_dir = configure_run(env_name, seed, results_dir)
    env = gym.make(env_name)
    seed_all(seed, env)
    try:
        main(env, double)
    finally:
        env.close()
    csv_plot(config.csv_dir, config.output_path)
    return run_dir, summarize_run(Path(config.csv_dir))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True, choices=sorted(ENV_CONFIGS.keys()))
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--double", action="store_true")
    return parser.parse_args()


def main_cli() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    for seed in args.seeds:
        run_dir, summary = run_once(args.env, seed, results_dir, double=args.double)
        print(f"RUN {run_dir}")
        for key, value in summary.items():
            print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main_cli()
