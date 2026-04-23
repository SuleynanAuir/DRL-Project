from __future__ import annotations

import argparse
from pathlib import Path

from ppo_train import PPOConfig, train


ENV_CONFIGS = {
    "CartPole-v0": {
        "total_timesteps": 30000,
        "rollout_steps": 1024,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_ratio": 0.2,
        "value_coef": 0.5,
        "entropy_coef": 0.006,
        "learning_rate": 2e-4,
        "max_grad_norm": 0.4,
        "update_epochs": 5,
        "minibatch_size": 256,
        "hidden_size": 128,
        "eval_freq": 2048,
        "eval_episodes": 60,
        "train_reward_window": 10,
    },
    "Acrobot-v1": {
        "total_timesteps": 120000,
        "rollout_steps": 1024,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_ratio": 0.2,
        "value_coef": 0.5,
        "entropy_coef": 0.012,
        "learning_rate": 1.5e-4,
        "max_grad_norm": 0.4,
        "update_epochs": 6,
        "minibatch_size": 256,
        "hidden_size": 128,
        "eval_freq": 4096,
        "eval_episodes": 60,
        "train_reward_window": 15,
    },
}


def build_config(env_name: str, seed: int, results_dir: str) -> PPOConfig:
    base = ENV_CONFIGS[env_name]
    return PPOConfig(
        env_name=env_name,
        seed=seed,
        total_timesteps=base["total_timesteps"],
        rollout_steps=base["rollout_steps"],
        gamma=base["gamma"],
        gae_lambda=base["gae_lambda"],
        clip_ratio=base["clip_ratio"],
        value_coef=base["value_coef"],
        entropy_coef=base["entropy_coef"],
        learning_rate=base["learning_rate"],
        max_grad_norm=base["max_grad_norm"],
        update_epochs=base["update_epochs"],
        minibatch_size=base["minibatch_size"],
        hidden_size=base["hidden_size"],
        eval_freq=base["eval_freq"],
        eval_episodes=base["eval_episodes"],
        train_reward_window=base["train_reward_window"],
        results_dir=results_dir,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True, choices=sorted(ENV_CONFIGS.keys()))
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--results-dir", default="ppo_results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)

    for seed in args.seeds:
        cfg = build_config(args.env, seed, args.results_dir)
        run_dir, summary = train(cfg)
        print(f"RUN {run_dir}")
        for key, value in summary.items():
            print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()
