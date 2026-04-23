from __future__ import annotations

import argparse
import csv
import os
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

try:
    import gymnasium as gym
except ImportError:
    import gym


@dataclass
class PPOConfig:
    env_name: str
    seed: int
    total_timesteps: int
    rollout_steps: int
    gamma: float
    gae_lambda: float
    clip_ratio: float
    value_coef: float
    entropy_coef: float
    learning_rate: float
    max_grad_norm: float
    update_epochs: int
    minibatch_size: int
    hidden_size: int
    eval_freq: int
    eval_episodes: int
    train_reward_window: int
    results_dir: str
    value_clip_ratio: float = 0.2
    target_kl: float = 0.005
    entropy_coef_end: float = 0.0
    loss_ema_decay: float = 0.97
    normalize_value_loss: bool = True
    value_loss_eps: float = 1e-3
    value_coef_end_ratio: float = 0.1


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_size: int):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden_size, act_dim)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(obs)
        return self.policy_head(features), self.value_head(features).squeeze(-1)

    def get_action_value(
        self,
        obs: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, value


class RolloutBuffer:
    def __init__(self, rollout_steps: int, obs_dim: int):
        self.obs = np.zeros((rollout_steps, obs_dim), dtype=np.float32)
        self.actions = np.zeros((rollout_steps,), dtype=np.int64)
        self.log_probs = np.zeros((rollout_steps,), dtype=np.float32)
        self.rewards = np.zeros((rollout_steps,), dtype=np.float32)
        self.dones = np.zeros((rollout_steps,), dtype=np.float32)
        self.values = np.zeros((rollout_steps,), dtype=np.float32)
        self.returns = np.zeros((rollout_steps,), dtype=np.float32)
        self.advantages = np.zeros((rollout_steps,), dtype=np.float32)
        self.step = 0
        self.max_steps = rollout_steps

    def add(
        self,
        obs: np.ndarray,
        action: int,
        log_prob: float,
        reward: float,
        done: bool,
        value: float,
    ) -> None:
        index = self.step
        self.obs[index] = obs
        self.actions[index] = action
        self.log_probs[index] = log_prob
        self.rewards[index] = reward
        self.dones[index] = float(done)
        self.values[index] = value
        self.step += 1

    def compute_returns_advantages(
        self,
        last_value: float,
        last_done: bool,
        gamma: float,
        gae_lambda: float,
    ) -> None:
        gae = 0.0
        next_value = last_value
        next_non_terminal = 1.0 - float(last_done)
        for index in reversed(range(self.max_steps)):
            delta = (
                self.rewards[index]
                + gamma * next_value * next_non_terminal
                - self.values[index]
            )
            gae = delta + gamma * gae_lambda * next_non_terminal * gae
            self.advantages[index] = gae
            self.returns[index] = gae + self.values[index]
            next_value = self.values[index]
            next_non_terminal = 1.0 - self.dones[index]

    def get_tensors(self, device: torch.device) -> dict[str, torch.Tensor]:
        advantages = self.advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return {
            "obs": torch.as_tensor(self.obs, dtype=torch.float32, device=device),
            "actions": torch.as_tensor(self.actions, dtype=torch.long, device=device),
            "log_probs": torch.as_tensor(self.log_probs, dtype=torch.float32, device=device),
            "returns": torch.as_tensor(self.returns, dtype=torch.float32, device=device),
            "advantages": torch.as_tensor(advantages, dtype=torch.float32, device=device),
            "old_values": torch.as_tensor(self.values, dtype=torch.float32, device=device),
        }


def set_seed(seed: int, env) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        env.reset(seed=seed)
    except TypeError:
        if hasattr(env, "seed"):
            env.seed(seed)
    if hasattr(env.action_space, "seed"):
        env.action_space.seed(seed)


def extract_state(reset_output):
    if isinstance(reset_output, tuple):
        return reset_output[0]
    return reset_output


def extract_step(step_output):
    if len(step_output) == 5:
        next_state, reward, terminated, truncated, info = step_output
        done = terminated or truncated
        return next_state, reward, done, info
    return step_output


def evaluate_policy(
    env,
    model: ActorCritic,
    episodes: int,
    device: torch.device,
) -> float:
    rewards = []
    for _ in range(episodes):
        state = extract_state(env.reset())
        done = False
        episode_reward = 0.0
        while not done:
            obs_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                logits, _ = model.forward(obs_tensor)
                action = torch.argmax(logits, dim=-1).item()
            next_state, reward, done, _ = extract_step(env.step(action))
            state = next_state
            episode_reward += float(reward)
        rewards.append(episode_reward)
    return float(np.mean(rewards))


def export_plot(values: list[float], ylabel: str, path: str) -> None:
    plt.figure()
    plt.plot(range(len(values)), values)
    plt.xlabel("Checkpoint")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def train(config: PPOConfig) -> tuple[str, dict[str, float]]:
    run_dir = Path(config.results_dir) / f"{config.env_name}_seed{config.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "log.csv"
    model_path = run_dir / "model.pt"

    env = gym.make(config.env_name)
    eval_env = gym.make(config.env_name)
    set_seed(config.seed, env)
    set_seed(config.seed + 1234, eval_env)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ActorCritic(obs_dim, act_dim, config.hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    state = extract_state(env.reset())
    episode_return = 0.0
    train_rewards = deque(maxlen=config.train_reward_window)
    train_reward_series: list[float] = []
    eval_reward_series: list[float] = []
    max_q_series: list[float] = []
    loss_series: list[float] = []
    loss_ema: float | None = None
    rollout_max_q_values: list[float] = []

    fieldnames = ["Timestep", "Training Rewards", "Eval Rewards", "Max Q", "Loss", "Entropy", "KL"]
    with log_path.open("w", newline="") as log_file:
        writer = csv.DictWriter(log_file, fieldnames=fieldnames)
        writer.writeheader()

        timesteps = 0
        next_eval = config.eval_freq
        while timesteps < config.total_timesteps:
            rollout = RolloutBuffer(config.rollout_steps, obs_dim)
            for _ in range(config.rollout_steps):
                obs_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    action_tensor, log_prob_tensor, _, value_tensor = model.get_action_value(obs_tensor)
                    logits, value_pred = model.forward(obs_tensor)
                action = int(action_tensor.item())
                log_prob = float(log_prob_tensor.item())
                value = float(value_tensor.item())
                rollout_max_q_values.append(float(value_pred.item()))

                next_state, reward, done, _ = extract_step(env.step(action))
                rollout.add(state, action, log_prob, float(reward), done, value)

                timesteps += 1
                episode_return += float(reward)
                state = next_state

                if done:
                    train_rewards.append(episode_return)
                    episode_return = 0.0
                    state = extract_state(env.reset())

                if timesteps >= config.total_timesteps:
                    break

            with torch.no_grad():
                last_obs = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                _, last_value = model.forward(last_obs)
                last_value = float(last_value.item())

            rollout.compute_returns_advantages(
                last_value=last_value,
                last_done=False,
                gamma=config.gamma,
                gae_lambda=config.gae_lambda,
            )

            batch = rollout.get_tensors(device)
            num_samples = batch["obs"].shape[0]
            indices = np.arange(num_samples)

            progress = min(float(timesteps) / float(config.total_timesteps), 1.0)
            lr_now = max(config.learning_rate * (1.0 - progress),
                         config.learning_rate * 0.1)
            for group in optimizer.param_groups:
                group["lr"] = max(lr_now, 1e-6)
            entropy_coef_now = (
                config.entropy_coef
                + (config.entropy_coef_end - config.entropy_coef) * progress
            )
            value_coef_now = (
                config.value_coef
                * ((1.0 - progress) + config.value_coef_end_ratio * progress)
            )

            epoch_losses = []
            epoch_value_losses = []
            epoch_entropies = []
            epoch_kls = []
            stop_early = False

            for _ in range(config.update_epochs):
                np.random.shuffle(indices)
                for start in range(0, num_samples, config.minibatch_size):
                    end = start + config.minibatch_size
                    mb_idx = indices[start:end]
                    mb_obs = batch["obs"][mb_idx]
                    mb_actions = batch["actions"][mb_idx]
                    mb_old_log_probs = batch["log_probs"][mb_idx]
                    mb_advantages = batch["advantages"][mb_idx]
                    mb_returns = batch["returns"][mb_idx]
                    mb_old_values = batch["old_values"][mb_idx]

                    _, new_log_probs, entropy, values = model.get_action_value(mb_obs, mb_actions)
                    ratio = (new_log_probs - mb_old_log_probs).exp()
                    clipped_ratio = torch.clamp(ratio, 1.0 - config.clip_ratio, 1.0 + config.clip_ratio)

                    policy_loss = -torch.min(ratio * mb_advantages, clipped_ratio * mb_advantages).mean()
                    unclipped_value_loss = (values - mb_returns) ** 2
                    value_clipped = mb_old_values + torch.clamp(
                        values - mb_old_values,
                        -config.value_clip_ratio,
                        config.value_clip_ratio,
                    )
                    clipped_value_loss = (value_clipped - mb_returns) ** 2
                    value_loss = 0.5 * torch.max(unclipped_value_loss, clipped_value_loss).mean()
                    if config.normalize_value_loss:
                        value_scale = mb_returns.detach().var(unbiased=False).clamp_min(config.value_loss_eps)
                        value_loss_for_opt = value_loss / value_scale
                    else:
                        value_loss_for_opt = value_loss
                    entropy_loss = entropy.mean()

                    loss = (
                        policy_loss
                        + value_coef_now * value_loss_for_opt
                        - entropy_coef_now * entropy_loss
                    )

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    optimizer.step()

                    approx_kl = (mb_old_log_probs - new_log_probs).mean().item()
                    epoch_losses.append(float(loss.item()))
                    epoch_value_losses.append(float((value_coef_now * value_loss_for_opt).item()))
                    epoch_entropies.append(float(entropy_loss.item()))
                    epoch_kls.append(float(approx_kl))
                    if approx_kl > config.target_kl:
                        stop_early = True
                        break
                if stop_early:
                    break

            mean_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
            mean_value_loss = float(np.mean(epoch_value_losses)) if epoch_value_losses else 0.0
            mean_entropy = float(np.mean(epoch_entropies)) if epoch_entropies else 0.0
            mean_kl = float(np.mean(epoch_kls)) if epoch_kls else 0.0
            if loss_ema is None:
                loss_ema = mean_value_loss
            else:
                loss_ema = config.loss_ema_decay * loss_ema + (1.0 - config.loss_ema_decay) * mean_value_loss
            loss_series.append(float(loss_ema))

            if timesteps >= next_eval or timesteps >= config.total_timesteps:
                eval_reward = evaluate_policy(eval_env, model, config.eval_episodes, device)
                avg_train_reward = float(np.mean(train_rewards)) if len(train_rewards) > 0 else 0.0
                avg_max_q = float(np.percentile(rollout_max_q_values, 95)) if len(rollout_max_q_values) > 0 else 0.0
                train_reward_series.append(avg_train_reward)
                eval_reward_series.append(eval_reward)
                max_q_series.append(avg_max_q)
                row = {
                    "Timestep": timesteps,
                    "Training Rewards": avg_train_reward,
                    "Eval Rewards": eval_reward,
                    "Max Q": avg_max_q,
                    "Loss": float(loss_ema),
                    "Entropy": mean_entropy,
                    "KL": mean_kl,
                }
                writer.writerow(row)
                log_file.flush()
                print(
                    f"Timestep {timesteps} | Train R {avg_train_reward:.2f} | "
                    f"Eval R {eval_reward:.2f} | Max Q {avg_max_q:.4f} | "
                    f"Loss {loss_ema:.4f} | "
                    f"Entropy {mean_entropy:.4f} | KL {mean_kl:.4f}"
                )
                rollout_max_q_values = []
                next_eval += config.eval_freq

    torch.save(model.state_dict(), model_path)

    export_plot(train_reward_series, "Training Rewards", str(run_dir / "training_rewards.png"))
    export_plot(eval_reward_series, "Eval Rewards", str(run_dir / "eval_rewards.png"))
    export_plot(max_q_series, "Max Q", str(run_dir / "max_q.png"))
    export_plot(loss_series, "Loss", str(run_dir / "loss.png"))

    env.close()
    eval_env.close()

    summary = {
        "final_training_reward": float(train_reward_series[-1]) if train_reward_series else 0.0,
        "best_training_reward": float(np.max(train_reward_series)) if train_reward_series else 0.0,
        "final_eval_reward": float(eval_reward_series[-1]) if eval_reward_series else 0.0,
        "best_eval_reward": float(np.max(eval_reward_series)) if eval_reward_series else 0.0,
        "final_max_q": float(max_q_series[-1]) if max_q_series else 0.0,
        "peak_max_q": float(np.max(max_q_series)) if max_q_series else 0.0,
        "final_loss": float(loss_series[-1]) if loss_series else 0.0,
        "min_loss": float(np.min(loss_series)) if loss_series else 0.0,
    }
    return str(run_dir), summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True, choices=["CartPole-v0", "Acrobot-v1"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results-dir", default="ppo_results")
    parser.add_argument("--total-timesteps", type=int)
    parser.add_argument("--rollout-steps", type=int)
    parser.add_argument("--eval-freq", type=int)
    parser.add_argument("--eval-episodes", type=int)
    return parser.parse_args()


def default_config(args: argparse.Namespace) -> PPOConfig:
    env_defaults = {
        "CartPole-v0": {
            "total_timesteps": 30000,
            "rollout_steps": 1024,
            "learning_rate": 2e-4,
            "update_epochs": 5,
            "minibatch_size": 256,
            "entropy_coef": 0.006,
            "eval_freq": 2048,
            "eval_episodes": 60,
            "train_reward_window": 10,
        },
        "Acrobot-v1": {
            "total_timesteps": 120000,
            "rollout_steps": 1024,
            "learning_rate": 1.5e-4,
            "update_epochs": 6,
            "minibatch_size": 256,
            "entropy_coef": 0.012,
            "eval_freq": 4096,
            "eval_episodes": 60,
            "train_reward_window": 15,
        },
    }
    defaults = env_defaults[args.env]
    return PPOConfig(
        env_name=args.env,
        seed=args.seed,
        total_timesteps=(args.total_timesteps if args.total_timesteps is not None else defaults["total_timesteps"]),
        rollout_steps=(args.rollout_steps if args.rollout_steps is not None else defaults["rollout_steps"]),
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        value_coef=0.5,
        entropy_coef=defaults["entropy_coef"],
        learning_rate=defaults["learning_rate"],
        max_grad_norm=0.4,
        update_epochs=defaults["update_epochs"],
        minibatch_size=defaults["minibatch_size"],
        hidden_size=128,
        eval_freq=(args.eval_freq if args.eval_freq is not None else defaults["eval_freq"]),
        eval_episodes=(args.eval_episodes if args.eval_episodes is not None else defaults["eval_episodes"]),
        train_reward_window=defaults["train_reward_window"],
        results_dir=args.results_dir,
    )


def main() -> None:
    args = parse_args()
    cfg = default_config(args)
    run_dir, summary = train(cfg)
    print(f"RUN {run_dir}")
    for key, value in summary.items():
        print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()
