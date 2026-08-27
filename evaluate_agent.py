"""Evaluation, plots, and behavior reports for trained agents."""

from __future__ import annotations

import csv
import json
import statistics
from dataclasses import replace
from pathlib import Path
from typing import Dict

import matplotlib
import numpy as np
import torch
from torch.distributions import Categorical

matplotlib.use("Agg")

from matplotlib import pyplot as plt

from monster_agent import MonsterActorCritic
from monster_agent import observation_to_tensors
from monster_gridworld import MonsterGridworld, MonsterGridworldConfig
from rollout_collector import RolloutCollector
from rollout_animation import RolloutFrame, animate_rollout


@torch.no_grad()
def evaluate_policy(
    model: MonsterActorCritic,
    env_config: MonsterGridworldConfig,
    *,
    n_episodes: int,
    seed: int,
    device: torch.device,
    deterministic: bool,
) -> Dict[str, Dict[str, float]]:
    torch.manual_seed(seed)
    collector = RolloutCollector(
        env_config, n_envs=n_episodes, seed=seed, deterministic=deterministic
    )
    batch = collector.collect(model, device)
    summary = {}
    for name, values in batch.episode_metrics.items():
        array = values.numpy()
        summary[name] = {
            "mean": float(array.mean()),
            "std": float(array.std()),
        }
    return summary


def evaluate_standard_and_counterfactual(
    model: MonsterActorCritic,
    training_config: MonsterGridworldConfig,
    *,
    n_episodes: int,
    seed: int,
    device: torch.device,
) -> Dict[str, object]:
    long_config = replace(training_config, episode_length=200)
    no_monsters_config = replace(long_config, n_monsters=0)
    was_training = model.training
    model.eval()
    result = {}
    settings = {
        "training_environment": training_config,
        "long_environment": long_config,
        "no_monsters_counterfactual": no_monsters_config,
    }
    for selection_name, deterministic in (("stochastic", False), ("argmax", True)):
        for index, (setting_name, config) in enumerate(settings.items()):
            result[f"{setting_name}_{selection_name}"] = evaluate_policy(
                model,
                config,
                n_episodes=n_episodes,
                seed=seed + index * 10_000,
                device=device,
                deterministic=deterministic,
            )
    model.train(was_training)
    return result


def plot_training_curves(metrics_path: Path, output_path: Path) -> None:
    with metrics_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        return

    steps = np.array([float(row["environment_steps"]) for row in rows])
    reward = np.array([float(row["mean_return"]) for row in rows])
    apples = np.array([float(row["mean_apples_collected"]) for row in rows])
    shields = np.array([float(row["mean_shields_collected"]) for row in rows])
    attacks = np.array([float(row["mean_unshielded_attacks"]) for row in rows])

    figure, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    axes[0].plot(steps, reward, alpha=0.3, label="batch mean")
    axes[0].plot(steps, _moving_average(reward), label="moving average")
    axes[0].set_ylabel("Episode return")
    axes[0].legend()

    axes[1].plot(steps, _moving_average(apples), label="apples")
    axes[1].plot(steps, _moving_average(shields), label="shields")
    axes[1].set_ylabel("Pickups per episode")
    axes[1].legend()

    axes[2].plot(steps, attacks, alpha=0.25, label="batch mean")
    axes[2].plot(steps, _moving_average(attacks), label="moving average")
    axes[2].set_xlabel("Environment steps")
    axes[2].set_ylabel("Unshielded attacks")
    axes[2].legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def write_behavior_report(
    summary: Dict[str, object],
    *,
    json_path: Path,
    markdown_path: Path,
) -> None:
    json_path.write_text(json.dumps(summary, indent=2) + "\n")
    lines = ["# Learned behavior summary", ""]
    argmax_metrics = summary.get("training_environment_argmax")
    stochastic_metrics = summary.get("training_environment_stochastic")
    no_monsters_metrics = summary.get("no_monsters_counterfactual_stochastic")
    if argmax_metrics and stochastic_metrics and no_monsters_metrics:
        action_metrics = {
            name.removeprefix("action_").removesuffix("_fraction"): values["mean"]
            for name, values in argmax_metrics.items()
            if name.startswith("action_")
        }
        dominant_action, dominant_fraction = max(
            action_metrics.items(), key=lambda item: item[1]
        )
        lines.extend(
            (
                "## Highlights",
                "",
                f"- Sampled-policy return in the training environment: "
                f"{stochastic_metrics['return']['mean']:.3f}.",
                f"- Argmax action: {dominant_action} "
                f"({dominant_fraction:.1%} of actions).",
                f"- Sampled policy spends "
                f"{no_monsters_metrics['corner_fraction']['mean']:.1%} of the "
                "no-monsters rollout in corners.",
                f"- No-monsters pickups: "
                f"{no_monsters_metrics['apples_collected']['mean']:.3f} apples and "
                f"{no_monsters_metrics['shields_collected']['mean']:.3f} shields.",
                "",
            )
        )
    for setting, metrics in summary.items():
        lines.extend((f"## {setting.replace('_', ' ').title()}", ""))
        lines.append("| Metric | Mean | Std |")
        lines.append("|---|---:|---:|")
        for name, values in metrics.items():
            lines.append(
                f"| {name.replace('_', ' ')} | {values['mean']:.3f} | "
                f"{values['std']:.3f} |"
            )
        lines.append("")
    markdown_path.write_text("\n".join(lines))


@torch.no_grad()
def save_policy_animation(
    model: MonsterActorCritic,
    env_config: MonsterGridworldConfig,
    output_path: Path,
    *,
    seed: int,
    device: torch.device,
    deterministic: bool = False,
) -> None:
    torch.manual_seed(seed)
    env = MonsterGridworld(env_config)
    observation, info = env.reset(seed=seed)
    frames = [
        RolloutFrame(
            grid=np.asarray(observation["grid"]).copy(),
            step=0,
            action=None,
            reward=0.0,
            total_reward=0.0,
            shield_inventory=info["shield_inventory"],
        )
    ]
    state = None
    previous_action = torch.tensor([-1], device=device)
    previous_reward = torch.tensor([0.0], device=device)
    total_reward = 0.0
    was_training = model.training
    model.eval()

    for _ in range(env_config.episode_length):
        grid, inventory = observation_to_tensors(observation, device=device)
        output = model(
            grid, inventory, previous_action, previous_reward, state
        )
        if deterministic:
            action = output.policy_logits.argmax(dim=-1)
        else:
            action = Categorical(logits=output.policy_logits).sample()
        observation, reward, terminated, truncated, info = env.step(action.item())
        total_reward += reward
        frames.append(
            RolloutFrame(
                grid=np.asarray(observation["grid"]).copy(),
                step=info["step"],
                action=action.item(),
                reward=reward,
                total_reward=total_reward,
                shield_inventory=info["shield_inventory"],
            )
        )
        state = output.state
        previous_action = action
        previous_reward = torch.tensor([reward], device=device)
        if terminated or truncated:
            break

    label = "Learned policy (argmax)" if deterministic else "Learned policy (sampled)"
    animation = animate_rollout(frames, label=label)
    animation.save(output_path, writer="pillow", fps=6)
    plt.close(animation._fig)
    model.train(was_training)


def write_training_summary(metrics_path: Path, output_path: Path) -> None:
    with metrics_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        return

    window = max(1, len(rows) // 10)
    early = rows[:window]
    late = rows[-window:]
    metric_names = (
        "mean_return",
        "mean_apples_collected",
        "mean_shields_collected",
        "mean_unshielded_attacks",
        "entropy",
        "value_loss",
    )

    def mean(part, name):
        return statistics.mean(float(row[name]) for row in part)

    lines = [
        "# Training summary",
        "",
        f"Total environment steps: {int(float(rows[-1]['environment_steps'])):,}",
        "",
        "| Metric | First 10% | Last 10% | Change |",
        "|---|---:|---:|---:|",
    ]
    for name in metric_names:
        early_mean = mean(early, name)
        late_mean = mean(late, name)
        lines.append(
            f"| {name.replace('_', ' ')} | {early_mean:.3f} | "
            f"{late_mean:.3f} | {late_mean - early_mean:+.3f} |"
        )
    elapsed = float(rows[-1]["elapsed_seconds"])
    steps = float(rows[-1]["environment_steps"])
    lines.extend(("", f"Observed throughput: {steps / elapsed:,.0f} environment steps/s.", ""))
    output_path.write_text("\n".join(lines))


def _moving_average(values: np.ndarray, window: int = 50) -> np.ndarray:
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    smoothed = np.convolve(values, kernel, mode="valid")
    prefix = np.full(window - 1, np.nan)
    return np.concatenate((prefix, smoothed))
