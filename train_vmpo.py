"""Train a Monster Gridworld policy with discrete V-MPO."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import time
from dataclasses import asdict, dataclass
from itertools import chain
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from evaluate_agent import (
    evaluate_standard_and_counterfactual,
    plot_training_curves,
    save_policy_animation,
    write_behavior_report,
    write_training_summary,
)
from monster_agent import MonsterActorCritic, PAPER_ADAM_BETAS, PAPER_LEARNING_RATE
from monster_gridworld import MonsterGridworldConfig
from rollout_collector import RolloutCollector
from vmpo import DiscreteVMPOLoss, discounted_returns


@dataclass
class TrainingConfig:
    total_steps: int = 1_000_000
    seed: int = 42
    n_envs: int = 32
    episode_length: int = 25
    gamma: float = 0.99
    target_update_period: int = 10
    epsilon_eta: float = 0.1
    epsilon_alpha: float = 0.01
    gradient_clip: float = 1.0
    log_every_updates: int = 25
    checkpoint_every_steps: int = 250_000
    evaluation_episodes: int = 32
    device: str = "auto"
    run_dir: str = "runs/vmpo_seed42_1m"
    resume_from: str | None = None


METRIC_FIELDS = (
    "update",
    "environment_steps",
    "elapsed_seconds",
    "mean_return",
    "std_return",
    "mean_apples_collected",
    "mean_shields_collected",
    "mean_monsters_destroyed",
    "mean_unshielded_attacks",
    "mean_final_inventory",
    "total_loss",
    "policy_loss",
    "value_loss",
    "kl",
    "entropy",
    "eta",
    "alpha",
)


def train(config: TrainingConfig) -> Path:
    _seed_everything(config.seed)
    device = _resolve_device(config.device)
    run_dir = Path(config.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.csv"
    if metrics_path.exists() and config.resume_from is None:
        raise FileExistsError(f"refusing to overwrite existing run: {run_dir}")
    (run_dir / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")

    env_config = MonsterGridworldConfig(episode_length=config.episode_length, n_apples=25)
    model = MonsterActorCritic().to(device)
    target_model = copy.deepcopy(model).to(device).eval()
    for parameter in target_model.parameters():
        parameter.requires_grad_(False)

    objective = DiscreteVMPOLoss(
        epsilon_eta=config.epsilon_eta,
        epsilon_alpha=config.epsilon_alpha,
    ).to(device)
    parameters = list(chain(model.parameters(), objective.parameters()))
    optimizer = torch.optim.Adam(
        parameters, lr=PAPER_LEARNING_RATE, betas=PAPER_ADAM_BETAS
    )
    collector = RolloutCollector(
        env_config, n_envs=config.n_envs, seed=config.seed
    )

    environment_steps = 0
    update = 0
    elapsed_offset = 0.0
    if config.resume_from is not None:
        checkpoint = torch.load(
            config.resume_from, map_location=device, weights_only=False
        )
        model.load_state_dict(checkpoint["model"])
        target_model.load_state_dict(checkpoint["model"])
        objective.load_state_dict(checkpoint["objective"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        environment_steps = checkpoint["environment_steps"]
        update = checkpoint["update"]
        if "torch_rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if "numpy_rng_state" in checkpoint:
            np.random.set_state(checkpoint["numpy_rng_state"])
        if "python_rng_state" in checkpoint:
            random.setstate(checkpoint["python_rng_state"])
        if metrics_path.exists():
            with metrics_path.open(newline="") as stream:
                previous_rows = list(csv.DictReader(stream))
            if previous_rows:
                elapsed_offset = float(previous_rows[-1]["elapsed_seconds"])
        collector = RolloutCollector(
            env_config,
            n_envs=config.n_envs,
            seed=config.seed + environment_steps,
        )

    started = time.perf_counter()
    next_checkpoint = (
        environment_steps // config.checkpoint_every_steps + 1
    ) * config.checkpoint_every_steps

    file_mode = "a" if config.resume_from is not None else "w"
    with metrics_path.open(file_mode, newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=METRIC_FIELDS)
        if file_mode == "w":
            writer.writeheader()

        while environment_steps < config.total_steps:
            batch = collector.collect(target_model, device)
            batch_on_device = batch.to(device)

            with torch.no_grad():
                target_output = target_model(
                    batch_on_device.grids,
                    batch_on_device.inventories,
                    batch_on_device.previous_actions,
                    batch_on_device.previous_rewards,
                )
                returns = discounted_returns(batch_on_device.rewards, config.gamma)
                model.value_head.update_statistics(returns)

            model.train()
            output = model(
                batch_on_device.grids,
                batch_on_device.inventories,
                batch_on_device.previous_actions,
                batch_on_device.previous_rewards,
            )

            losses = objective(
                output.policy_logits,
                output.value,
                target_output.policy_logits,
                batch_on_device.actions,
                returns,
                output.normalized_value,
                model.value_head.normalize(returns),
            )
            optimizer.zero_grad(set_to_none=True)
            losses.total.backward()
            torch.nn.utils.clip_grad_norm_(parameters, config.gradient_clip)
            optimizer.step()

            update += 1
            environment_steps += batch.n_steps
            if update % config.target_update_period == 0:
                target_model.load_state_dict(model.state_dict())

            row = _metric_row(
                update,
                environment_steps,
                elapsed_offset + time.perf_counter() - started,
                batch.episode_metrics,
                losses,
            )
            writer.writerow(row)
            stream.flush()

            if update == 1 or update % config.log_every_updates == 0:
                print(
                    f"steps={environment_steps:,} update={update:,} "
                    f"return={row['mean_return']:.2f} "
                    f"apples={row['mean_apples_collected']:.2f} "
                    f"shields={row['mean_shields_collected']:.2f} "
                    f"loss={row['total_loss']:.3f}",
                    flush=True,
                )

            if environment_steps >= next_checkpoint:
                _save_checkpoint(
                    run_dir / f"checkpoint_{environment_steps}.pt",
                    model,
                    objective,
                    optimizer,
                    config,
                    environment_steps,
                    update,
                )
                next_checkpoint += config.checkpoint_every_steps

    checkpoint_path = run_dir / "checkpoint_final.pt"
    _save_checkpoint(
        checkpoint_path,
        model,
        objective,
        optimizer,
        config,
        environment_steps,
        update,
    )
    plot_training_curves(metrics_path, run_dir / "training_curves.png")
    write_training_summary(metrics_path, run_dir / "training_summary.md")
    summary = evaluate_standard_and_counterfactual(
        model,
        env_config,
        n_episodes=config.evaluation_episodes,
        seed=config.seed + 100_000,
        device=device,
    )
    write_behavior_report(
        summary,
        json_path=run_dir / "behavior_summary.json",
        markdown_path=run_dir / "behavior_summary.md",
    )
    save_policy_animation(
        model,
        MonsterGridworldConfig(episode_length=200),
        run_dir / "learned_rollout.gif",
        seed=config.seed + 200_000,
        device=device,
    )
    print(f"completed {environment_steps:,} steps; outputs: {run_dir}", flush=True)
    return checkpoint_path


def _metric_row(update, steps, elapsed, episode_metrics, losses) -> Dict[str, float]:
    def mean(name):
        return float(episode_metrics[name].mean())

    return {
        "update": update,
        "environment_steps": steps,
        "elapsed_seconds": elapsed,
        "mean_return": mean("return"),
        "std_return": float(episode_metrics["return"].std()),
        "mean_apples_collected": mean("apples_collected"),
        "mean_shields_collected": mean("shields_collected"),
        "mean_monsters_destroyed": mean("monsters_destroyed"),
        "mean_unshielded_attacks": mean("unshielded_attacks"),
        "mean_final_inventory": mean("final_inventory"),
        "total_loss": float(losses.total.detach()),
        "policy_loss": float(losses.policy.detach()),
        "value_loss": float(losses.value.detach()),
        "kl": float(losses.kl.detach()),
        "entropy": float(losses.entropy.detach()),
        "eta": float(losses.eta.detach()),
        "alpha": float(losses.alpha.detach()),
    }


def _save_checkpoint(path, model, objective, optimizer, config, steps, update):
    torch.save(
        {
            "model": model.state_dict(),
            "objective": objective.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": asdict(config),
            "environment_steps": steps,
            "update": update,
            "torch_rng_state": torch.get_rng_state(),
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
        },
        path,
    )


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-envs", type=int, default=32)
    parser.add_argument("--episode-length", type=int, default=25)
    parser.add_argument("--evaluation-episodes", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--run-dir", default="runs/vmpo_seed42_1m")
    parser.add_argument("--resume-from")
    args = parser.parse_args()
    return TrainingConfig(
        total_steps=args.steps,
        seed=args.seed,
        n_envs=args.n_envs,
        episode_length=args.episode_length,
        evaluation_episodes=args.evaluation_episodes,
        device=args.device,
        run_dir=args.run_dir,
        resume_from=args.resume_from,
    )


if __name__ == "__main__":
    train(parse_args())
