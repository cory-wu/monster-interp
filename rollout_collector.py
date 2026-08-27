"""Synchronous, batched rollout collection for Monster Gridworld."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict

import numpy as np
import torch
from torch import Tensor
from torch.distributions import Categorical

from monster_agent import MonsterActorCritic
from monster_gridworld import (
    AGENT,
    STEP_EVENT_NAMES,
    MonsterGridworld,
    MonsterGridworldConfig,
)

ACTION_METRIC_NAMES = {
    0: "action_up_fraction",
    1: "action_right_fraction",
    2: "action_down_fraction",
    3: "action_left_fraction",
}


@dataclass
class RolloutBatch:
    grids: Tensor
    inventories: Tensor
    previous_actions: Tensor
    previous_rewards: Tensor
    actions: Tensor
    rewards: Tensor
    episode_metrics: Dict[str, Tensor]

    @property
    def n_steps(self) -> int:
        return self.actions.numel()

    def to(self, device: torch.device) -> "RolloutBatch":
        return RolloutBatch(
            grids=self.grids.to(device),
            inventories=self.inventories.to(device),
            previous_actions=self.previous_actions.to(device),
            previous_rewards=self.previous_rewards.to(device),
            actions=self.actions.to(device),
            rewards=self.rewards.to(device),
            episode_metrics=self.episode_metrics,
        )


class RolloutCollector:
    """Collect full, equally sized episodes from independent environments."""

    def __init__(
        self,
        config: MonsterGridworldConfig,
        *,
        n_envs: int,
        seed: int,
        deterministic: bool = False,
    ) -> None:
        if n_envs < 1:
            raise ValueError("n_envs must be positive")
        self.config = replace(config)
        self.n_envs = n_envs
        self.seed = seed
        self.deterministic = deterministic
        self.envs = [MonsterGridworld(replace(config)) for _ in range(n_envs)]
        self.batch_index = 0

    @torch.no_grad()
    def collect(
        self,
        policy: MonsterActorCritic,
        device: torch.device,
    ) -> RolloutBatch:
        observations = []
        for index, env in enumerate(self.envs):
            reset_seed = self.seed + index if self.batch_index == 0 else None
            observation, _ = env.reset(seed=reset_seed)
            observations.append(observation)
        self.batch_index += 1

        grids = []
        inventories = []
        previous_actions_history = []
        previous_rewards_history = []
        actions_history = []
        rewards_history = []

        previous_actions = torch.full(
            (self.n_envs,), -1, dtype=torch.long, device=device
        )
        previous_rewards = torch.zeros(self.n_envs, device=device)
        state = policy.initial_state(self.n_envs, device=device)

        metrics = {
            "return": torch.zeros(self.n_envs),
            "final_inventory": torch.zeros(self.n_envs),
            "boundary_fraction": torch.zeros(self.n_envs),
            "corner_fraction": torch.zeros(self.n_envs),
            **{name: torch.zeros(self.n_envs) for name in STEP_EVENT_NAMES},
            **{name: torch.zeros(self.n_envs) for name in ACTION_METRIC_NAMES.values()},
        }

        for _ in range(self.config.episode_length):
            grid = torch.from_numpy(
                np.stack([observation["grid"] for observation in observations])
            ).permute(0, 3, 1, 2)
            inventory = torch.tensor(
                [observation["shield_inventory"] for observation in observations],
                dtype=torch.float32,
            )
            for index, observation in enumerate(observations):
                row, column = np.argwhere(observation["grid"][..., AGENT] == 1)[0]
                on_row_edge = row in (0, self.config.size - 1)
                on_column_edge = column in (0, self.config.size - 1)
                metrics["boundary_fraction"][index] += (
                    on_row_edge or on_column_edge
                ) / self.config.episode_length
                metrics["corner_fraction"][index] += (
                    on_row_edge and on_column_edge
                ) / self.config.episode_length

            grids.append(grid)
            inventories.append(inventory)
            previous_actions_history.append(previous_actions.cpu())
            previous_rewards_history.append(previous_rewards.cpu())

            output = policy(
                grid.to(device),
                inventory.to(device),
                previous_actions,
                previous_rewards,
                state,
            )
            if self.deterministic:
                actions = output.policy_logits.argmax(dim=-1)
            else:
                actions = Categorical(logits=output.policy_logits).sample()
            for index, action in enumerate(actions.tolist()):
                metrics[ACTION_METRIC_NAMES[action]][index] += (
                    1 / self.config.episode_length
                )
            state = output.state

            next_observations = []
            step_rewards = []
            for index, (env, action) in enumerate(zip(self.envs, actions.tolist())):
                observation, reward, _, _, info = env.step(action)
                next_observations.append(observation)
                step_rewards.append(reward)
                metrics["return"][index] += reward
                metrics["final_inventory"][index] = info["shield_inventory"]
                for name in STEP_EVENT_NAMES:
                    metrics[name][index] += info[name]

            rewards = torch.tensor(step_rewards, dtype=torch.float32, device=device)
            actions_history.append(actions.cpu())
            rewards_history.append(rewards.cpu())
            observations = next_observations
            previous_actions = actions
            previous_rewards = rewards

        return RolloutBatch(
            grids=torch.stack(grids),
            inventories=torch.stack(inventories),
            previous_actions=torch.stack(previous_actions_history),
            previous_rewards=torch.stack(previous_rewards_history),
            actions=torch.stack(actions_history),
            rewards=torch.stack(rewards_history),
            episode_metrics=metrics,
        )
