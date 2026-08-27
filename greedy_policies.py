"""Simple baseline policies for Monster Gridworld."""

from __future__ import annotations

from typing import Callable, Dict

import numpy as np

from monster_gridworld import ACTION_DELTAS, AGENT


def make_greedy_policy(
    target_channel: int,
    *,
    seed: int = 42,
) -> Callable[[Dict[str, object]], int]:
    """Return a policy that moves toward the nearest target entity.

    The policy ignores monsters and breaks equally good moves randomly.
    """
    rng = np.random.default_rng(seed)

    def policy(observation: Dict[str, object]) -> int:
        grid = np.asarray(observation["grid"])
        agent = np.argwhere(grid[..., AGENT] == 1)[0]
        targets = np.argwhere(grid[..., target_channel] == 1)

        if len(targets) == 0:
            return int(rng.choice(list(ACTION_DELTAS)))

        scores = []
        for action, delta in ACTION_DELTAS.items():
            candidate = agent + delta
            if not (
                0 <= candidate[0] < grid.shape[0]
                and 0 <= candidate[1] < grid.shape[1]
            ):
                candidate = agent
            scores.append((action, np.abs(targets - candidate).sum(axis=1).min()))

        best_distance = min(distance for _, distance in scores)
        best_actions = [
            action for action, distance in scores if distance == best_distance
        ]
        return int(rng.choice(best_actions))

    return policy
