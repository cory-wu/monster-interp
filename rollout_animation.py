"""Minimal Matplotlib helpers for recording and animating rollouts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Union

import numpy as np

from monster_gridworld import AGENT, APPLE, MONSTER, SHIELD, MonsterGridworld

ACTION_NAMES = {0: "UP", 1: "RIGHT", 2: "DOWN", 3: "LEFT"}

BACKGROUND_COLOR = np.array([24, 27, 34], dtype=np.uint8)
ENTITY_COLORS = np.array(
    [
        [245, 245, 245],  # agent
        [220, 55, 55],  # monster
        [145, 85, 190],  # shield
        [65, 175, 85],  # apple
    ],
    dtype=np.uint8,
)


@dataclass(frozen=True)
class RolloutFrame:
    """Everything needed to render one environment step."""

    grid: np.ndarray
    step: int
    action: Optional[int]
    reward: float
    total_reward: float
    shield_inventory: int


ActionSource = Union[
    Iterable[int],
    Callable[[Dict[str, object]], int],
]


def record_rollout(
    env: MonsterGridworld,
    actions: ActionSource,
    *,
    seed: Optional[int] = None,
    max_steps: Optional[int] = None,
) -> List[RolloutFrame]:
    """Reset ``env`` and record frames produced by actions or a policy.

    A callable action source receives the latest observation and requires
    ``max_steps``. An iterable action source naturally stops when exhausted.
    Environment termination or truncation always stops the rollout.
    """
    is_policy = callable(actions)
    if is_policy and max_steps is None:
        raise ValueError("max_steps is required when actions is a callable policy")

    observation, info = env.reset(seed=seed)
    frames = [_make_frame(observation, info, action=None, reward=0.0, total=0.0)]
    total_reward = 0.0
    action_iterator = None if is_policy else iter(actions)

    while max_steps is None or len(frames) - 1 < max_steps:
        if is_policy:
            action = int(actions(observation))
        else:
            try:
                action = int(next(action_iterator))
            except StopIteration:
                break

        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        frames.append(
            _make_frame(observation, info, action, reward, total_reward)
        )
        if terminated or truncated:
            break

    return frames


def animate_rollout(
    frames: Iterable[RolloutFrame],
    *,
    interval: int = 250,
    repeat: bool = False,
    label: str = "",
):
    """Return a ``matplotlib.animation.FuncAnimation`` for recorded frames."""
    from matplotlib import pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.patches import Patch

    frames = list(frames)
    if not frames:
        raise ValueError("frames must not be empty")

    figure, axis = plt.subplots()
    image = axis.imshow(grid_to_rgb(frames[0].grid), interpolation="nearest")
    title = axis.set_title("")
    axis.set_xticks([])
    axis.set_yticks([])
    axis.legend(
        handles=[
            Patch(color=ENTITY_COLORS[AGENT] / 255, label="Agent"),
            Patch(color=ENTITY_COLORS[MONSTER] / 255, label="Monster"),
            Patch(color=ENTITY_COLORS[SHIELD] / 255, label="Shield"),
            Patch(color=ENTITY_COLORS[APPLE] / 255, label="Apple"),
        ],
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        borderaxespad=0,
    )

    def draw(index: int):
        frame = frames[index]
        action = "RESET" if frame.action is None else ACTION_NAMES.get(
            frame.action, str(frame.action)
        )
        image.set_data(grid_to_rgb(frame.grid))
        heading = f"{label}\n" if label else ""
        title.set_text(
            f"{heading}step={frame.step}  action={action}  reward={frame.reward:+g}\n"
            f"total={frame.total_reward:+g}  shields={frame.shield_inventory}"
        )
        return image, title

    return FuncAnimation(
        figure,
        draw,
        init_func=lambda: draw(0),
        frames=len(frames),
        interval=interval,
        repeat=repeat,
        blit=False,
    )


def grid_to_rgb(grid: np.ndarray) -> np.ndarray:
    """Convert a one-hot environment grid to an RGB image."""
    grid = np.asarray(grid)
    if grid.ndim != 3 or grid.shape[-1] != len(ENTITY_COLORS):
        raise ValueError("grid must have shape [height, width, 4]")

    rgb = np.empty((*grid.shape[:2], 3), dtype=np.uint8)
    rgb[:] = BACKGROUND_COLOR
    occupied = grid.any(axis=-1)
    entities = grid.argmax(axis=-1)
    rgb[occupied] = ENTITY_COLORS[entities[occupied]]
    return rgb


def _make_frame(
    observation: Dict[str, object],
    info: Dict[str, int],
    action: Optional[int],
    reward: float,
    total: float,
) -> RolloutFrame:
    return RolloutFrame(
        grid=np.asarray(observation["grid"]).copy(),
        step=info["step"],
        action=action,
        reward=float(reward),
        total_reward=float(total),
        shield_inventory=info["shield_inventory"],
    )
