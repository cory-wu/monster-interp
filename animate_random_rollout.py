"""Show a random Monster Gridworld rollout as a Matplotlib animation."""

import numpy as np
from matplotlib import pyplot as plt

from monster_gridworld import MonsterGridworld
from rollout_animation import animate_rollout, record_rollout


def main() -> None:
    env = MonsterGridworld()
    rng = np.random.default_rng(42)
    frames = record_rollout(
        env,
        lambda observation: int(rng.integers(4)),
        seed=42,
        max_steps=25,
    )

    animation = animate_rollout(frames)
    plt.show()


if __name__ == "__main__":
    main()
