"""Generate greedy-apples and greedy-shields rollout GIFs using seed 42."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt

from greedy_policies import make_greedy_policy
from monster_gridworld import APPLE, SHIELD, MonsterGridworld
from rollout_animation import animate_rollout, record_rollout

SEED = 42
N_STEPS = 25


def save_rollout(target_channel: int, label: str, output: Path) -> None:
    env = MonsterGridworld()
    policy = make_greedy_policy(target_channel, seed=SEED)
    frames = record_rollout(env, policy, seed=SEED, max_steps=N_STEPS)
    animation = animate_rollout(frames, label=label)
    animation.save(output, writer="pillow", fps=4)
    plt.close(animation._fig)
    print(f"Saved {output}")


def main() -> None:
    output_dir = Path(__file__).parent / "rollouts"
    output_dir.mkdir(exist_ok=True)
    save_rollout(APPLE, "Greedy apples", output_dir / "greedy_apples_seed42.gif")
    save_rollout(SHIELD, "Greedy shields", output_dir / "greedy_shields_seed42.gif")


if __name__ == "__main__":
    main()
