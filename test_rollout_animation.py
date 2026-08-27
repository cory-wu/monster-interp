import matplotlib
import numpy as np

matplotlib.use("Agg")

from matplotlib import pyplot as plt

from greedy_policies import make_greedy_policy
from monster_gridworld import (
    AGENT,
    APPLE,
    DOWN,
    RIGHT,
    SHIELD,
    MonsterGridworld,
    MonsterGridworldConfig,
)
from rollout_animation import (
    BACKGROUND_COLOR,
    ENTITY_COLORS,
    animate_rollout,
    grid_to_rgb,
    record_rollout,
)


def test_record_rollout_includes_initial_frame_and_stops_at_time_limit():
    env = MonsterGridworld(
        MonsterGridworldConfig(
            n_monsters=0,
            n_shields=0,
            n_apples=0,
            episode_length=2,
        )
    )

    frames = record_rollout(env, [RIGHT, RIGHT, RIGHT], seed=0)

    assert [frame.step for frame in frames] == [0, 1, 2]
    assert frames[0].action is None
    assert [frame.reward for frame in frames] == [0.0, 0.0, 0.0]


def test_record_rollout_accepts_a_policy():
    env = MonsterGridworld(
        MonsterGridworldConfig(n_monsters=0, n_shields=0, n_apples=0)
    )

    frames = record_rollout(env, lambda observation: RIGHT, seed=0, max_steps=3)

    assert len(frames) == 4
    assert [frame.action for frame in frames[1:]] == [RIGHT, RIGHT, RIGHT]


def test_grid_to_rgb_uses_background_and_entity_colors():
    grid = np.zeros((2, 2, 4), dtype=np.uint8)
    grid[0, 1, AGENT] = 1

    rgb = grid_to_rgb(grid)

    assert np.array_equal(rgb[0, 0], BACKGROUND_COLOR)
    assert np.array_equal(rgb[0, 1], ENTITY_COLORS[AGENT])


def test_animate_rollout_returns_a_func_animation():
    from matplotlib.animation import FuncAnimation

    env = MonsterGridworld(
        MonsterGridworldConfig(n_monsters=0, n_shields=0, n_apples=0)
    )
    frames = record_rollout(env, [RIGHT], seed=0)

    animation = animate_rollout(frames, interval=100)
    animation._init_draw()

    assert isinstance(animation, FuncAnimation)
    plt.close(animation._fig)


def test_greedy_policy_moves_toward_requested_entity():
    grid = np.zeros((5, 5, 4), dtype=np.uint8)
    grid[2, 2, AGENT] = 1
    grid[2, 4, APPLE] = 1
    grid[4, 2, SHIELD] = 1
    observation = {"grid": grid, "shield_inventory": 0}

    apple_policy = make_greedy_policy(APPLE, seed=42)
    shield_policy = make_greedy_policy(SHIELD, seed=42)

    assert apple_policy(observation) == RIGHT
    assert shield_policy(observation) == DOWN
