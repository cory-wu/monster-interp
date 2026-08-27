import numpy as np

from monster_gridworld import (
    APPLE,
    MONSTER,
    MonsterGridworld,
    MonsterGridworldConfig,
    State,
    DOWN,
    RIGHT,
)


def env_no_extra_moves(**kwargs):
    cfg = MonsterGridworldConfig(monster_double_move_prob=0.0, **kwargs)
    return MonsterGridworld(cfg)


def test_reset_shape_and_counts():
    env = env_no_extra_moves()
    obs, info = env.reset(seed=123)
    assert obs["grid"].shape == (14, 14, 4)
    assert obs["grid"].dtype == np.uint8
    assert int(obs["grid"][..., MONSTER].sum()) == 5
    assert int(obs["grid"][..., APPLE].sum()) == 5
    assert info["shield_inventory"] == 0
    # Non-overlap => at most one entity channel active in each cell.
    assert int(obs["grid"].sum(axis=-1).max()) <= 1


def test_seed_reproducibility():
    env1 = env_no_extra_moves()
    env2 = env_no_extra_moves()
    obs1, _ = env1.reset(seed=7)
    obs2, _ = env2.reset(seed=7)
    assert np.array_equal(obs1["grid"], obs2["grid"])


def test_apple_gives_reward_and_respawns():
    env = env_no_extra_moves()
    env.set_state(
        State(
            agent=(5, 5),
            monsters=set(),
            shields_on_grid=set(),
            apples={(5, 6)},
            shield_inventory=0,
        )
    )
    _, reward, _, _, info = env.step(RIGHT)
    assert reward == 1.0
    assert env.state.agent == (5, 6)
    assert len(env.state.apples) == 1
    assert info["n_apples_on_grid"] == 1


def test_shielded_monster_attack_consumes_shield_and_monster():
    env = env_no_extra_moves(respawn_apples=False, respawn_shields=False)
    env.set_state(
        State(
            agent=(5, 5),
            monsters={(5, 4)},
            shields_on_grid=set(),
            apples=set(),
            shield_inventory=1,
        )
    )
    # Monster moves first into the agent and is destroyed. Agent then moves down.
    _, reward, _, _, info = env.step(DOWN)
    assert reward == 0.0
    assert info["n_monsters"] == 0
    assert info["shield_inventory"] == 0


def test_unshielded_monster_attack_penalizes_but_monster_survives():
    env = env_no_extra_moves(respawn_apples=False, respawn_shields=False)
    env.set_state(
        State(
            agent=(5, 5),
            monsters={(5, 4)},
            shields_on_grid=set(),
            apples=set(),
            shield_inventory=0,
        )
    )
    _, reward, _, _, info = env.step(DOWN)
    assert reward == -1.0
    assert info["n_monsters"] == 1


def test_counterfactual_remove_monsters():
    env = env_no_extra_moves()
    env.reset(seed=0)
    cf = env.remove_all_monsters()
    assert len(cf.monsters) == 0
    # Original state unchanged.
    assert len(env.state.monsters) == 5
