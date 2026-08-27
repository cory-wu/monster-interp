import numpy as np

from monster_gridworld import (
    APPLE,
    MONSTER,
    DOWN,
    RIGHT,
    UP,
    MonsterGridworld,
    MonsterGridworldConfig,
    State,
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
    assert info["apples_collected"] == 1


def test_shield_pickup_has_zero_reward_and_increments_inventory():
    env = env_no_extra_moves(respawn_apples=False, respawn_shields=False)
    env.set_state(
        State(
            agent=(5, 5),
            monsters=set(),
            shields_on_grid={(5, 6)},
            apples=set(),
            shield_inventory=2,
        )
    )

    _, reward, _, _, info = env.step(RIGHT)

    assert reward == 0.0
    assert env.state.agent == (5, 6)
    assert info["shield_inventory"] == 3
    assert info["n_shields_on_grid"] == 0
    assert info["shields_collected"] == 1


def test_shield_inventory_is_capped_at_ten():
    env = env_no_extra_moves(respawn_apples=False, respawn_shields=False)
    env.set_state(
        State(
            agent=(5, 5),
            monsters=set(),
            shields_on_grid={(5, 6)},
            apples=set(),
            shield_inventory=10,
        )
    )

    _, reward, _, _, info = env.step(RIGHT)

    assert reward == 0.0
    assert info["shield_inventory"] == 10
    assert info["n_shields_on_grid"] == 0


def test_ordinary_and_blocked_movement_have_zero_reward():
    env = env_no_extra_moves(respawn_apples=False, respawn_shields=False)
    env.set_state(
        State(
            agent=(0, 0),
            monsters=set(),
            shields_on_grid=set(),
            apples=set(),
            shield_inventory=0,
        )
    )

    _, blocked_reward, _, _, _ = env.step(UP)
    _, movement_reward, _, _, _ = env.step(RIGHT)

    assert blocked_reward == 0.0
    assert movement_reward == 0.0


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
    assert info["monsters_destroyed"] == 1


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
    assert info["unshielded_attacks"] == 1


def test_each_unshielded_attack_is_penalized():
    env = env_no_extra_moves(respawn_apples=False, respawn_shields=False)
    env.set_state(
        State(
            agent=(5, 5),
            monsters={(5, 4), (5, 6)},
            shields_on_grid=set(),
            apples=set(),
            shield_inventory=0,
        )
    )

    _, reward, _, _, info = env.step(DOWN)

    assert reward == -2.0
    assert info["n_monsters"] == 2


def test_same_monster_attacking_twice_costs_two_reward():
    env = MonsterGridworld(
        MonsterGridworldConfig(
            monster_double_move_prob=1.0,
            respawn_apples=False,
            respawn_shields=False,
        )
    )
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

    assert reward == -2.0
    assert info["n_monsters"] == 1


def test_one_shield_only_blocks_one_of_two_attacks():
    env = env_no_extra_moves(respawn_apples=False, respawn_shields=False)
    env.set_state(
        State(
            agent=(5, 5),
            monsters={(5, 4), (5, 6)},
            shields_on_grid=set(),
            apples=set(),
            shield_inventory=1,
        )
    )

    _, reward, _, _, info = env.step(DOWN)

    assert reward == -1.0
    assert info["n_monsters"] == 1
    assert info["shield_inventory"] == 0


def test_attack_and_apple_rewards_add_within_a_step():
    env = env_no_extra_moves(respawn_apples=False, respawn_shields=False)
    env.set_state(
        State(
            agent=(5, 5),
            monsters={(5, 4)},
            shields_on_grid=set(),
            apples={(6, 5)},
            shield_inventory=0,
        )
    )

    _, reward, _, _, info = env.step(DOWN)

    assert reward == 0.0  # -1 for the attack, then +1 for the apple.
    assert info["n_apples_on_grid"] == 0


def test_counterfactual_remove_monsters():
    env = env_no_extra_moves()
    env.reset(seed=0)
    cf = env.remove_all_monsters()
    assert len(cf.monsters) == 0
    # Original state unchanged.
    assert len(env.state.monsters) == 5
