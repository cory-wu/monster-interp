import torch

from monster_agent import MonsterActorCritic
from monster_gridworld import MonsterGridworldConfig
from rollout_collector import RolloutCollector


def test_rollout_collector_shapes_and_event_accounting():
    config = MonsterGridworldConfig(episode_length=3)
    collector = RolloutCollector(config, n_envs=2, seed=42)
    model = MonsterActorCritic()

    batch = collector.collect(model, torch.device("cpu"))

    assert batch.grids.shape == (3, 2, 4, 14, 14)
    assert batch.inventories.shape == (3, 2)
    assert batch.actions.shape == (3, 2)
    assert batch.rewards.shape == (3, 2)
    assert batch.n_steps == 6
    assert batch.episode_metrics["return"].shape == (2,)
    assert batch.episode_metrics["boundary_fraction"].shape == (2,)
    action_fraction = sum(
        batch.episode_metrics[name]
        for name in (
            "action_up_fraction",
            "action_right_fraction",
            "action_down_fraction",
            "action_left_fraction",
        )
    )
    assert torch.allclose(action_fraction, torch.ones(2))
