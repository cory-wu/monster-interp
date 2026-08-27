import torch

from monster_agent import (
    PAPER_ADAM_BETAS,
    PAPER_LEARNING_RATE,
    MonsterActorCritic,
    make_paper_optimizer,
    observation_to_tensors,
)
from monster_gridworld import MonsterGridworld


def test_single_step_output_and_recurrent_state_shapes():
    model = MonsterActorCritic()
    grid = torch.zeros(2, 4, 14, 14)
    inventory = torch.tensor([0.0, 5.0])
    previous_action = torch.tensor([-1, 2])
    previous_reward = torch.tensor([0.0, -1.0])

    output = model(grid, inventory, previous_action, previous_reward)

    assert output.policy_logits.shape == (2, 4)
    assert output.value.shape == (2,)
    assert output.normalized_value.shape == (2,)
    assert output.state[0].shape == (1, 2, 256)
    assert output.state[1].shape == (1, 2, 256)


def test_sequence_forward_supports_backpropagation():
    model = MonsterActorCritic()
    grid = torch.zeros(3, 2, 4, 14, 14)
    inventory = torch.zeros(3, 2)
    previous_action = torch.full((3, 2), -1)
    previous_reward = torch.zeros(3, 2)

    output = model(grid, inventory, previous_action, previous_reward)
    loss = output.policy_logits.square().mean() + output.value.square().mean()
    loss.backward()

    assert output.policy_logits.shape == (3, 2, 4)
    assert output.value.shape == (3, 2)
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_environment_observation_conversion():
    env = MonsterGridworld()
    observation, _ = env.reset(seed=42)

    grid, inventory = observation_to_tensors(observation)

    assert grid.shape == (1, 4, 14, 14)
    assert inventory.shape == (1,)
    assert grid.dtype == torch.float32
    assert inventory.item() == 0.0


def test_paper_optimizer_settings():
    optimizer = make_paper_optimizer(MonsterActorCritic())

    assert optimizer.defaults["lr"] == PAPER_LEARNING_RATE
    assert optimizer.defaults["betas"] == PAPER_ADAM_BETAS


def test_popart_update_preserves_unnormalized_values():
    torch.manual_seed(42)
    model = MonsterActorCritic()
    features = torch.randn(3, 256)
    _, values_before = model.value_head(features)

    model.value_head.update_statistics(torch.tensor([-100.0, -50.0, 10.0]))
    _, values_after = model.value_head(features)

    assert torch.allclose(values_before, values_after, atol=1e-5)
