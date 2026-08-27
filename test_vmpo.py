import torch

from vmpo import DiscreteVMPOLoss, discounted_returns


def test_discounted_returns():
    rewards = torch.tensor([[1.0], [2.0], [3.0]])

    returns = discounted_returns(rewards, gamma=0.5)

    assert torch.allclose(returns[:, 0], torch.tensor([2.75, 3.5, 3.0]))


def test_vmpo_loss_is_finite_and_differentiable():
    torch.manual_seed(42)
    objective = DiscreteVMPOLoss()
    logits = torch.randn(3, 2, 4, requires_grad=True)
    values = torch.randn(3, 2, requires_grad=True)
    target_logits = torch.randn(3, 2, 4)
    actions = torch.randint(0, 4, (3, 2))
    returns = torch.randn(3, 2)

    losses = objective(logits, values, target_logits, actions, returns)
    losses.total.backward()

    assert torch.isfinite(losses.total)
    assert logits.grad is not None
    assert values.grad is not None
    assert objective.raw_eta.grad is not None
    assert objective.raw_alpha.grad is not None
