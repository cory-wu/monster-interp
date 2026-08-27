"""A compact discrete-action V-MPO objective."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class VMPOLossOutput:
    total: Tensor
    policy: Tensor
    value: Tensor
    temperature: Tensor
    trust_region: Tensor
    kl: Tensor
    entropy: Tensor
    eta: Tensor
    alpha: Tensor


class DiscreteVMPOLoss(nn.Module):
    """V-MPO policy improvement plus a squared value loss."""

    def __init__(
        self,
        *,
        epsilon_eta: float = 0.1,
        epsilon_alpha: float = 0.01,
        initial_eta: float = 1.0,
        initial_alpha: float = 5.0,
    ) -> None:
        super().__init__()
        self.epsilon_eta = epsilon_eta
        self.epsilon_alpha = epsilon_alpha
        self.raw_eta = nn.Parameter(_inverse_softplus(initial_eta))
        self.raw_alpha = nn.Parameter(_inverse_softplus(initial_alpha))

    @property
    def eta(self) -> Tensor:
        return F.softplus(self.raw_eta) + 1e-8

    @property
    def alpha(self) -> Tensor:
        return F.softplus(self.raw_alpha) + 1e-8

    def forward(
        self,
        policy_logits: Tensor,
        values: Tensor,
        target_policy_logits: Tensor,
        actions: Tensor,
        returns: Tensor,
        normalized_values: Tensor | None = None,
        normalized_returns: Tensor | None = None,
    ) -> VMPOLossOutput:
        if policy_logits.shape[:-1] != values.shape:
            raise ValueError("policy and value leading dimensions must match")
        if target_policy_logits.shape != policy_logits.shape:
            raise ValueError("target_policy_logits must match policy_logits")
        if actions.shape != values.shape or returns.shape != values.shape:
            raise ValueError("actions and returns must match value shape")

        flat_logits = policy_logits.reshape(-1, policy_logits.shape[-1])
        flat_target_logits = target_policy_logits.reshape_as(flat_logits).detach()
        flat_actions = actions.reshape(-1)
        flat_values = values.reshape(-1)
        flat_returns = returns.reshape(-1).detach()
        advantages = (flat_returns - flat_values.detach())

        top_count = max(1, advantages.numel() // 2)
        top_advantages, top_indices = torch.topk(advantages, top_count, sorted=False)

        log_probs = F.log_softmax(flat_logits, dim=-1)
        selected_log_probs = log_probs.gather(1, flat_actions[:, None]).squeeze(1)
        selected_top_log_probs = selected_log_probs[top_indices]

        scaled_advantages = top_advantages / self.eta.detach()
        weights = F.softmax(scaled_advantages, dim=0).detach()
        policy_loss = -(weights * selected_top_log_probs).sum()

        log_mean_exp = torch.logsumexp(top_advantages / self.eta, dim=0)
        log_mean_exp = log_mean_exp - math.log(top_count)
        temperature_loss = self.eta * (self.epsilon_eta + log_mean_exp)

        target_log_probs = F.log_softmax(flat_target_logits, dim=-1)
        target_probs = target_log_probs.exp()
        kl = (target_probs * (target_log_probs - log_probs)).sum(dim=-1).mean()
        trust_region_loss = (
            self.alpha * (self.epsilon_alpha - kl.detach())
            + self.alpha.detach() * kl
        )

        if normalized_values is None:
            value_predictions = flat_values
            value_targets = flat_returns
        else:
            if normalized_returns is None:
                raise ValueError("normalized_returns is required with normalized_values")
            value_predictions = normalized_values.reshape(-1)
            value_targets = normalized_returns.reshape(-1).detach()
        value_loss = 0.5 * F.mse_loss(value_predictions, value_targets)
        entropy = -(log_probs.exp() * log_probs).sum(dim=-1).mean()
        total = policy_loss + value_loss + temperature_loss + trust_region_loss

        return VMPOLossOutput(
            total=total,
            policy=policy_loss,
            value=value_loss,
            temperature=temperature_loss,
            trust_region=trust_region_loss,
            kl=kl,
            entropy=entropy,
            eta=self.eta,
            alpha=self.alpha,
        )


def discounted_returns(rewards: Tensor, gamma: float) -> Tensor:
    """Return finite-horizon discounted returns for ``[time, batch]`` rewards."""
    if rewards.ndim != 2:
        raise ValueError("rewards must have shape [time, batch]")
    returns = torch.empty_like(rewards)
    running = torch.zeros_like(rewards[-1])
    for index in range(len(rewards) - 1, -1, -1):
        running = rewards[index] + gamma * running
        returns[index] = running
    return returns


def _inverse_softplus(value: float) -> Tensor:
    if value <= 0:
        raise ValueError("initial Lagrange multipliers must be positive")
    return torch.tensor(math.log(math.expm1(value)), dtype=torch.float32)
