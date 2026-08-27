"""Paper-inspired recurrent actor-critic network for Monster Gridworld."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from monster_gridworld import N_CHANNELS

PAPER_LEARNING_RATE = 2e-4
PAPER_ADAM_BETAS = (0.0, 0.999)

RecurrentState = Tuple[Tensor, Tensor]


@dataclass
class ActorCriticOutput:
    policy_logits: Tensor
    value: Tensor
    normalized_value: Tensor
    state: RecurrentState


class PopArtValueHead(nn.Module):
    """Value MLP with scale-invariant PopArt target normalization."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        *,
        statistics_lr: float = 1e-4,
        scale_min: float = 1e-2,
        scale_max: float = 1e6,
    ) -> None:
        super().__init__()
        self.statistics_lr = statistics_lr
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.hidden = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
        )
        self.output = nn.Linear(hidden_size, 1)
        self.register_buffer("mean", torch.tensor(0.0))
        self.register_buffer("second_moment", torch.tensor(1.0))

    @property
    def scale(self) -> Tensor:
        variance = (self.second_moment - self.mean.square()).clamp_min(
            self.scale_min**2
        )
        return variance.sqrt().clamp(self.scale_min, self.scale_max)

    def forward(self, inputs: Tensor) -> Tuple[Tensor, Tensor]:
        normalized = self.output(self.hidden(inputs)).squeeze(-1)
        value = normalized * self.scale + self.mean
        return normalized, value

    def normalize(self, targets: Tensor) -> Tensor:
        return (targets - self.mean) / self.scale

    @torch.no_grad()
    def update_statistics(self, targets: Tensor) -> None:
        old_mean = self.mean.clone()
        old_scale = self.scale.clone()
        rate = self.statistics_lr
        self.mean.lerp_(targets.mean(), rate)
        self.second_moment.lerp_(targets.square().mean(), rate)
        new_scale = self.scale

        # Preserve every unnormalized value prediction after changing moments.
        self.output.weight.mul_(old_scale / new_scale)
        self.output.bias.copy_(
            (old_scale * self.output.bias + old_mean - self.mean) / new_scale
        )


class MonsterActorCritic(nn.Module):
    """Shared convolutional/LSTM trunk with policy and value heads.

    Inputs may describe one step:
        grid: [batch, 4, 14, 14]
        shield_inventory, previous_action, previous_reward: [batch]

    Or an unroll:
        grid: [time, batch, 4, 14, 14]
        remaining inputs: [time, batch]

    Use previous_action=-1 at the first step of an episode. It is encoded as
    an all-zero vector. The inventory is normalized by max_inventory.
    """

    def __init__(
        self,
        *,
        grid_size: int = 14,
        n_actions: int = 4,
        max_inventory: int = 10,
        lstm_size: int = 256,
        head_size: int = 256,
    ) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.n_actions = n_actions
        self.max_inventory = max_inventory
        self.lstm_size = lstm_size

        # The paper specifies two 3x3, stride-1 convolutions with 16 channels.
        # Padding is not specified; padding=1 preserves boundary information.
        self.encoder = nn.Sequential(
            nn.Conv2d(N_CHANNELS, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        encoded_size = 16 * grid_size * grid_size
        recurrent_input_size = encoded_size + 1 + n_actions + 1
        self.lstm = nn.LSTM(recurrent_input_size, lstm_size)

        self.policy_head = nn.Sequential(
            nn.Linear(lstm_size, head_size),
            nn.ReLU(),
            nn.Linear(head_size, n_actions),
        )
        self.value_head = PopArtValueHead(lstm_size, head_size)

    def initial_state(
        self,
        batch_size: int,
        *,
        device: Optional[torch.device] = None,
    ) -> RecurrentState:
        """Return a zeroed ``(hidden, cell)`` LSTM state."""
        if device is None:
            device = next(self.parameters()).device
        shape = (1, batch_size, self.lstm_size)
        return torch.zeros(shape, device=device), torch.zeros(shape, device=device)

    def forward(
        self,
        grid: Tensor,
        shield_inventory: Tensor,
        previous_action: Tensor,
        previous_reward: Tensor,
        state: Optional[RecurrentState] = None,
    ) -> ActorCriticOutput:
        sequence_input = grid.ndim == 5
        if grid.ndim == 4:
            grid = grid.unsqueeze(0)
        elif grid.ndim != 5:
            raise ValueError("grid must have shape [B,C,H,W] or [T,B,C,H,W]")

        time_steps, batch_size, channels, height, width = grid.shape
        if (channels, height, width) != (
            N_CHANNELS,
            self.grid_size,
            self.grid_size,
        ):
            raise ValueError(
                f"grid must end with [{N_CHANNELS},{self.grid_size},{self.grid_size}]"
            )

        device = grid.device
        inventory = self._as_time_batch(
            shield_inventory, time_steps, batch_size, "shield_inventory", device
        ).float()
        actions = self._as_time_batch(
            previous_action, time_steps, batch_size, "previous_action", device
        ).long()
        rewards = self._as_time_batch(
            previous_reward, time_steps, batch_size, "previous_reward", device
        ).float()

        if torch.any((actions < -1) | (actions >= self.n_actions)):
            raise ValueError("previous_action must be -1 or a valid action index")

        encoded = self.encoder(
            grid.reshape(time_steps * batch_size, channels, height, width).float()
        ).reshape(time_steps, batch_size, -1)

        has_previous_action = (actions >= 0).unsqueeze(-1)
        action_one_hot = F.one_hot(actions.clamp_min(0), self.n_actions).float()
        action_one_hot = action_one_hot * has_previous_action

        recurrent_input = torch.cat(
            (
                encoded,
                (inventory / self.max_inventory).unsqueeze(-1),
                action_one_hot,
                rewards.unsqueeze(-1),
            ),
            dim=-1,
        )
        if state is None:
            state = self.initial_state(batch_size, device=device)

        recurrent_output, next_state = self.lstm(recurrent_input, state)
        policy_logits = self.policy_head(recurrent_output)
        normalized_value, value = self.value_head(recurrent_output)

        if not sequence_input:
            policy_logits = policy_logits.squeeze(0)
            value = value.squeeze(0)
            normalized_value = normalized_value.squeeze(0)

        return ActorCriticOutput(policy_logits, value, normalized_value, next_state)

    @staticmethod
    def _as_time_batch(
        tensor: Tensor,
        time_steps: int,
        batch_size: int,
        name: str,
        device: torch.device,
    ) -> Tensor:
        tensor = torch.as_tensor(tensor, device=device)
        if tensor.shape == (batch_size,):
            tensor = tensor.unsqueeze(0)
        if tensor.shape != (time_steps, batch_size):
            raise ValueError(
                f"{name} must have shape [{batch_size}] or "
                f"[{time_steps},{batch_size}]"
            )
        return tensor


def observation_to_tensors(
    observation: Dict[str, object],
    *,
    device: Optional[torch.device] = None,
) -> Tuple[Tensor, Tensor]:
    """Convert one environment observation to batched network inputs."""
    grid = torch.as_tensor(observation["grid"], dtype=torch.float32, device=device)
    grid = grid.permute(2, 0, 1).unsqueeze(0)
    inventory = torch.as_tensor(
        [observation["shield_inventory"]], dtype=torch.float32, device=device
    )
    return grid, inventory


def make_paper_optimizer(model: nn.Module) -> torch.optim.Adam:
    """Construct Adam with the Monster Gridworld paper's hyperparameters."""
    return torch.optim.Adam(
        model.parameters(),
        lr=PAPER_LEARNING_RATE,
        betas=PAPER_ADAM_BETAS,
    )
