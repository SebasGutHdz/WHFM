"""NODE integration helpers for learned HFM velocity fields."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class NodeTrajectory:
    time_grid: Tensor
    states: Tensor
    velocities: Tensor


def call_velocity_model(model: torch.nn.Module, x: Tensor, t: Tensor) -> Tensor:
    t_col = t.reshape(t.shape[0], -1)[:, :1] if t.dim() > 1 else t.reshape(-1, 1)
    return model(torch.cat([x, t_col.to(device=x.device, dtype=x.dtype)], dim=-1))


class NodeSolver:
    """Integrate a learned drift on a fixed grid."""

    def __init__(
        self,
        method: str = "euler",
        node_steps: int = 100,
        time_end: float = 1.0,
    ):
        if method not in {"euler", "rk4"}:
            raise ValueError("method must be 'euler' or 'rk4'.")
        if node_steps <= 0:
            raise ValueError("node_steps must be positive.")
        if not 0.0 < time_end <= 1.0:
            raise ValueError("time_end must be in (0, 1].")
        self.method = method
        self.node_steps = int(node_steps)
        self.time_end = float(time_end)

    def time_grid(self, *, device, dtype) -> Tensor:
        return torch.linspace(
            0.0, self.time_end, self.node_steps + 1, device=device, dtype=dtype
        )

    @torch.no_grad()
    def integrate(self, model: torch.nn.Module, x0: Tensor) -> NodeTrajectory:
        was_training = model.training
        model.eval()
        t_grid = self.time_grid(device=x0.device, dtype=x0.dtype)
        states = [x0.detach()]
        velocities = []
        x = x0.detach()
        for i in range(t_grid.numel() - 1):
            t = t_grid[i]
            dt = t_grid[i + 1] - t
            if self.method == "euler":
                v = self._velocity_at(model, x, t)
                velocities.append(v)
                x = x + dt * v
            else:
                x, v = self._rk4_step(model, x, t, dt)
                velocities.append(v)
            states.append(x.detach())
        velocities.append(self._velocity_at(model, states[-1], t_grid[-1]))
        if was_training:
            model.train()
        return NodeTrajectory(
            time_grid=t_grid,
            states=torch.stack(states, dim=0),
            velocities=torch.stack(velocities, dim=0),
        )

    def evaluate_velocities(self, model: torch.nn.Module, states: Tensor, time_grid: Tensor) -> Tensor:
        was_training = model.training
        model.eval()
        velocities = []
        with torch.no_grad():
            for i, t_value in enumerate(time_grid.reshape(-1)):
                velocities.append(self._velocity_at(model, states[i], t_value))
        if was_training:
            model.train()
        return torch.stack(velocities, dim=0)

    @staticmethod
    def bridge_indices(num_node_states: int, bridge_steps: int, *, device=None) -> Tensor:
        if bridge_steps <= 0:
            raise ValueError("bridge_steps must be positive.")
        if bridge_steps > num_node_states - 1:
            raise ValueError("bridge_steps must be <= node_steps.")
        return torch.linspace(
            0,
            num_node_states - 1,
            bridge_steps + 1,
            device=device,
            dtype=torch.float64,
        ).round().long()

    @classmethod
    def subsample_for_bridge(cls, states: Tensor, velocities: Tensor, bridge_steps: int):
        if states.shape != velocities.shape:
            raise ValueError("states and velocities must have matching shapes.")
        indices = cls.bridge_indices(states.shape[0], bridge_steps, device=states.device)
        return states.index_select(0, indices), velocities.index_select(0, indices), indices

    @staticmethod
    def _interpolate_time_grid(
        values: Tensor, source_grid: Tensor, target_grid: Tensor
    ) -> Tensor:
        right = torch.searchsorted(source_grid, target_grid, right=False).clamp(
            1, source_grid.numel() - 1
        )
        left = right - 1
        weight = (target_grid - source_grid[left]) / (
            source_grid[right] - source_grid[left]
        ).clamp_min(torch.finfo(source_grid.dtype).eps)
        return values[left] + weight.reshape(-1, 1, 1) * (values[right] - values[left])

    @classmethod
    def prepare_bridge_guess(
        cls,
        states: Tensor,
        velocities: Tensor,
        x0: Tensor,
        x1: Tensor,
        *,
        reverse: bool,
        bridge_steps: int,
        correct_terminal_guess: bool = True,
        time_grid: Tensor | None = None,
    ):
        if states.dim() != 3:
            raise ValueError("states must have shape (time, batch, dim).")
        if velocities.shape != states.shape:
            raise ValueError("velocities must match states shape.")
        x0_flat = x0.detach().reshape(x0.shape[0], -1).to(
            device=states.device, dtype=states.dtype
        )
        x1_flat = x1.detach().reshape(x1.shape[0], -1).to(
            device=states.device, dtype=states.dtype
        )
        mean = states.detach()
        mean_velocity = velocities.detach()
        source_grid = None
        if time_grid is not None:
            source_grid = time_grid.detach().reshape(-1).to(
                device=states.device, dtype=states.dtype
            )
            if source_grid.numel() != states.shape[0]:
                raise ValueError("time_grid must have one value per trajectory state.")
            if not torch.all(source_grid[1:] > source_grid[:-1]):
                raise ValueError("time_grid must be strictly increasing.")
            if not torch.isclose(source_grid[0], torch.zeros_like(source_grid[0])):
                raise ValueError("time_grid must start at 0.")
            if source_grid[-1] <= 0.0 or source_grid[-1] > 1.0:
                raise ValueError("time_grid must end in (0, 1].")
        if reverse:
            mean = torch.flip(mean, dims=(0,))
            mean_velocity = -torch.flip(mean_velocity, dims=(0,))
            if source_grid is not None:
                source_grid = source_grid[-1] - torch.flip(source_grid, dims=(0,))

        if source_grid is None or source_grid[-1] == 1.0:
            mean, mean_velocity, _ = cls.subsample_for_bridge(
                mean, mean_velocity, bridge_steps
            )
        else:
            target_grid = torch.linspace(
                0.0, 1.0, bridge_steps + 1, device=states.device, dtype=states.dtype
            )
            time_end = source_grid[-1]
            interpolation_grid = target_grid.clamp_max(time_end)
            mean = cls._interpolate_time_grid(mean, source_grid, interpolation_grid)
            mean_velocity = cls._interpolate_time_grid(
                mean_velocity, source_grid, interpolation_grid
            )
            if correct_terminal_guess:
                tail_mask = target_grid > time_end
                tail_duration = 1.0 - time_end
                alpha = ((target_grid - time_end) / tail_duration).clamp(0.0, 1.0)
                last_mean = mean[-1]
                tail_mean = last_mean.unsqueeze(0) + alpha.reshape(-1, 1, 1) * (
                    x1_flat.unsqueeze(0) - last_mean.unsqueeze(0)
                )
                tail_velocity = (x1_flat - last_mean) / tail_duration
                mean = torch.where(tail_mask.reshape(-1, 1, 1), tail_mean, mean)
                mean_velocity = torch.where(
                    tail_mask.reshape(-1, 1, 1),
                    tail_velocity.unsqueeze(0).expand_as(mean_velocity),
                    mean_velocity,
                )

        mean_guess = mean.permute(1, 0, 2).contiguous().clone()
        velocity_guess = mean_velocity.permute(1, 0, 2).contiguous().clone()
        mean_guess[:, 0, :] = x0_flat
        if correct_terminal_guess:
            mean_guess[:, -1, :] = x1_flat
        expected_shape = (x0_flat.shape[0], int(bridge_steps) + 1, x0_flat.shape[1])
        if tuple(mean_guess.shape) != expected_shape:
            raise ValueError(
                f"mean bridge guess must have shape {expected_shape}; got {tuple(mean_guess.shape)}."
            )
        if tuple(velocity_guess.shape) != expected_shape:
            raise ValueError(
                f"velocity bridge guess must have shape {expected_shape}; got {tuple(velocity_guess.shape)}."
            )
        if not torch.allclose(mean_guess[:, 0, :], x0_flat):
            raise ValueError("mean bridge guess does not start at the BVP source endpoint.")
        if correct_terminal_guess and not torch.allclose(mean_guess[:, -1, :], x1_flat):
            raise ValueError("mean bridge guess does not end at the BVP target endpoint.")
        return mean_guess.detach().cpu().numpy(), velocity_guess.detach().cpu().numpy()

    def _velocity_at(self, model: torch.nn.Module, x: Tensor, t_value: Tensor) -> Tensor:
        t = t_value.reshape(1, 1).expand(x.shape[0], 1)
        return call_velocity_model(model, x, t)

    def _rk4_step(self, model: torch.nn.Module, x: Tensor, t: Tensor, dt: Tensor):
        k1 = self._velocity_at(model, x, t)
        k2 = self._velocity_at(model, x + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = self._velocity_at(model, x + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = self._velocity_at(model, x + dt * k3, t + dt)
        v = (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        return x + dt * v, k1
