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
    """Integrate a learned velocity field on a fixed time grid."""

    def __init__(self, method: str = "euler", node_steps: int = 100):
        if method not in {"euler", "rk4"}:
            raise ValueError("method must be 'euler' or 'rk4'.")
        if node_steps <= 0:
            raise ValueError("node_steps must be positive.")
        self.method = method
        self.node_steps = int(node_steps)

    def time_grid(self, *, device, dtype) -> Tensor:
        return torch.linspace(0.0, 1.0, self.node_steps + 1, device=device, dtype=dtype)

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
    ):
        if states.dim() != 3:
            raise ValueError("states must have shape (time, batch, dim).")
        if velocities.shape != states.shape:
            raise ValueError("velocities must match states shape.")
        mean = states.detach()
        mean_velocity = velocities.detach()
        if reverse:
            mean = torch.flip(mean, dims=(0,))
            mean_velocity = -torch.flip(mean_velocity, dims=(0,))
        mean, mean_velocity, _ = cls.subsample_for_bridge(mean, mean_velocity, bridge_steps)
        mean_guess = mean.permute(1, 0, 2).contiguous().clone()
        velocity_guess = mean_velocity.permute(1, 0, 2).contiguous().clone()
        x0_flat = x0.detach().reshape(x0.shape[0], -1).to(
            device=mean_guess.device, dtype=mean_guess.dtype
        )
        x1_flat = x1.detach().reshape(x1.shape[0], -1).to(
            device=mean_guess.device, dtype=mean_guess.dtype
        )
        mean_guess[:, 0, :] = x0_flat
        mean_guess[:, -1, :] = x1_flat
        expected_shape = (x0_flat.shape[0], int(bridge_steps) + 1, x0_flat.shape[1])
        if tuple(mean_guess.shape) != expected_shape:
            raise ValueError(f"mean bridge guess must have shape {expected_shape}; got {tuple(mean_guess.shape)}.")
        if tuple(velocity_guess.shape) != expected_shape:
            raise ValueError(
                f"velocity bridge guess must have shape {expected_shape}; got {tuple(velocity_guess.shape)}."
            )
        if not torch.allclose(mean_guess[:, 0, :], x0_flat):
            raise ValueError("mean bridge guess does not start at the BVP source endpoint.")
        if not torch.allclose(mean_guess[:, -1, :], x1_flat):
            raise ValueError("mean bridge guess does not end at the BVP target endpoint.")
        return mean_guess.detach().cpu().numpy(), velocity_guess.detach().cpu().numpy()

    @staticmethod
    def _velocity_at(model: torch.nn.Module, x: Tensor, t_value: Tensor) -> Tensor:
        t = t_value.reshape(1, 1).expand(x.shape[0], 1)
        return call_velocity_model(model, x, t)

    def _rk4_step(self, model: torch.nn.Module, x: Tensor, t: Tensor, dt: Tensor):
        k1 = self._velocity_at(model, x, t)
        k2 = self._velocity_at(model, x + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = self._velocity_at(model, x + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = self._velocity_at(model, x + dt * k3, t + dt)
        v = (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        return x + dt * v, k1
