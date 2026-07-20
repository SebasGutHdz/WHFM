"""Shared schedules for Hamiltonian potentials."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import Tensor


@dataclass(frozen=True)
class SigmaSchedule:
    """Tabulated sigma and sigma-prime values on a monotone time grid."""

    t: Tensor
    sigma: Tensor
    sigma_prime: Tensor

    def to(self, *, device=None, dtype=None) -> "SigmaSchedule":
        return SigmaSchedule(
            t=self.t.to(device=device, dtype=dtype),
            sigma=self.sigma.to(device=device, dtype=dtype),
            sigma_prime=self.sigma_prime.to(device=device, dtype=dtype),
        )

    def evaluate(self, t: Tensor) -> Tuple[Tensor, Tensor]:
        """Linearly interpolate ``sigma`` and ``sigma_prime`` at ``t``."""

        if not torch.is_tensor(t):
            t = torch.as_tensor(t, dtype=self.t.dtype, device=self.t.device)

        grid = self.t.to(device=t.device, dtype=t.dtype).reshape(-1)
        sigma = self.sigma.to(device=t.device, dtype=t.dtype).reshape(grid.numel(), -1)
        sigma_prime = self.sigma_prime.to(device=t.device, dtype=t.dtype).reshape(
            grid.numel(), -1
        )

        t_shape = t.shape if t.dim() > 0 else torch.Size([1])
        t_flat = t.reshape(-1).clamp(grid[0], grid[-1])
        right = torch.searchsorted(grid, t_flat, right=False).clamp(1, grid.numel() - 1)
        left = right - 1

        t_left = grid[left]
        t_right = grid[right]
        weight = ((t_flat - t_left) / (t_right - t_left).clamp_min(torch.finfo(t.dtype).eps))
        weight = weight.unsqueeze(-1)

        sigma_t = sigma[left] + weight * (sigma[right] - sigma[left])
        sigma_prime_t = sigma_prime[left] + weight * (
            sigma_prime[right] - sigma_prime[left]
        )

        if sigma_t.shape[-1] == 1:
            return sigma_t.reshape(t_shape), sigma_prime_t.reshape(t_shape)
        return (
            sigma_t.reshape(*t_shape, sigma_t.shape[-1]),
            sigma_prime_t.reshape(*t_shape, sigma_prime_t.shape[-1]),
        )
