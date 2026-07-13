"""Scalar Gaussian bridge solving for HFM target generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import Tensor

from .gaussian_paths import MeanStdBVPGaussianPath


@dataclass
class BridgeSolution:
    """Validated scalar Gaussian bridge batch.

    Shapes use kept/successful pairs: ``mean`` and ``mean_velocity`` have shape
    ``(batch, time, dim)``; ``std`` and ``std_velocity`` have shape
    ``(batch, time, 1)``.
    """

    time_grid: Tensor
    mean: Tensor
    mean_velocity: Tensor
    std: Tensor
    std_velocity: Tensor
    x0: Tensor
    x1: Tensor
    path: MeanStdBVPGaussianPath
    success_mask: Tensor
    success_indices: Tensor
    failed_indices: Tensor
    endpoint_errors: Tensor
    residual_norms: Tensor
    failure_messages: Dict[int, str]
    solve_time_seconds: float
    pair_solve_time_seconds: list[float]
    solver_iterations: list[Optional[int]]
    solver_mesh_nodes: list[Optional[int]]
    guess_source: str

    @property
    def num_pairs(self) -> int:
        return int(self.success_mask.numel())

    @property
    def num_successful(self) -> int:
        return int(self.x0.shape[0])


class GaussianBridgeSolver:
    """SciPy-backed scalar Gaussian bridge solver."""

    def __init__(
        self,
        potential,
        *,
        sigma: float,
        bridge_steps: int,
        tol: float,
        max_nodes: int,
        quadrature_order: int,
        use_monte_carlo: bool = False,
        monte_carlo_samples: int = 100,
        failure_policy: str = "skip_pair",
    ):
        if failure_policy not in {"skip_pair", "raise"}:
            raise ValueError("failure_policy must be 'skip_pair' or 'raise'.")
        self.potential = potential
        self.sigma = float(sigma)
        self.bridge_steps = int(bridge_steps)
        self.tol = float(tol)
        self.max_nodes = int(max_nodes)
        self.quadrature_order = int(quadrature_order)
        self.use_monte_carlo = bool(use_monte_carlo)
        self.monte_carlo_samples = int(monte_carlo_samples)
        self.failure_policy = failure_policy

    def solve_batch(
        self,
        x0: Tensor,
        x1: Tensor,
        *,
        mean_guess=None,
        mean_velocity_guess=None,
        std_guess=None,
        std_velocity_guess=None,
    ) -> BridgeSolution:
        x0_detached = x0.detach()
        x1_detached = x1.detach()
        if mean_guess is not None and mean_velocity_guess is not None:
            guess_source = "learned_trajectory"
        elif mean_guess is None and mean_velocity_guess is None:
            guess_source = "straight_line"
        else:
            guess_source = "partial_learned_trajectory"

        path = MeanStdBVPGaussianPath(
            self.potential,
            sigma=self.sigma,
            n_steps=self.bridge_steps,
            tol=self.tol,
            max_nodes=self.max_nodes,
            quadrature_order=self.quadrature_order,
            use_monte_carlo=self.use_monte_carlo,
            monte_carlo_samples=self.monte_carlo_samples,
            mu_guess=mean_guess,
            mu_dot_guess=mean_velocity_guess,
            sigma_guess=std_guess,
            sigma_dot_guess=std_velocity_guess,
        )

        try:
            states = path.batch_solve(x0_detached, x1_detached)
        except RuntimeError:
            if self.failure_policy == "raise":
                raise
            failure_messages = dict(path.failure_messages)
            path._record_solve_metadata(x0_detached.shape[0], [], {})
            path.failure_messages = failure_messages
            states = torch.empty(
                (0, self.bridge_steps + 1, 2 * x0_detached.reshape(x0_detached.shape[0], -1).shape[1] + 2),
                dtype=x0_detached.detach().cpu().dtype,
            )
            path._store_cache(x0_detached[:0], x1_detached[:0], states)

        if self.failure_policy == "raise" and path.failed_indices is not None:
            if int(path.failed_indices.numel()) > 0:
                preview = list(path.failure_messages.items())[:5]
                raise RuntimeError(f"Bridge solve failed for {path.failed_indices.numel()} pairs: {preview}")

        success_mask = path.success_mask
        if success_mask is None:
            raise RuntimeError("Bridge path did not report a success mask.")
        success_mask_device = success_mask.to(device=x0_detached.device)
        x0_keep = x0_detached[success_mask_device]
        x1_keep = x1_detached[success_mask_device]

        states_device = states.to(device=x0_detached.device, dtype=x0_detached.dtype)
        dim = x0_detached.reshape(x0_detached.shape[0], -1).shape[1]
        time_grid = path.t_grid.to(device=x0_detached.device, dtype=x0_detached.dtype)
        mean = states_device[:, :, :dim]
        mean_velocity = states_device[:, :, dim : 2 * dim]
        std = states_device[:, :, 2 * dim : 2 * dim + 1]
        std_velocity = states_device[:, :, 2 * dim + 1 : 2 * dim + 2]
        endpoint_errors = self._endpoint_errors(mean, std, x0_keep, x1_keep)
        residual_norms = torch.full(
            (x0_keep.shape[0],), float("nan"), device=x0_detached.device, dtype=x0_detached.dtype
        )
        solve_metadata = getattr(path, "solve_metadata", {}) or {}
        pair_solve_times = list(solve_metadata.get("pair_solve_time_seconds", []))
        solver_iterations = list(solve_metadata.get("pair_solver_iterations", []))
        solver_mesh_nodes = list(solve_metadata.get("pair_solver_mesh_nodes", []))
        solve_time_seconds = float(solve_metadata.get("total_solve_time_seconds", float("nan")))
        return BridgeSolution(
            time_grid=time_grid,
            mean=mean,
            mean_velocity=mean_velocity,
            std=std,
            std_velocity=std_velocity,
            x0=x0_keep,
            x1=x1_keep,
            path=path,
            success_mask=success_mask,
            success_indices=path.success_indices,
            failed_indices=path.failed_indices,
            endpoint_errors=endpoint_errors,
            residual_norms=residual_norms,
            failure_messages=dict(path.failure_messages),
            solve_time_seconds=solve_time_seconds,
            pair_solve_time_seconds=pair_solve_times,
            solver_iterations=solver_iterations,
            solver_mesh_nodes=solver_mesh_nodes,
            guess_source=guess_source,
        )

    def _endpoint_errors(self, mean: Tensor, std: Tensor, x0: Tensor, x1: Tensor) -> Tensor:
        if mean.numel() == 0:
            return torch.empty((0, 4), device=x0.device, dtype=x0.dtype)
        x0_flat = x0.reshape(x0.shape[0], -1)
        x1_flat = x1.reshape(x1.shape[0], -1)
        sigma = torch.as_tensor(self.sigma, dtype=x0.dtype, device=x0.device)
        return torch.stack(
            [
                (mean[:, 0, :] - x0_flat).norm(dim=-1),
                (mean[:, -1, :] - x1_flat).norm(dim=-1),
                (std[:, 0, 0] - sigma).abs(),
                (std[:, -1, 0] - sigma).abs(),
            ],
            dim=-1,
        )
