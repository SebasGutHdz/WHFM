"""Gaussian probability paths for Hamiltonian flow matching."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Tuple
import time

import numpy as np
import torch
from torch import Tensor

from .solvers import (
    make_double_well_gaussian_bvp_rhs,
    make_particle_bvp_rhs,
    scipy_solve_bvp,
)


def _time_column(t, x: Tensor) -> Tensor:
    if not torch.is_tensor(t):
        return torch.full((x.shape[0], 1), float(t), dtype=x.dtype, device=x.device)
    t = t.to(device=x.device, dtype=x.dtype)
    if t.dim() == 0:
        return t.reshape(1, 1).expand(x.shape[0], 1)
    if t.dim() == 1:
        return t.reshape(-1, 1)
    return t.reshape(t.shape[0], -1)[:, :1]


def _time_like_x(t, x: Tensor) -> Tensor:
    return _time_column(t, x).reshape(-1, *([1] * (x.dim() - 1)))


def _sigma_like_x(sigma: Tensor, x: Tensor) -> Tensor:
    if sigma.dim() == 0:
        return sigma.reshape(1, *([1] * (x.dim() - 1))).expand_as(x[..., :1])
    return sigma.reshape(-1, *([1] * (x.dim() - 1)))


def _pair_key(x0: Tensor, x1: Tensor):
    left = tuple(float(v) for v in x0.detach().cpu().reshape(-1).tolist())
    right = tuple(float(v) for v in x1.detach().cpu().reshape(-1).tolist())
    return left, right


class GaussianPath(ABC):
    """Base class for Gaussian path interpolants."""

    @abstractmethod
    def compute(self, x0: Tensor, x1: Tensor, t: Tensor, return_derivatives: bool = True):
        """Return path mean/sigma values, optionally with derivatives."""

    def sample_xt(self, x0: Tensor, x1: Tensor, t: Tensor, epsilon: Tensor) -> Tensor:
        mu_t, sigma_t = self.compute(x0, x1, t, return_derivatives=False)
        return mu_t + _sigma_like_x(sigma_t, x0) * epsilon

    def compute_ut(self, x0: Tensor, x1: Tensor, t: Tensor, xt: Tensor) -> Tensor:
        mu_t, mu_t_prime, sigma_t, sigma_t_prime = self.compute(
            x0, x1, t, return_derivatives=True
        )
        sigma_t = _sigma_like_x(sigma_t, xt)
        sigma_t_prime = _sigma_like_x(sigma_t_prime, xt)
        return sigma_t_prime * (xt - mu_t) / (sigma_t + 1e-8) + mu_t_prime


class HarmonicGaussianPath(GaussianPath):
    """Closed-form harmonic Gaussian path."""

    def __init__(self, U: Tensor, sigma: float = 0.5):
        U = torch.as_tensor(U, dtype=torch.get_default_dtype())
        self.U = U
        self.D, self.Q = torch.linalg.eigh(U, UPLO="U")
        self.sigma = sigma
        self.sqrt_trace_U = torch.sqrt(torch.trace(U))

    def compute(self, x0: Tensor, x1: Tensor, t: Tensor, return_derivatives: bool = True):
        original_shape = x0.shape
        x0_flat = x0.reshape(x0.shape[0], -1)
        x1_flat = x1.reshape(x1.shape[0], -1)
        t_col = _time_column(t, x0_flat)

        D = self.D.to(device=x0.device, dtype=x0.dtype)
        Q = self.Q.to(device=x0.device, dtype=x0.dtype)
        sqrt_D = torch.sqrt(D.clamp_min(0.0))
        D_t = sqrt_D * t_col

        cos_D_t = torch.diag_embed(torch.cos(D_t))
        sin_D_t = torch.diag_embed(torch.sin(D_t))
        cos_D_1 = torch.diag(torch.cos(sqrt_D))
        inv_sin_D_1 = torch.diag(1.0 / (torch.sin(sqrt_D) + 1e-8))
        sqrt_D_mat = torch.diag(sqrt_D)

        x0_v = x0_flat.unsqueeze(-1)
        x1_v = x1_flat.unsqueeze(-1)
        qtx0 = Q.T @ x0_v
        qtx1 = Q.T @ x1_v
        endpoint_term = -cos_D_1 @ qtx0 + qtx1

        mu = (Q @ (cos_D_t @ qtx0 + sin_D_t @ inv_sin_D_1 @ endpoint_term)).squeeze(-1)
        mu = mu.reshape(original_shape)

        sqrt_trace = self.sqrt_trace_U.to(device=x0.device, dtype=x0.dtype)
        sigma_value = torch.as_tensor(self.sigma, dtype=x0.dtype, device=x0.device)
        coeff = (1.0 - torch.cos(sqrt_trace)) / (torch.sin(sqrt_trace) + 1e-8)
        sigma_t = sigma_value * (torch.cos(sqrt_trace * t_col) + torch.sin(sqrt_trace * t_col) * coeff)

        if not return_derivatives:
            return mu, sigma_t

        mu_prime = (
            Q
            @ sqrt_D_mat
            @ (-sin_D_t @ qtx0 + cos_D_t @ inv_sin_D_1 @ endpoint_term)
        ).squeeze(-1)
        mu_prime = mu_prime.reshape(original_shape)
        sigma_t_prime = sigma_value * sqrt_trace * (
            -torch.sin(sqrt_trace * t_col) + torch.cos(sqrt_trace * t_col) * coeff
        )
        return mu, mu_prime, sigma_t, sigma_t_prime


class HillGaussianPath(GaussianPath):
    """Closed-form Gaussian path for the hill potential."""

    def __init__(self, alpha: float = 3.0, sigma: float = 0.01):
        self.alpha = alpha
        self.sigma = sigma

    def compute(self, x0: Tensor, x1: Tensor, t: Tensor, return_derivatives: bool = True):
        t_x = _time_like_x(t, x0)
        t_col = _time_column(t, x0)
        alpha = torch.as_tensor(self.alpha, dtype=x0.dtype, device=x0.device)
        root = torch.sqrt(alpha)
        exp_root = torch.exp(root)
        exp_neg_root = torch.exp(-root)
        const_mu = 1.0 / (exp_root - exp_neg_root)

        a = -x0 * exp_neg_root + x1
        b = x0 * exp_root - x1
        mu = const_mu * (torch.exp(root * t_x) * a + torch.exp(-root * t_x) * b)

        root_sigma = torch.sqrt(2.0 * alpha)
        sigma_value = torch.as_tensor(self.sigma, dtype=x0.dtype, device=x0.device)
        const_1 = sigma_value / (torch.exp(root_sigma) - torch.exp(-root_sigma))
        const_2 = 1.0 - torch.exp(-root_sigma)
        const_3 = torch.exp(root_sigma) - 1.0
        sigma_t = const_1 * (
            const_2 * torch.exp(root_sigma * t_col)
            + const_3 * torch.exp(-root_sigma * t_col)
        )

        if not return_derivatives:
            return mu, sigma_t

        mu_prime = root * const_mu * (torch.exp(root * t_x) * a - torch.exp(-root * t_x) * b)
        sigma_t_prime = const_1 * root_sigma * (
            const_2 * torch.exp(root_sigma * t_col)
            - const_3 * torch.exp(-root_sigma * t_col)
        )
        return mu, mu_prime, sigma_t, sigma_t_prime


class DensityGaussianPath(GaussianPath):
    """Linear-mean path with a precomputed sigma schedule."""

    def __init__(
        self,
        potential,
        sigma_0,
        sigma_dot_0=None,
        *,
        n_steps: int = 200,
        method: str = "scipy",
    ):
        self.potential = potential
        self.schedule = potential.compute_sigma_schedule(
            sigma_0, sigma_dot_0, n_steps=n_steps, method=method
        )

    def compute(self, x0: Tensor, x1: Tensor, t: Tensor, return_derivatives: bool = True):
        t_x = _time_like_x(t, x0)
        mu = (1.0 - t_x) * x0 + t_x * x1
        sigma_t, sigma_t_prime = self.schedule.to(device=x0.device, dtype=x0.dtype).evaluate(
            _time_column(t, x0)
        )
        if not return_derivatives:
            return mu, sigma_t
        mu_prime = x1 - x0
        return mu, mu_prime, sigma_t, sigma_t_prime


class InteractionGaussianPath(DensityGaussianPath):
    """Alias path for interaction-driven sigma schedules."""


class _CachedBVPPath(GaussianPath):
    def __init__(self, n_steps: int = 50, tol: float = 1e-4, max_nodes: int = 1000):
        self.n_steps = n_steps
        self.tol = tol
        self.max_nodes = max_nodes
        self.t_grid = torch.linspace(0.0, 1.0, n_steps + 1)
        self._cache = None
        self._cache_index: Dict[Tuple[Tuple[float, ...], Tuple[float, ...]], int] = {}
        self.success_mask = None
        self.success_indices = None
        self.failed_indices = None
        self.failure_messages = {}
        self.solve_metadata = {}

    def _record_solve_metadata(self, n_pairs: int, success_indices, failure_messages, solve_metadata=None):
        success_indices_t = torch.as_tensor(success_indices, dtype=torch.long)
        success_mask = torch.zeros(n_pairs, dtype=torch.bool)
        if success_indices_t.numel() > 0:
            success_mask[success_indices_t] = True
        self.success_mask = success_mask
        self.success_indices = success_indices_t
        success_set = set(success_indices)
        self.failed_indices = torch.as_tensor(
            [i for i in range(n_pairs) if i not in success_set], dtype=torch.long
        )
        self.failure_messages = dict(failure_messages)
        self.solve_metadata = {} if solve_metadata is None else dict(solve_metadata)

    def _store_cache(self, x0: Tensor, x1: Tensor, states: Tensor):
        self._cache = {
            "x0": x0.detach().cpu(),
            "x1": x1.detach().cpu(),
            "states": states.detach().cpu(),
        }
        self._cache_index = {
            _pair_key(self._cache["x0"][i], self._cache["x1"][i]): i
            for i in range(self._cache["x0"].shape[0])
        }

    def _store_successful_cache(
        self,
        x0: Tensor,
        x1: Tensor,
        states,
        success_indices,
        failure_messages,
        failure_prefix: str,
        solve_metadata=None,
    ) -> Tensor:
        self._record_solve_metadata(x0.shape[0], success_indices, failure_messages, solve_metadata)
        if not states:
            messages = "; ".join(
                f"{idx}: {message}" for idx, message in list(failure_messages.items())[:5]
            )
            suffix = f" First failures: {messages}" if messages else ""
            raise RuntimeError(f"{failure_prefix}: all {x0.shape[0]} BVP solves failed.{suffix}")

        states_t = torch.stack(states, dim=0)
        keep = torch.as_tensor(success_indices, dtype=torch.long)
        self._store_cache(x0[keep], x1[keep], states_t)
        return states_t

    def _lookup_states(self, x0: Tensor, x1: Tensor) -> Tensor:
        if self._cache is None:
            raise RuntimeError("BVP path cache is empty. Call batch_solve(x0, x1) before sampling.")
        indices = []
        for i in range(x0.shape[0]):
            key = _pair_key(x0[i], x1[i])
            if key not in self._cache_index:
                raise RuntimeError("Requested BVP pair was not found in the precomputed cache.")
            indices.append(self._cache_index[key])
        states = self._cache["states"][torch.as_tensor(indices, dtype=torch.long)]
        return states.to(device=x0.device, dtype=x0.dtype)

    def _interpolate_states(self, states: Tensor, t: Tensor) -> Tensor:
        t_col = _time_column(t, states)
        grid = self.t_grid.to(device=states.device, dtype=states.dtype)
        t_flat = t_col.reshape(-1).clamp(0.0, 1.0)
        right = torch.searchsorted(grid, t_flat, right=False).clamp(1, grid.numel() - 1)
        left = right - 1
        weight = ((t_flat - grid[left]) / (grid[right] - grid[left]).clamp_min(1e-12)).reshape(
            -1, 1
        )
        return states[torch.arange(states.shape[0], device=states.device), left] + weight * (
            states[torch.arange(states.shape[0], device=states.device), right]
            - states[torch.arange(states.shape[0], device=states.device), left]
        )


class ParticleBVPGaussianPath(_CachedBVPPath):
    """SciPy-BVP-backed particle path ``[x, v]``."""

    def __init__(self, potential, sigma: float = 0.01, n_steps: int = 50, tol: float = 1e-4):
        super().__init__(n_steps=n_steps, tol=tol)
        self.potential = potential
        self.sigma = sigma

    def batch_solve(self, x0: Tensor, x1: Tensor) -> Tensor:
        x0_cpu = x0.detach().cpu()
        x1_cpu = x1.detach().cpu()
        dim = x0_cpu.reshape(x0_cpu.shape[0], -1).shape[1]
        grid = np.linspace(0.0, 1.0, self.n_steps + 1)
        rhs = make_particle_bvp_rhs(self.potential)
        states = []
        success_indices = []
        failure_messages = {}
        for i, (start_t, end_t) in enumerate(
            zip(x0_cpu.reshape(x0_cpu.shape[0], -1), x1_cpu.reshape(x1_cpu.shape[0], -1))
        ):
            start = start_t.numpy()
            end = end_t.numpy()
            guess_x = ((1.0 - grid[:, None]) * start + grid[:, None] * end).T
            guess_v = np.repeat((end - start)[:, None], grid.size, axis=1)
            guess = np.vstack([guess_x, guess_v])

            def bc(ya, yb):
                return np.concatenate([ya[:dim] - start, yb[:dim] - end])

            result = scipy_solve_bvp(
                rhs,
                bc,
                grid,
                guess,
                tol=self.tol,
                max_nodes=self.max_nodes,
                to_tensor=False,
            )
            if not result.success:
                failure_messages[i] = result.message
                continue
            states.append(torch.as_tensor(result.raw.sol(grid).T, dtype=x0_cpu.dtype))
            success_indices.append(i)

        return self._store_successful_cache(
            x0_cpu,
            x1_cpu,
            states,
            success_indices,
            failure_messages,
            "SciPy particle BVP failed",
        )

    def compute(self, x0: Tensor, x1: Tensor, t: Tensor, return_derivatives: bool = True):
        states = self._interpolate_states(self._lookup_states(x0, x1), t)
        dim = x0.reshape(x0.shape[0], -1).shape[1]
        mu = states[:, :dim].reshape_as(x0)
        sigma_t = torch.full((_time_column(t, x0).shape[0], 1), self.sigma, dtype=x0.dtype, device=x0.device)
        if not return_derivatives:
            return mu, sigma_t
        mu_prime = states[:, dim:].reshape_as(x0)
        sigma_t_prime = torch.zeros_like(sigma_t)
        return mu, mu_prime, sigma_t, sigma_t_prime


class DeterministicBVPPath(_CachedBVPPath):
    """SciPy-BVP-backed deterministic Hamiltonian path ``[gamma, gamma_prime]``."""

    def __init__(
        self,
        potential,
        n_steps: int = 50,
        tol: float = 1e-4,
        max_nodes: int = 1000,
        gamma_guess=None,
        gamma_prime_guess=None,
    ):
        super().__init__(n_steps=n_steps, tol=tol, max_nodes=max_nodes)
        self.potential = potential
        self.gamma_guess = gamma_guess
        self.gamma_prime_guess = gamma_prime_guess

    @staticmethod
    def _to_numpy_guess(value, name: str):
        if value is None:
            return None
        if torch.is_tensor(value):
            value = value.detach().cpu().numpy()
        array = np.asarray(value, dtype=float)
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain only finite values.")
        return array

    def _prepare_vector_guess(self, value, name: str, n_pairs: int, dim: int):
        array = self._to_numpy_guess(value, name)
        if array is None:
            return None
        expected = (n_pairs, self.n_steps + 1, dim)
        if array.shape != expected:
            raise ValueError(f"{name} must have shape {expected}; got {array.shape}.")
        return array

    def batch_solve(self, x0: Tensor, x1: Tensor) -> Tensor:
        x0_cpu = x0.detach().cpu()
        x1_cpu = x1.detach().cpu()
        x0_flat = x0_cpu.reshape(x0_cpu.shape[0], -1)
        x1_flat = x1_cpu.reshape(x1_cpu.shape[0], -1)
        n_pairs, dim = x0_flat.shape
        grid = np.linspace(0.0, 1.0, self.n_steps + 1)
        rhs = make_particle_bvp_rhs(self.potential)

        gamma_guess = self._prepare_vector_guess(
            self.gamma_guess, "gamma_guess", n_pairs, dim
        )
        gamma_prime_guess = self._prepare_vector_guess(
            self.gamma_prime_guess, "gamma_prime_guess", n_pairs, dim
        )

        default_gamma = (
            (1.0 - grid[None, :, None]) * x0_flat[:, None, :].numpy()
            + grid[None, :, None] * x1_flat[:, None, :].numpy()
        )
        default_gamma_prime = np.repeat(
            (x1_flat - x0_flat).numpy()[:, None, :], grid.size, axis=1
        )

        solve_gamma_guess = default_gamma if gamma_guess is None else gamma_guess
        solve_gamma_prime_guess = (
            default_gamma_prime if gamma_prime_guess is None else gamma_prime_guess
        )

        states = []
        success_indices = []
        failure_messages = {}
        batch_solve_start = time.perf_counter()
        pair_solve_times = [float("nan")] * n_pairs
        pair_solver_iterations = [None] * n_pairs
        pair_solver_mesh_nodes = [None] * n_pairs
        for i, (start_t, end_t) in enumerate(zip(x0_flat, x1_flat)):
            start = start_t.numpy()
            end = end_t.numpy()
            guess = np.vstack(
                [
                    solve_gamma_guess[i].T,
                    solve_gamma_prime_guess[i].T,
                ]
            )

            def bc(ya, yb):
                return np.concatenate([ya[:dim] - start, yb[:dim] - end])

            solve_start = time.perf_counter()
            result = scipy_solve_bvp(
                rhs,
                bc,
                grid,
                guess,
                tol=self.tol,
                max_nodes=self.max_nodes,
                to_tensor=False,
            )
            pair_solve_times[i] = time.perf_counter() - solve_start
            pair_solver_iterations[i] = getattr(result.raw, "niter", None)
            if pair_solver_iterations[i] is not None:
                pair_solver_iterations[i] = int(pair_solver_iterations[i])
            pair_solver_mesh_nodes[i] = int(np.asarray(getattr(result.raw, "x", grid)).size)
            if not result.success:
                failure_messages[i] = result.message
                continue
            states.append(torch.as_tensor(result.raw.sol(grid).T, dtype=x0_cpu.dtype))
            success_indices.append(i)

        solve_metadata = {
            "total_solve_time_seconds": time.perf_counter() - batch_solve_start,
            "pair_solve_time_seconds": pair_solve_times,
            "pair_solver_iterations": pair_solver_iterations,
            "pair_solver_mesh_nodes": pair_solver_mesh_nodes,
        }
        return self._store_successful_cache(
            x0_cpu,
            x1_cpu,
            states,
            success_indices,
            failure_messages,
            "SciPy deterministic BVP failed",
            solve_metadata=solve_metadata,
        )

    def sample_xt(self, x0: Tensor, x1: Tensor, t: Tensor, epsilon: Tensor) -> Tensor:
        gamma, _ = self.compute(x0, x1, t, return_derivatives=False)
        return gamma

    def compute_ut(self, x0: Tensor, x1: Tensor, t: Tensor, xt: Tensor) -> Tensor:
        _, gamma_prime, _, _ = self.compute(x0, x1, t, return_derivatives=True)
        return gamma_prime

    def compute(self, x0: Tensor, x1: Tensor, t: Tensor, return_derivatives: bool = True):
        states = self._interpolate_states(self._lookup_states(x0, x1), t)
        dim = x0.reshape(x0.shape[0], -1).shape[1]
        gamma = states[:, :dim].reshape_as(x0)
        sigma_t = torch.zeros((_time_column(t, x0).shape[0], 1), dtype=x0.dtype, device=x0.device)
        if not return_derivatives:
            return gamma, sigma_t
        gamma_prime = states[:, dim:].reshape_as(x0)
        sigma_t_prime = torch.zeros_like(sigma_t)
        return gamma, gamma_prime, sigma_t, sigma_t_prime


class MeanStdBVPGaussianPath(_CachedBVPPath):
    """SciPy-BVP-backed Gaussian path with direct scalar standard deviation."""

    def __init__(
        self,
        potential,
        sigma: float = 1e-3,
        n_steps: int = 30,
        tol: float = 1e-3,
        max_nodes: int = 1000,
        quadrature_order: int = 7,
        mu_guess=None,
        mu_dot_guess=None,
        sigma_guess=None,
        sigma_dot_guess=None,
        interaction_potential=None,
        interaction_coefficient: float = 1.0,
        n_density_samples: int = 1,
        n_reference_grid: int = None,
        use_monte_carlo: bool = False,
        monte_carlo_samples: int = 100,
    ):
        if sigma <= 0:
            raise ValueError("sigma must be positive. Use DeterministicBVPPath for sigma=0.")
        if quadrature_order < 1:
            raise ValueError("quadrature_order must be positive.")
        if n_density_samples < 1:
            raise ValueError("n_density_samples must be positive.")
        if n_reference_grid is not None and n_reference_grid < 2:
            raise ValueError("n_reference_grid must be at least 2.")
        if monte_carlo_samples < 1:
            raise ValueError("monte_carlo_samples must be positive.")
        super().__init__(n_steps=n_steps, tol=tol, max_nodes=max_nodes)
        self.potential = potential
        self.sigma = float(sigma)
        self.quadrature_order = int(quadrature_order)
        self.n_density_samples = int(n_density_samples)
        self.n_reference_grid = n_reference_grid
        self.use_monte_carlo = bool(use_monte_carlo)
        self.monte_carlo_samples = int(monte_carlo_samples)
        self._monte_carlo_rules = {}
        self.mu_guess = mu_guess
        self.mu_dot_guess = mu_dot_guess
        self.sigma_guess = sigma_guess
        self.sigma_dot_guess = sigma_dot_guess
        configured_interaction = interaction_potential is None and getattr(
            potential, "has_interaction", False
        )
        self.interaction_potential = potential if configured_interaction else interaction_potential
        self.interaction_coefficient = 1.0 if configured_interaction else float(interaction_coefficient)
        self.reference_t_grid = None
        self.reference_samples = None
        self.reference_indices = None
        self.reference_noise = None
        self.reference_states = None

    @staticmethod
    def _to_numpy_guess(value, name: str):
        if value is None:
            return None
        if torch.is_tensor(value):
            value = value.detach().cpu().numpy()
        array = np.asarray(value, dtype=float)
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain only finite values.")
        return array

    def _prepare_vector_guess(self, value, name: str, n_pairs: int, dim: int):
        array = self._to_numpy_guess(value, name)
        if array is None:
            return None
        expected = (n_pairs, self.n_steps + 1, dim)
        if array.shape != expected:
            raise ValueError(f"{name} must have shape {expected}; got {array.shape}.")
        return array

    def _prepare_scalar_guess(self, value, name: str, n_pairs: int, positive: bool = False):
        array = self._to_numpy_guess(value, name)
        if array is None:
            return None
        if array.shape == (n_pairs, self.n_steps + 1, 1):
            array = array[..., 0]
        expected = (n_pairs, self.n_steps + 1)
        if array.shape != expected:
            raise ValueError(
                f"{name} must have shape {expected} or {expected + (1,)}; got {array.shape}."
            )
        if positive and np.any(array <= 0):
            raise ValueError(f"{name} must contain only positive values.")
        return array

    def _normal_quadrature(self, dim: int):
        nodes_1d, weights_1d = np.polynomial.hermite.hermgauss(self.quadrature_order)
        nodes_1d = np.sqrt(2.0) * nodes_1d
        weights_1d = weights_1d / np.sqrt(np.pi)
        if self.quadrature_order == 1:
            return np.zeros((1, dim), dtype=float), np.ones(1, dtype=float)
        node_grids = np.meshgrid(*([nodes_1d] * dim), indexing="ij")
        weight_grids = np.meshgrid(*([weights_1d] * dim), indexing="ij")
        eps = np.stack([grid.reshape(-1) for grid in node_grids], axis=-1)
        weights = np.prod(np.stack(weight_grids, axis=-1), axis=-1).reshape(-1)
        return eps, weights

    def _normal_monte_carlo(self, dim: int):
        cached = self._monte_carlo_rules.get(dim)
        if cached is not None:
            return cached

        n_pairs = (self.monte_carlo_samples + 1) // 2
        base = torch.randn(n_pairs, dim, dtype=torch.float64)
        eps_t = torch.cat([base, -base], dim=0)[: self.monte_carlo_samples]
        weights_t = torch.full(
            (self.monte_carlo_samples,),
            1.0 / self.monte_carlo_samples,
            dtype=torch.float64,
        )
        rule = (eps_t.numpy(), weights_t.numpy())
        self._monte_carlo_rules[dim] = rule
        return rule

    def _normal_integration_rule(self, dim: int):
        if self.use_monte_carlo:
            return self._normal_monte_carlo(dim)
        return self._normal_quadrature(dim)

    @staticmethod
    def _interpolate_state_grid(states: np.ndarray, source_grid: np.ndarray, target_grid: np.ndarray):
        if np.array_equal(source_grid, target_grid):
            return states.copy()
        flat = states.reshape(states.shape[0], -1)
        interpolated = np.stack(
            [np.interp(target_grid, source_grid, flat[:, j]) for j in range(flat.shape[1])],
            axis=-1,
        )
        return interpolated.reshape(target_grid.shape[0], *states.shape[1:])

    @staticmethod
    def _interpolate_reference_samples(
        t,
        n_mesh: int,
        reference_t_grid: np.ndarray,
        reference_samples: np.ndarray,
    ):
        t_flat = np.asarray(t, dtype=float).reshape(-1)
        if t_flat.size == 1 and n_mesh != 1:
            t_flat = np.full(n_mesh, float(t_flat[0]), dtype=float)
        if t_flat.size != n_mesh:
            raise ValueError(
                f"RHS time array has {t_flat.size} entries, but state has {n_mesh} mesh columns."
            )

        t_flat = np.clip(t_flat, reference_t_grid[0], reference_t_grid[-1])
        right = np.searchsorted(reference_t_grid, t_flat, side="left")
        right = np.clip(right, 1, reference_t_grid.size - 1)
        left = right - 1
        denom = np.maximum(reference_t_grid[right] - reference_t_grid[left], 1e-12)
        weight = ((t_flat - reference_t_grid[left]) / denom).reshape(-1, 1, 1)
        return reference_samples[left] + weight * (
            reference_samples[right] - reference_samples[left]
        )

    def _state_from_guesses(self, mu, mu_dot, sigma, sigma_dot):
        return np.concatenate(
            [
                mu,
                mu_dot,
                sigma.reshape(-1, 1),
                sigma_dot.reshape(-1, 1),
            ],
            axis=1,
        )

    def _cached_state_for_pair(self, x0_i: Tensor, x1_i: Tensor):
        if self._cache is None:
            return None
        index = self._cache_index.get(_pair_key(x0_i, x1_i))
        if index is None:
            return None
        return self._cache["states"][index].detach().cpu().numpy()

    def _build_reference_samples(
        self,
        x0_flat: Tensor,
        x1_flat: Tensor,
        mu_guess: np.ndarray,
        mu_dot_guess: np.ndarray,
        sigma_guess: np.ndarray,
        sigma_dot_guess: np.ndarray,
    ):
        n_pairs, dim = x0_flat.shape
        solve_grid = np.linspace(0.0, 1.0, self.n_steps + 1)
        n_reference_grid = self.n_steps + 1 if self.n_reference_grid is None else self.n_reference_grid
        reference_t_grid = np.linspace(0.0, 1.0, n_reference_grid)

        reference_states = []
        for i in range(n_pairs):
            cached = self._cached_state_for_pair(x0_flat[i], x1_flat[i])
            if cached is None:
                cached = self._state_from_guesses(
                    mu_guess[i],
                    mu_dot_guess[i],
                    sigma_guess[i],
                    sigma_dot_guess[i],
                )
            reference_states.append(cached)
        reference_states = np.stack(reference_states, axis=0)

        sample_indices = torch.randint(n_pairs, (self.n_density_samples,)).detach().cpu().numpy()
        sample_noise = torch.randn(self.n_density_samples, dim).detach().cpu().numpy()

        selected_states = reference_states[sample_indices]
        selected_states = np.stack(
            [
                self._interpolate_state_grid(state, solve_grid, reference_t_grid)
                for state in selected_states
            ],
            axis=1,
        )
        mu_ref = selected_states[:, :, :dim]
        sigma_ref = selected_states[:, :, 2 * dim : 2 * dim + 1]
        reference_samples = mu_ref + sigma_ref * sample_noise[None, :, :]

        self.reference_t_grid = reference_t_grid
        self.reference_samples = reference_samples
        self.reference_indices = sample_indices
        self.reference_noise = sample_noise
        self.reference_states = reference_states
        return reference_t_grid, reference_samples

    def _make_rhs(self, dim: int, reference_t_grid=None, reference_samples=None):
        eps, weights = self._normal_integration_rule(dim)
        q = eps.shape[0]
        has_interaction = self.interaction_potential is not None
        if has_interaction and (reference_t_grid is None or reference_samples is None):
            reference_t_grid = self.reference_t_grid
            reference_samples = self.reference_samples
        if has_interaction and (reference_t_grid is None or reference_samples is None):
            raise RuntimeError(
                "Interaction RHS requires frozen reference samples. Call batch_solve "
                "or pass reference_t_grid/reference_samples to _make_rhs."
            )
        linear_gradient = getattr(self.potential, "linear_gradient", None)
        if linear_gradient is None:
            linear_gradient = self.potential.gradient

        def rhs(_t, state):
            n_mesh = state.shape[1]
            mu = state[:dim].T
            mu_dot = state[dim : 2 * dim]
            sigma = state[2 * dim]
            sigma_dot = state[2 * dim + 1]

            x_quad = mu[:, None, :] + sigma[:, None, None] * eps[None, :, :]
            x_quad_t = torch.as_tensor(x_quad.reshape(-1, dim), dtype=torch.float64)
            grad = linear_gradient(x_quad_t).detach().cpu().numpy().reshape(n_mesh, q, dim)

            mean_grad = -np.sum(weights[None, :, None] * grad, axis=1)
            sigma_accel = -np.sum(
                weights[None, :] * np.sum(grad * eps[None, :, :], axis=-1),
                axis=1,
            ) / dim

            if has_interaction:
                y_ref = self._interpolate_reference_samples(
                    _t, n_mesh, reference_t_grid, reference_samples
                )
                interaction_x = np.broadcast_to(
                    x_quad[:, :, None, :],
                    (n_mesh, q, y_ref.shape[1], dim),
                )
                interaction_y = np.broadcast_to(
                    y_ref[:, None, :, :],
                    (n_mesh, q, y_ref.shape[1], dim),
                )
                interaction_x_t = torch.as_tensor(
                    np.array(interaction_x.reshape(-1, dim), copy=True), dtype=torch.float64
                )
                interaction_y_t = torch.as_tensor(
                    np.array(interaction_y.reshape(-1, dim), copy=True), dtype=torch.float64
                )
                interaction_grad = (
                    self.interaction_potential.interaction_gradient(interaction_x_t, interaction_y_t)
                    .detach()
                    .cpu()
                    .numpy()
                    .reshape(n_mesh, q, y_ref.shape[1], dim)
                )
                interaction_mean_grad = interaction_grad.mean(axis=2)
                interaction_mu_accel = np.sum(
                    weights[None, :, None] * interaction_mean_grad,
                    axis=1,
                )
                interaction_sigma_accel = np.sum(
                    weights[None, :]
                    * np.sum(interaction_mean_grad * eps[None, :, :], axis=-1),
                    axis=1,
                ) / dim
                mean_grad = mean_grad - self.interaction_coefficient * interaction_mu_accel
                sigma_accel = sigma_accel - self.interaction_coefficient * interaction_sigma_accel

            return np.vstack(
                [mu_dot, mean_grad.T, sigma_dot.reshape(1, -1), sigma_accel.reshape(1, -1)]
            )

        return rhs

    def batch_solve(self, x0: Tensor, x1: Tensor) -> Tensor:
        x0_cpu = x0.detach().cpu()
        x1_cpu = x1.detach().cpu()
        x0_flat = x0_cpu.reshape(x0_cpu.shape[0], -1)
        x1_flat = x1_cpu.reshape(x1_cpu.shape[0], -1)
        n_pairs, dim = x0_flat.shape
        grid = np.linspace(0.0, 1.0, self.n_steps + 1)

        mu_guess = self._prepare_vector_guess(self.mu_guess, "mu_guess", n_pairs, dim)
        mu_dot_guess = self._prepare_vector_guess(
            self.mu_dot_guess, "mu_dot_guess", n_pairs, dim
        )
        sigma_guess = self._prepare_scalar_guess(
            self.sigma_guess, "sigma_guess", n_pairs, positive=True
        )
        sigma_dot_guess = self._prepare_scalar_guess(
            self.sigma_dot_guess, "sigma_dot_guess", n_pairs
        )

        default_mu = (
            (1.0 - grid[None, :, None]) * x0_flat[:, None, :].numpy()
            + grid[None, :, None] * x1_flat[:, None, :].numpy()
        )
        default_mu_dot = np.repeat(
            (x1_flat - x0_flat).numpy()[:, None, :], grid.size, axis=1
        )
        default_sigma = np.full((n_pairs, grid.size), self.sigma)
        default_sigma_dot = np.zeros((n_pairs, grid.size))

        solve_mu_guess = default_mu if mu_guess is None else mu_guess
        solve_mu_dot_guess = default_mu_dot if mu_dot_guess is None else mu_dot_guess
        solve_sigma_guess = default_sigma if sigma_guess is None else sigma_guess
        solve_sigma_dot_guess = default_sigma_dot if sigma_dot_guess is None else sigma_dot_guess

        if self.interaction_potential is not None:
            reference_t_grid, reference_samples = self._build_reference_samples(
                x0_flat,
                x1_flat,
                solve_mu_guess,
                solve_mu_dot_guess,
                solve_sigma_guess,
                solve_sigma_dot_guess,
            )
            rhs = self._make_rhs(dim, reference_t_grid, reference_samples)
        else:
            rhs = self._make_rhs(dim)

        states = []
        success_indices = []
        failure_messages = {}
        batch_solve_start = time.perf_counter()
        pair_solve_times = [float("nan")] * n_pairs
        pair_solver_iterations = [None] * n_pairs
        pair_solver_mesh_nodes = [None] * n_pairs
        for i, (start_t, end_t) in enumerate(zip(x0_flat, x1_flat)):
            start = start_t.numpy()
            end = end_t.numpy()

            guess_mu = solve_mu_guess[i]
            guess_mu_dot = solve_mu_dot_guess[i]
            guess_sigma = solve_sigma_guess[i]
            guess_sigma_dot = solve_sigma_dot_guess[i]
            guess = np.vstack(
                [
                    guess_mu.T,
                    guess_mu_dot.T,
                    guess_sigma.reshape(1, -1),
                    guess_sigma_dot.reshape(1, -1),
                ]
            )

            def bc(ya, yb):
                return np.concatenate(
                    [
                        ya[:dim] - start,
                        yb[:dim] - end,
                        np.asarray([ya[2 * dim] - self.sigma, yb[2 * dim] - self.sigma]),
                    ]
                )

            solve_start = time.perf_counter()
            result = scipy_solve_bvp(
                rhs,
                bc,
                grid,
                guess,
                tol=self.tol,
                max_nodes=self.max_nodes,
                to_tensor=False,
            )
            pair_solve_times[i] = time.perf_counter() - solve_start
            pair_solver_iterations[i] = getattr(result.raw, "niter", None)
            if pair_solver_iterations[i] is not None:
                pair_solver_iterations[i] = int(pair_solver_iterations[i])
            pair_solver_mesh_nodes[i] = int(np.asarray(getattr(result.raw, "x", grid)).size)
            if not result.success:
                failure_messages[i] = result.message
                continue
            state = torch.as_tensor(result.raw.sol(grid).T, dtype=x0_cpu.dtype)
            if torch.any(state[:, 2 * dim] <= 0):
                failure_messages[i] = "SciPy mean/std BVP returned a nonpositive sigma path."
                continue
            states.append(state)
            success_indices.append(i)

        solve_metadata = {
            "total_solve_time_seconds": time.perf_counter() - batch_solve_start,
            "pair_solve_time_seconds": pair_solve_times,
            "pair_solver_iterations": pair_solver_iterations,
            "pair_solver_mesh_nodes": pair_solver_mesh_nodes,
        }
        return self._store_successful_cache(
            x0_cpu,
            x1_cpu,
            states,
            success_indices,
            failure_messages,
            "SciPy mean/std BVP failed",
            solve_metadata=solve_metadata,
        )

    def compute(self, x0: Tensor, x1: Tensor, t: Tensor, return_derivatives: bool = True):
        states = self._interpolate_states(self._lookup_states(x0, x1), t)
        dim = x0.reshape(x0.shape[0], -1).shape[1]
        mu = states[:, :dim].reshape_as(x0)
        sigma_t = states[:, 2 * dim : 2 * dim + 1]
        if torch.any(sigma_t <= 0):
            raise RuntimeError("Cached mean/std BVP contains a nonpositive sigma value.")
        if not return_derivatives:
            return mu, sigma_t
        mu_prime = states[:, dim : 2 * dim].reshape_as(x0)
        sigma_t_prime = states[:, 2 * dim + 1 : 2 * dim + 2]
        return mu, mu_prime, sigma_t, sigma_t_prime


class ParametricBVPGaussianPath(_CachedBVPPath):
    """SciPy-BVP-backed 1D Gaussian-parametric path ``[mu, mu', sigma, sigma']``."""

    def __init__(self, potential, sigma: float = 1e-4, n_steps: int = 30, tol: float = 1.0):
        super().__init__(n_steps=n_steps, tol=tol)
        self.potential = potential
        self.sigma = sigma

    def batch_solve(self, x0: Tensor, x1: Tensor) -> Tensor:
        x0_cpu = x0.detach().cpu().reshape(x0.shape[0], -1)
        x1_cpu = x1.detach().cpu().reshape(x1.shape[0], -1)
        if x0_cpu.shape[1] != 1:
            raise ValueError("ParametricBVPGaussianPath currently supports 1D endpoints only.")
        grid = np.linspace(0.0, 1.0, self.n_steps + 1)
        rhs = make_double_well_gaussian_bvp_rhs(self.potential)
        states = []
        for start_t, end_t in zip(x0_cpu, x1_cpu):
            start = float(start_t[0])
            end = float(end_t[0])
            mu_guess = (1.0 - grid) * start + grid * end
            mu_prime_guess = np.full_like(grid, end - start)
            sigma_guess = np.full_like(grid, self.sigma)
            sigma_prime_guess = np.zeros_like(grid)
            guess = np.vstack([mu_guess, mu_prime_guess, sigma_guess, sigma_prime_guess])

            def bc(ya, yb):
                return np.asarray([ya[0] - start, ya[2] - self.sigma, yb[0] - end, yb[2] - self.sigma])

            result = scipy_solve_bvp(
                rhs,
                bc,
                grid,
                guess,
                tol=self.tol,
                max_nodes=self.max_nodes,
                to_tensor=False,
            )
            if not result.success:
                raise RuntimeError(f"SciPy parametric BVP failed: {result.message}")
            states.append(torch.as_tensor(result.raw.sol(grid).T, dtype=x0_cpu.dtype))
        states_t = torch.stack(states, dim=0)
        self._store_cache(x0.detach().cpu(), x1.detach().cpu(), states_t)
        return states_t

    def compute(self, x0: Tensor, x1: Tensor, t: Tensor, return_derivatives: bool = True):
        states = self._interpolate_states(self._lookup_states(x0, x1), t)
        mu = states[:, 0:1].reshape_as(x0)
        sigma_t = states[:, 2:3].abs()
        if not return_derivatives:
            return mu, sigma_t
        mu_prime = states[:, 1:2].reshape_as(x0)
        sigma_t_prime = states[:, 3:4].abs()
        return mu, mu_prime, sigma_t, sigma_t_prime
