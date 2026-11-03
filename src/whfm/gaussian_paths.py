"""Gaussian probability paths for Hamiltonian flow matching."""

from __future__ import annotations

from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
import multiprocessing
import os
from typing import Dict, Tuple
import pickle
import time

import numpy as np
import torch
from torch import Tensor

from .solvers import make_particle_bvp_rhs, scipy_solve_bvp


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


def _mean_std_bvp_worker_init():
    try:
        torch.set_num_threads(1)
    except RuntimeError:
        pass



@contextmanager
def _multiprocessing_worker_environment():
    key = "MKL_THREADING_LAYER"
    previous = os.environ.get(key)
    changed = previous is None or previous.upper() == "INTEL"
    if changed:
        os.environ[key] = "GNU"
    try:
        yield
    finally:
        if changed:
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous


def _safe_multiprocessing_context():
    for method in ("forkserver", "spawn"):
        try:
            return multiprocessing.get_context(method)
        except ValueError:
            continue
    raise RuntimeError("No safe multiprocessing start method is available.")


def _interpolate_reference_samples_numpy(
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


def _make_mean_std_bvp_rhs(
    potential,
    dim: int,
    eps: np.ndarray,
    weights: np.ndarray,
    *,
    interaction_potential=None,
    interaction_coefficient: float = 1.0,
    reference_t_grid=None,
    reference_samples=None,
    reference_means=None,
    reference_stds=None,
):
    n_quadrature = eps.shape[0]
    has_interaction = interaction_potential is not None
    has_internal = bool(getattr(potential, "has_internal", False))
    if has_interaction and (reference_t_grid is None or reference_samples is None):
        raise RuntimeError(
            "Interaction RHS requires frozen reference samples. Call batch_solve "
            "or pass reference_t_grid/reference_samples to _make_rhs."
        )
    if has_internal and (reference_t_grid is None or reference_means is None or reference_stds is None):
        raise RuntimeError(
            "Internal entropy RHS requires frozen reference Gaussian mixture parameters. "
            "Call batch_solve or pass reference_t_grid/reference_means/reference_stds to _make_rhs."
        )
    internal_gradient = None
    if has_internal:
        internal_gradient = getattr(potential, "internal_gradient_from_gaussian_mixture", None)
        if internal_gradient is None:
            raise TypeError(
                "Configured internal BVP forces require "
                "internal_gradient_from_gaussian_mixture(x, means, stds)."
            )
    linear_gradient = getattr(potential, "linear_gradient", None)
    if linear_gradient is None:
        linear_gradient = potential.gradient

    def rhs(_t, state):
        n_mesh = state.shape[1]
        mu = state[:dim].T
        mu_dot = state[dim : 2 * dim]
        sigma = state[2 * dim]
        sigma_dot = state[2 * dim + 1]
        # SciPy Newton iterations can temporarily propose negative sigma
        # values even when the converged path is positive.
        x_quad = mu[:, None, :] + sigma[:, None, None] * eps[None, :, :]
        x_quad_t = torch.as_tensor(x_quad.reshape(-1, dim), dtype=torch.float64)
        grad = linear_gradient(x_quad_t).detach().cpu().numpy().reshape(n_mesh, n_quadrature, dim)

        potential_mu_gradient = np.sum(weights[None, :, None] * grad, axis=1)
        potential_sigma_gradient = np.sum(
            weights[None, :] * np.sum(grad * eps[None, :, :], axis=-1),
            axis=1,
        )

        if has_internal:
            mu_ref = _interpolate_reference_samples_numpy(
                _t, n_mesh, reference_t_grid, reference_means
            )
            sigma_ref = _interpolate_reference_samples_numpy(
                _t, n_mesh, reference_t_grid, reference_stds
            )
            internal_x_t = torch.as_tensor(
                np.array(x_quad, copy=True), dtype=torch.float64
            )
            internal_means_t = torch.as_tensor(
                np.array(mu_ref, copy=True), dtype=torch.float64
            )
            internal_stds_t = torch.as_tensor(
                np.array(sigma_ref, copy=True), dtype=torch.float64
            )
            internal_grad = (
                internal_gradient(internal_x_t, internal_means_t, internal_stds_t)
                .detach()
                .cpu()
                .numpy()
                .reshape(n_mesh, n_quadrature, dim)
            )
            internal_mu_gradient = np.sum(
                weights[None, :, None] * internal_grad,
                axis=1,
            )
            internal_sigma_gradient = np.sum(
                weights[None, :] * np.sum(internal_grad * eps[None, :, :], axis=-1),
                axis=1,
            )
            potential_mu_gradient = potential_mu_gradient + internal_mu_gradient
            potential_sigma_gradient = potential_sigma_gradient + internal_sigma_gradient

        if has_interaction:
            # The reference particles are frozen for the duration of this BVP
            # solve. Average grad_x W first over that empirical density and then
            # over X = mu + q Z. With coefficient c this produces
            #   mu second derivative = -c E[grad_x W(X, Y)],
            #   sigma second derivative = -(c / dim) E[grad_x W(X, Y) . Z]
            # ConfiguredPotential already includes c in its
            # interaction gradient; the explicit coefficient remains for a
            # directly supplied raw interaction potential.
            y_ref = _interpolate_reference_samples_numpy(
                _t, n_mesh, reference_t_grid, reference_samples
            )
            interaction_x = np.broadcast_to(
                x_quad[:, :, None, :],
                (n_mesh, n_quadrature, y_ref.shape[1], dim),
            )
            interaction_y = np.broadcast_to(
                y_ref[:, None, :, :],
                (n_mesh, n_quadrature, y_ref.shape[1], dim),
            )
            interaction_x_t = torch.as_tensor(
                np.array(interaction_x.reshape(-1, dim), copy=True), dtype=torch.float64
            )
            interaction_y_t = torch.as_tensor(
                np.array(interaction_y.reshape(-1, dim), copy=True), dtype=torch.float64
            )
            interaction_grad = (
                interaction_potential.interaction_gradient(interaction_x_t, interaction_y_t)
                .detach()
                .cpu()
                .numpy()
                .reshape(n_mesh, n_quadrature, y_ref.shape[1], dim)
            )
            interaction_mean_grad = interaction_grad.mean(axis=2)
            interaction_mu_gradient = np.sum(
                weights[None, :, None] * interaction_mean_grad,
                axis=1,
            )
            interaction_sigma_gradient = np.sum(
                weights[None, :]
                * np.sum(interaction_mean_grad * eps[None, :, :], axis=-1),
                axis=1,
            )
            potential_mu_gradient = (
                potential_mu_gradient + interaction_coefficient * interaction_mu_gradient
            )
            potential_sigma_gradient = (
                potential_sigma_gradient + interaction_coefficient * interaction_sigma_gradient
            )

        mean_accel = -potential_mu_gradient
        sigma_accel = -potential_sigma_gradient / dim
        return np.vstack(
            [mu_dot, mean_accel.T, sigma_dot.reshape(1, -1), sigma_accel.reshape(1, -1)]
        )

    return rhs


def _solve_mean_std_bvp_pair(job):
    dim = int(job["dim"])
    start = job["start"]
    end = job["end"]
    sigma_source = float(job["sigma_source"])
    sigma_target = float(job["sigma_target"])
    grid = job["grid"]
    guess = job["guess"]

    def bc(ya, yb):
        return np.concatenate(
            [
                ya[:dim] - start,
                yb[:dim] - end,
                np.asarray([
                    ya[2 * dim] - sigma_source,
                    yb[2 * dim] - sigma_target,
                ]),
            ]
        )

    rhs = _make_mean_std_bvp_rhs(
        job["potential"], dim, job["eps"], job["weights"],
        interaction_potential=job["interaction_potential"],
        interaction_coefficient=job["interaction_coefficient"],
        reference_t_grid=job["reference_t_grid"],
        reference_samples=job["reference_samples"],
        reference_means=job["reference_means"],
        reference_stds=job["reference_stds"],
    )

    solve_start = time.perf_counter()
    try:
        result = scipy_solve_bvp(
            rhs,
            bc,
            grid,
            guess,
            tol=job["tol"],
            max_nodes=job["max_nodes"],
            to_tensor=False,
        )
        solve_time = time.perf_counter() - solve_start
        iterations = getattr(result.raw, "niter", None)
        if iterations is not None:
            iterations = int(iterations)
        mesh_nodes = int(np.asarray(getattr(result.raw, "x", grid)).size)
        if not result.success:
            return {
                "index": job["index"],
                "state": None,
                "failure_message": result.message,
                "solve_time": solve_time,
                "iterations": iterations,
                "mesh_nodes": mesh_nodes,
            }
        state = result.raw.sol(grid).T
        sigma_path = state[:, 2 * dim]
        zero_terminal = sigma_target == 0.0
        boundary_tolerance = max(1e-10, min(float(job["tol"]), 1e-6))
        invalid_sigma = (
            np.any(sigma_path[:-1] <= 0)
            or (zero_terminal and abs(float(sigma_path[-1])) > boundary_tolerance)
            or (not zero_terminal and np.any(sigma_path <= 0))
        )
        if invalid_sigma:
            return {
                "index": job["index"],
                "state": None,
                "failure_message": "SciPy mean/std BVP returned a nonpositive sigma path.",
                "solve_time": solve_time,
                "iterations": iterations,
                "mesh_nodes": mesh_nodes,
            }
        if zero_terminal:
            # Canonicalize only after validating the raw boundary residual and
            # every preterminal scale; this is boundary enforcement, not a clamp.
            state[-1, 2 * dim] = 0.0
        return {
            "index": job["index"],
            "state": state,
            "failure_message": None,
            "solve_time": solve_time,
            "iterations": iterations,
            "mesh_nodes": mesh_nodes,
        }
    except Exception as exc:
        return {
            "index": job["index"],
            "state": None,
            "failure_message": f"{type(exc).__name__}: {exc}",
            "solve_time": time.perf_counter() - solve_start,
            "iterations": None,
            "mesh_nodes": None,
        }


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
        sigma: float | None = 1e-3,
        sigma_source: float | None = None,
        sigma_target: float | None = None,
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
        entropy_density_std_floor: float | None = None,
        entropy_reference_max_iterations: int = 1,
        entropy_reference_tolerance: float = 1e-3,
        entropy_reference_relaxation: float = 0.5,
        use_monte_carlo: bool = False,
        monte_carlo_samples: int = 100,
        num_workers: int = 1,
        failure_policy: str = "skip_pair",
    ):
        if (sigma_source is None) != (sigma_target is None):
            raise ValueError("sigma_source and sigma_target must be provided together.")
        if sigma_source is None:
            if sigma is None:
                raise ValueError("sigma or both sigma_source and sigma_target must be provided.")
            sigma_value = float(sigma)
            sigma_source = sigma_value
            sigma_target = sigma_value
        else:
            sigma_value = None if sigma is None else float(sigma)
            sigma_source = float(sigma_source)
            sigma_target = float(sigma_target)
        if sigma_source <= 0:
            raise ValueError("sigma_source must be positive.")
        if sigma_target < 0:
            raise ValueError("sigma_target must be nonnegative.")
        if quadrature_order < 1:
            raise ValueError("quadrature_order must be positive.")
        if n_density_samples < 1:
            raise ValueError("n_density_samples must be positive.")
        if n_reference_grid is not None and n_reference_grid < 2:
            raise ValueError("n_reference_grid must be at least 2.")
        if entropy_density_std_floor is not None and entropy_density_std_floor <= 0.0:
            raise ValueError("entropy_density_std_floor must be positive when provided.")
        if entropy_reference_max_iterations < 1:
            raise ValueError("entropy_reference_max_iterations must be positive.")
        if entropy_reference_tolerance <= 0.0:
            raise ValueError("entropy_reference_tolerance must be positive.")
        if not 0.0 < entropy_reference_relaxation <= 1.0:
            raise ValueError("entropy_reference_relaxation must be in (0, 1].")
        if monte_carlo_samples < 1:
            raise ValueError("monte_carlo_samples must be positive.")
        if num_workers < 0:
            raise ValueError("num_workers must be nonnegative.")
        if failure_policy not in {"skip_pair", "raise"}:
            raise ValueError("failure_policy must be \"skip_pair\" or \"raise\".")
        super().__init__(n_steps=n_steps, tol=tol, max_nodes=max_nodes)
        self.potential = potential
        self.sigma = sigma_value
        self.sigma_source = sigma_source
        self.sigma_target = sigma_target
        self.quadrature_order = int(quadrature_order)
        self.n_density_samples = int(n_density_samples)
        self.n_reference_grid = n_reference_grid
        self.entropy_density_std_floor = (
            None if entropy_density_std_floor is None else float(entropy_density_std_floor)
        )
        self.entropy_reference_max_iterations = int(entropy_reference_max_iterations)
        self.entropy_reference_tolerance = float(entropy_reference_tolerance)
        self.entropy_reference_relaxation = float(entropy_reference_relaxation)
        self.use_monte_carlo = bool(use_monte_carlo)
        self.monte_carlo_samples = int(monte_carlo_samples)
        self.num_workers = int(num_workers)
        self.failure_policy = failure_policy
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
        self.has_internal_force = bool(getattr(potential, "has_internal", False))
        self.reference_t_grid = None
        self.reference_samples = None
        self.reference_means = None
        self.reference_stds = None
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

    def _prepare_scalar_guess(
        self,
        value,
        name: str,
        n_pairs: int,
        positive: bool = False,
        allow_zero_terminal: bool = False,
    ):
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
        if positive:
            invalid = (
                np.any(array[:, :-1] <= 0) or np.any(array[:, -1] < 0)
                if allow_zero_terminal
                else np.any(array <= 0)
            )
            if invalid:
                requirement = (
                    "positive values except a nonnegative terminal"
                    if allow_zero_terminal
                    else "only positive values"
                )
                raise ValueError(f"{name} must contain {requirement}.")
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
        *,
        build_samples: bool = True,
        reference_states: np.ndarray | None = None,
        sample_indices: np.ndarray | None = None,
    ):
        """Freeze an empirical reference density for one batch of BVP solves.

        Reference components come from cached solved paths when available and
        otherwise from the supplied/default initializer. A single normal draw
        per selected component is carried across the reference time grid. This
        is a frozen Monte Carlo approximation, not a density solved
        self-consistently alongside each pair's BVP.
        """
        n_pairs, dim = x0_flat.shape
        solve_grid = np.linspace(0.0, 1.0, self.n_steps + 1)
        n_reference_grid = self.n_steps + 1 if self.n_reference_grid is None else self.n_reference_grid
        reference_t_grid = np.linspace(0.0, 1.0, n_reference_grid)

        if reference_states is None:
            built_states = []
            for i in range(n_pairs):
                cached = self._cached_state_for_pair(x0_flat[i], x1_flat[i])
                if cached is None:
                    cached = self._state_from_guesses(
                        mu_guess[i],
                        mu_dot_guess[i],
                        sigma_guess[i],
                        sigma_dot_guess[i],
                    )
                built_states.append(cached)
            reference_states = np.stack(built_states, axis=0)
        else:
            reference_states = np.asarray(reference_states, dtype=float)
            expected = (n_pairs, self.n_steps + 1, 2 * dim + 2)
            if reference_states.shape != expected:
                raise ValueError(
                    f"reference_states must have shape {expected}; "
                    f"got {reference_states.shape}."
                )

        if sample_indices is None:
            sample_indices = (
                torch.randint(n_pairs, (self.n_density_samples,))
                .detach()
                .cpu()
                .numpy()
            )
        else:
            sample_indices = np.asarray(sample_indices, dtype=int).reshape(-1)
            if sample_indices.size == 0:
                raise ValueError("sample_indices must contain at least one component.")
            if np.any(sample_indices < 0) or np.any(sample_indices >= n_pairs):
                raise ValueError("sample_indices contains an out-of-range component index.")

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
        invalid_reference_scale = (
            np.any(sigma_ref[:-1] <= 0) or np.any(sigma_ref[-1] < 0)
            if self.sigma_target == 0.0
            else np.any(sigma_ref <= 0)
        )
        if invalid_reference_scale:
            raise RuntimeError("Reference Gaussian mixture contains a nonpositive sigma value.")
        density_sigma_ref = sigma_ref
        if self.entropy_density_std_floor is not None:
            density_sigma_ref = np.maximum(sigma_ref, self.entropy_density_std_floor)
        sample_noise = None
        reference_samples = None
        if build_samples:
            sample_noise = torch.randn(sample_indices.size, dim).detach().cpu().numpy()
            reference_samples = mu_ref + sigma_ref * sample_noise[None, :, :]

        self.reference_t_grid = reference_t_grid
        self.reference_samples = reference_samples
        self.reference_means = mu_ref
        self.reference_stds = density_sigma_ref
        self.reference_indices = sample_indices
        self.reference_noise = sample_noise
        self.reference_states = reference_states
        return reference_t_grid, reference_samples, mu_ref, density_sigma_ref

    def _reference_path_delta(
        self,
        previous_states: np.ndarray,
        solved_states: np.ndarray,
        dim: int,
    ) -> float:
        """Return the maximum normalized RMS mean/std path change."""

        solve_grid = np.linspace(0.0, 1.0, self.n_steps + 1)
        reference_grid = (
            solve_grid
            if self.n_reference_grid is None
            else np.linspace(0.0, 1.0, self.n_reference_grid)
        )
        previous = np.stack(
            [self._interpolate_state_grid(state, solve_grid, reference_grid) for state in previous_states]
        )
        solved = np.stack(
            [self._interpolate_state_grid(state, solve_grid, reference_grid) for state in solved_states]
        )
        previous_mean = previous[..., :dim]
        solved_mean = solved[..., :dim]
        previous_q = previous[..., 2 * dim : 2 * dim + 1]
        solved_q = solved[..., 2 * dim : 2 * dim + 1]
        if self.entropy_density_std_floor is not None:
            previous_q = np.maximum(previous_q, self.entropy_density_std_floor)
            solved_q = np.maximum(solved_q, self.entropy_density_std_floor)

        def normalized_rms(new, old, denominator_floor):
            change = float(np.sqrt(np.mean(np.square(new - old))))
            scale = max(float(np.sqrt(np.mean(np.square(old)))), denominator_floor)
            return change / scale

        mean_delta = normalized_rms(solved_mean, previous_mean, 1.0)
        std_delta = normalized_rms(solved_q, previous_q, 1e-12)
        return max(mean_delta, std_delta)

    def _make_rhs(
        self,
        dim: int,
        reference_t_grid=None,
        reference_samples=None,
        reference_means=None,
        reference_stds=None,
    ):
        eps, weights = self._normal_integration_rule(dim)
        has_interaction = self.interaction_potential is not None
        if has_interaction and (reference_t_grid is None or reference_samples is None):
            reference_t_grid = self.reference_t_grid
            reference_samples = self.reference_samples
        if self.has_internal_force and (reference_t_grid is None or reference_means is None or reference_stds is None):
            reference_t_grid = self.reference_t_grid
            reference_means = self.reference_means
            reference_stds = self.reference_stds
        return _make_mean_std_bvp_rhs(
            self.potential,
            dim,
            eps,
            weights,
            interaction_potential=self.interaction_potential,
            interaction_coefficient=self.interaction_coefficient,
            reference_t_grid=reference_t_grid,
            reference_samples=reference_samples,
            reference_means=reference_means,
            reference_stds=reference_stds,
        )

    def _build_pair_solve_jobs(
        self,
        x0_flat: Tensor,
        x1_flat: Tensor,
        grid: np.ndarray,
        solve_mu_guess: np.ndarray,
        solve_mu_dot_guess: np.ndarray,
        solve_sigma_guess: np.ndarray,
        solve_sigma_dot_guess: np.ndarray,
        reference_t_grid,
        reference_samples,
        reference_means,
        reference_stds,
    ):
        _, dim = x0_flat.shape
        eps, weights = self._normal_integration_rule(dim)
        common = {
            "dim": dim,
            "grid": grid,
            "sigma_source": self.sigma_source,
            "sigma_target": self.sigma_target,
            "tol": self.tol,
            "max_nodes": self.max_nodes,
            "potential": self.potential,
            "eps": eps,
            "weights": weights,
            "interaction_potential": self.interaction_potential,
            "interaction_coefficient": self.interaction_coefficient,
            "reference_t_grid": reference_t_grid,
            "reference_samples": reference_samples,
            "reference_means": reference_means,
            "reference_stds": reference_stds,
        }
        jobs = []
        for i, (start_t, end_t) in enumerate(zip(x0_flat, x1_flat)):
            guess = np.vstack(
                [
                    solve_mu_guess[i].T,
                    solve_mu_dot_guess[i].T,
                    solve_sigma_guess[i].reshape(1, -1),
                    solve_sigma_dot_guess[i].reshape(1, -1),
                ]
            )
            job = dict(common)
            job.update(
                {
                    "index": i,
                    "start": start_t.numpy(),
                    "end": end_t.numpy(),
                    "guess": guess,
                }
            )
            jobs.append(job)
        return jobs

    def _solve_pair_jobs(self, jobs):
        if self.num_workers <= 1 or len(jobs) <= 1:
            return [_solve_mean_std_bvp_pair(job) for job in jobs]

        try:
            pickle.dumps((self.potential, self.interaction_potential))
        except Exception as exc:
            raise ValueError(
                "Parallel SciPy BVP solves require picklable potential objects; "
                "set bridge_solver.num_workers: 1 to use the serial solver."
            ) from exc

        max_workers = min(self.num_workers, len(jobs))
        context = _safe_multiprocessing_context()
        start_method = context.get_start_method()
        try:
            with _multiprocessing_worker_environment():
                with ProcessPoolExecutor(
                    max_workers=max_workers,
                    initializer=_mean_std_bvp_worker_init,
                    mp_context=context,
                ) as executor:
                    return list(executor.map(_solve_mean_std_bvp_pair, jobs))
        except Exception as exc:
            raise ValueError(
                "Parallel SciPy BVP solve failed with "
                f"multiprocessing start method {start_method!r}: {exc}. "
                "Set bridge_solver.num_workers: 1 to use the serial solver."
            ) from exc

    def batch_solve(self, x0: Tensor, x1: Tensor) -> Tensor:
        """Solve a batch, optionally iterating entropy references to a fixed point."""

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
            self.sigma_guess, "sigma_guess", n_pairs, positive=True,
            allow_zero_terminal=self.sigma_target == 0.0,
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
        default_sigma_line = np.linspace(self.sigma_source, self.sigma_target, grid.size)
        default_sigma = np.broadcast_to(
            default_sigma_line[None, :], (n_pairs, grid.size)
        ).copy()
        default_sigma_dot = np.full(
            (n_pairs, grid.size), self.sigma_target - self.sigma_source
        )

        solve_mu_guess = default_mu if mu_guess is None else mu_guess
        solve_mu_dot_guess = default_mu_dot if mu_dot_guess is None else mu_dot_guess
        solve_sigma_guess = default_sigma if sigma_guess is None else sigma_guess
        solve_sigma_dot_guess = default_sigma_dot if sigma_dot_guess is None else sigma_dot_guess
        guess_states = np.concatenate(
            [
                solve_mu_guess,
                solve_mu_dot_guess,
                solve_sigma_guess[..., None],
                solve_sigma_dot_guess[..., None],
            ],
            axis=-1,
        )
        reference_states = guess_states.copy()

        needs_reference_density = (
            self.interaction_potential is not None or self.has_internal_force
        )
        if not needs_reference_density:
            sampled_global_indices = np.empty((0,), dtype=int)
        elif self.n_density_samples == n_pairs:
            sampled_global_indices = np.arange(n_pairs, dtype=int)
        else:
            sampled_global_indices = (
                torch.randint(n_pairs, (self.n_density_samples,))
                .detach()
                .cpu()
                .numpy()
            )

        active_global = np.arange(n_pairs, dtype=int)
        iteration_limit = (
            self.entropy_reference_max_iterations if self.has_internal_force else 1
        )
        reference_delta_history = []
        reference_converged = None if not self.has_internal_force else False
        component_fallbacks = 0
        failure_messages = {}
        pair_solve_times = [0.0] * n_pairs
        pair_solver_iterations = [None] * n_pairs
        pair_solver_mesh_nodes = [None] * n_pairs
        final_states = []
        final_success_indices = []
        batch_solve_start = time.perf_counter()

        for reference_iteration in range(iteration_limit):
            global_to_local = {
                int(global_index): local_index
                for local_index, global_index in enumerate(active_global)
            }
            if needs_reference_density:
                selected_global = np.asarray(
                    [
                        index
                        for index in sampled_global_indices
                        if int(index) in global_to_local
                    ],
                    dtype=int,
                )
                if selected_global.size == 0:
                    selected_global = active_global.copy()
                    component_fallbacks += 1
                selected_local = np.asarray(
                    [global_to_local[int(index)] for index in selected_global],
                    dtype=int,
                )
            else:
                selected_local = np.empty((0,), dtype=int)

            active_index_t = torch.as_tensor(active_global, dtype=torch.long)
            active_x0 = x0_flat.index_select(0, active_index_t)
            active_x1 = x1_flat.index_select(0, active_index_t)
            active_guess_states = guess_states[active_global]
            active_reference_states = reference_states[active_global]
            if needs_reference_density:
                reference_t_grid, reference_samples, reference_means, reference_stds = (
                    self._build_reference_samples(
                        active_x0,
                        active_x1,
                        active_guess_states[..., :dim],
                        active_guess_states[..., dim : 2 * dim],
                        active_guess_states[..., 2 * dim],
                        active_guess_states[..., 2 * dim + 1],
                        build_samples=self.interaction_potential is not None,
                        reference_states=active_reference_states,
                        sample_indices=selected_local,
                    )
                )
            else:
                reference_t_grid = None
                reference_samples = None
                reference_means = None
                reference_stds = None
            jobs = self._build_pair_solve_jobs(
                active_x0,
                active_x1,
                grid,
                active_guess_states[..., :dim],
                active_guess_states[..., dim : 2 * dim],
                active_guess_states[..., 2 * dim],
                active_guess_states[..., 2 * dim + 1],
                reference_t_grid,
                reference_samples,
                reference_means,
                reference_stds,
            )
            pair_results = sorted(
                self._solve_pair_jobs(jobs), key=lambda item: item["index"]
            )

            solved_states = []
            successful_global = []
            for pair_result in pair_results:
                local_index = int(pair_result["index"])
                global_index = int(active_global[local_index])
                pair_solve_times[global_index] += float(pair_result["solve_time"])
                iterations = pair_result["iterations"]
                if iterations is not None:
                    previous_iterations = pair_solver_iterations[global_index]
                    pair_solver_iterations[global_index] = (
                        (0 if previous_iterations is None else previous_iterations)
                        + int(iterations)
                    )
                pair_solver_mesh_nodes[global_index] = pair_result["mesh_nodes"]
                failure_message = pair_result["failure_message"]
                if failure_message is not None:
                    if iteration_limit == 1:
                        failure_messages[global_index] = failure_message
                    else:
                        failure_messages[global_index] = (
                            f"entropy reference iteration {reference_iteration + 1}: "
                            f"{failure_message}"
                        )
                    continue
                solved_states.append(np.asarray(pair_result["state"], dtype=float))
                successful_global.append(global_index)

            if self.failure_policy == "raise" and failure_messages:
                failure_metadata = {
                    "total_solve_time_seconds": time.perf_counter() - batch_solve_start,
                    "pair_solve_time_seconds": pair_solve_times,
                    "pair_solver_iterations": pair_solver_iterations,
                    "pair_solver_mesh_nodes": pair_solver_mesh_nodes,
                    "entropy_reference_iterations": (
                        reference_iteration + 1 if self.has_internal_force else 0
                    ),
                    "entropy_reference_converged": reference_converged,
                    "entropy_reference_final_delta": (
                        reference_delta_history[-1]
                        if reference_delta_history
                        else None
                    ),
                    "entropy_reference_delta_history": reference_delta_history,
                    "entropy_reference_component_fallbacks": component_fallbacks,
                    "entropy_reference_component_indices": (
                        sampled_global_indices.tolist()
                    ),
                }
                self._record_solve_metadata(
                    n_pairs, successful_global, failure_messages, failure_metadata
                )
                preview = list(failure_messages.items())[:5]
                raise RuntimeError(
                    "Bridge solve failed during entropy reference iteration "
                    f"{reference_iteration + 1}: {preview}"
                )

            if not solved_states:
                failure_metadata = {
                    "total_solve_time_seconds": time.perf_counter() - batch_solve_start,
                    "pair_solve_time_seconds": pair_solve_times,
                    "pair_solver_iterations": pair_solver_iterations,
                    "pair_solver_mesh_nodes": pair_solver_mesh_nodes,
                    "entropy_reference_iterations": (
                        reference_iteration + 1 if self.has_internal_force else 0
                    ),
                    "entropy_reference_converged": reference_converged,
                    "entropy_reference_final_delta": (
                        reference_delta_history[-1]
                        if reference_delta_history
                        else None
                    ),
                    "entropy_reference_delta_history": reference_delta_history,
                    "entropy_reference_component_fallbacks": component_fallbacks,
                    "entropy_reference_component_indices": (
                        sampled_global_indices.tolist()
                    ),
                }
                self._record_solve_metadata(
                    n_pairs, [], failure_messages, failure_metadata
                )
                self.failure_messages = failure_messages
                preview = "; ".join(
                    f"{index}: {message}"
                    for index, message in list(failure_messages.items())[:5]
                )
                suffix = f" First failures: {preview}" if preview else ""
                raise RuntimeError(
                    "SciPy mean/std BVP failed: all active pairs failed during "
                    f"entropy reference iteration {reference_iteration + 1}."
                    f"{suffix}"
                )

            solved_array = np.stack(solved_states, axis=0)
            successful_global_array = np.asarray(successful_global, dtype=int)
            previous_reference = reference_states[successful_global_array]
            if self.has_internal_force:
                delta = self._reference_path_delta(
                    previous_reference, solved_array, dim
                )
                reference_delta_history.append(delta)
                reference_converged = delta <= self.entropy_reference_tolerance

            final_states = [
                torch.as_tensor(state, dtype=x0_cpu.dtype) for state in solved_array
            ]
            final_success_indices = successful_global
            active_global = successful_global_array
            guess_states[active_global] = solved_array

            if not self.has_internal_force or reference_converged:
                break
            if reference_iteration + 1 >= iteration_limit:
                break
            relaxation = self.entropy_reference_relaxation
            reference_states[active_global] = (
                (1.0 - relaxation) * previous_reference
                + relaxation * solved_array
            )

        solve_metadata = {
            "total_solve_time_seconds": time.perf_counter() - batch_solve_start,
            "pair_solve_time_seconds": pair_solve_times,
            "pair_solver_iterations": pair_solver_iterations,
            "pair_solver_mesh_nodes": pair_solver_mesh_nodes,
            "entropy_reference_iterations": len(reference_delta_history),
            "entropy_reference_converged": reference_converged,
            "entropy_reference_final_delta": (
                reference_delta_history[-1] if reference_delta_history else None
            ),
            "entropy_reference_delta_history": reference_delta_history,
            "entropy_reference_component_fallbacks": component_fallbacks,
            "entropy_reference_component_indices": sampled_global_indices.tolist(),
        }
        return self._store_successful_cache(
            x0_cpu,
            x1_cpu,
            final_states,
            final_success_indices,
            failure_messages,
            "SciPy mean/std BVP failed",
            solve_metadata=solve_metadata,
        )

    def sample_xt(self, x0: Tensor, x1: Tensor, t: Tensor, epsilon: Tensor) -> Tensor:
        mu, sigma = self.compute(x0, x1, t, return_derivatives=False)
        return mu + _sigma_like_x(sigma, x0) * epsilon

    def compute_ut(self, x0: Tensor, x1: Tensor, t: Tensor, xt: Tensor) -> Tensor:
        mu, mu_dot, sigma, sigma_dot = self.compute(
            x0, x1, t, return_derivatives=True
        )
        if torch.any(sigma <= 0):
            raise RuntimeError(
                "Conditional velocity is undefined at the zero-scale terminal endpoint."
            )
        coefficient = sigma_dot / sigma
        return mu_dot + _sigma_like_x(coefficient, xt) * (xt - mu)

    def compute(self, x0: Tensor, x1: Tensor, t: Tensor, return_derivatives: bool = True):
        states = self._interpolate_states(self._lookup_states(x0, x1), t)
        dim = x0.reshape(x0.shape[0], -1).shape[1]
        mu = states[:, :dim].reshape_as(x0)
        sigma_t = states[:, 2 * dim : 2 * dim + 1]
        if torch.any(sigma_t < 0):
            raise RuntimeError("Cached mean/std BVP contains a negative sigma value.")
        zero_scale = sigma_t == 0
        if torch.any(zero_scale):
            terminal_time = torch.isclose(
                _time_column(t, x0), torch.ones((), device=x0.device, dtype=x0.dtype)
            )
            if self.sigma_target != 0.0 or torch.any(zero_scale & ~terminal_time):
                raise RuntimeError(
                    "Cached mean/std BVP contains a zero scale before the terminal endpoint."
                )
        if not return_derivatives:
            return mu, sigma_t
        mu_prime = states[:, dim : 2 * dim].reshape_as(x0)
        sigma_t_prime = states[:, 2 * dim + 1 : 2 * dim + 2]
        return mu, mu_prime, sigma_t, sigma_t_prime
