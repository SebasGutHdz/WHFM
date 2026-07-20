"""ODE/BVP solver helpers for Hamiltonian flow matching."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

import numpy as np
import torch
from torch import Tensor


def euler_step(f: Callable, t, y: Tensor, dt) -> Tensor:
    """One explicit Euler step."""

    return y + dt * f(t, y)


def euler_integrate(f: Callable, y0: Tensor, t_grid: Tensor) -> Tensor:
    """Integrate an ODE on ``t_grid`` with explicit Euler."""

    states = [y0]
    y = y0
    for i in range(t_grid.numel() - 1):
        t = t_grid[i]
        dt = t_grid[i + 1] - t
        y = euler_step(f, t, y, dt)
        states.append(y)
    return torch.stack(states, dim=0)


def semi_implicit_euler_step(
    force: Callable,
    t,
    position: Tensor,
    velocity: Tensor,
    dt,
) -> Tuple[Tensor, Tensor]:
    """One semi-implicit Euler step for ``x' = v, v' = force(t, x)``."""

    velocity_next = velocity + dt * force(t, position)
    position_next = position + dt * velocity_next
    return position_next, velocity_next


def semi_implicit_euler_integrate(
    force: Callable,
    position0: Tensor,
    velocity0: Tensor,
    t_grid: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Integrate a second-order separable system with semi-implicit Euler."""

    positions = [position0]
    velocities = [velocity0]
    position = position0
    velocity = velocity0
    for i in range(t_grid.numel() - 1):
        t = t_grid[i]
        dt = t_grid[i + 1] - t
        position, velocity = semi_implicit_euler_step(force, t, position, velocity, dt)
        positions.append(position)
        velocities.append(velocity)
    return torch.stack(positions, dim=0), torch.stack(velocities, dim=0)


def rk4_step(f: Callable, t, y: Tensor, dt) -> Tensor:
    """One fourth-order Runge-Kutta step."""

    k1 = f(t, y)
    k2 = f(t + dt / 2, y + dt * k1 / 2)
    k3 = f(t + dt / 2, y + dt * k2 / 2)
    k4 = f(t + dt, y + dt * k3)
    return y + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def rk4_integrate(f: Callable, y0: Tensor, t_grid: Tensor) -> Tensor:
    """Integrate an ODE on ``t_grid`` with RK4."""

    states = [y0]
    y = y0
    for i in range(t_grid.numel() - 1):
        t = t_grid[i]
        dt = t_grid[i + 1] - t
        y = rk4_step(f, t, y, dt)
        states.append(y)
    return torch.stack(states, dim=0)


@dataclass
class ScipySolverResult:
    """Small wrapper that keeps SciPy's raw result and tensor-converted arrays."""

    raw: Any
    t: Any
    y: Any
    _like: Optional[Tensor] = None

    @property
    def success(self) -> bool:
        return bool(getattr(self.raw, "success", False))

    @property
    def message(self) -> str:
        return str(getattr(self.raw, "message", ""))

    def evaluate(self, t):
        if getattr(self.raw, "sol", None) is None:
            raise ValueError("SciPy result has no dense solution; pass dense_output=True.")
        value = self.raw.sol(_to_numpy(t))
        if self._like is None:
            return value
        return _to_tensor_like(value, self._like)

    def __getattr__(self, name: str):
        return getattr(self.raw, name)


def _to_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _to_tensor_like(value, like: Tensor) -> Tensor:
    return torch.as_tensor(value, dtype=like.dtype, device=like.device)


def scipy_solve_ivp(
    fun: Callable,
    t_span,
    y0,
    *,
    t_eval=None,
    dense_output: bool = True,
    to_tensor: bool = True,
    **kwargs,
) -> ScipySolverResult:
    """Run ``scipy.integrate.solve_ivp`` and convert arrays back to tensors when possible."""

    from scipy.integrate import solve_ivp

    like = y0 if torch.is_tensor(y0) and to_tensor else None
    result = solve_ivp(
        fun,
        tuple(_to_numpy(t_span).tolist()),
        _to_numpy(y0).reshape(-1),
        t_eval=None if t_eval is None else _to_numpy(t_eval).reshape(-1),
        dense_output=dense_output,
        **kwargs,
    )
    t = _to_tensor_like(result.t, like) if like is not None else result.t
    y = _to_tensor_like(result.y, like) if like is not None else result.y
    return ScipySolverResult(raw=result, t=t, y=y, _like=like)


def scipy_solve_bvp(
    fun: Callable,
    bc: Callable,
    x,
    y,
    *,
    to_tensor: bool = True,
    **kwargs,
) -> ScipySolverResult:
    """Run ``scipy.integrate.solve_bvp`` and convert arrays back to tensors when possible."""

    from scipy.integrate import solve_bvp

    like = y if torch.is_tensor(y) and to_tensor else None
    result = solve_bvp(fun, bc, _to_numpy(x).reshape(-1), _to_numpy(y), **kwargs)
    t = _to_tensor_like(result.x, like) if like is not None else result.x
    y_value = _to_tensor_like(result.y, like) if like is not None else result.y
    return ScipySolverResult(raw=result, t=t, y=y_value, _like=like)


def make_particle_bvp_rhs(potential) -> Callable:
    """Return a SciPy-compatible RHS for ``x' = v, v' = -grad V(x)``."""

    gradient = getattr(potential, "linear_gradient", None)
    if gradient is None:
        gradient = potential.gradient

    def rhs(_t, state):
        dim = state.shape[0] // 2
        x = torch.as_tensor(state[:dim].T, dtype=torch.float64)
        grad = gradient(x).detach().cpu().numpy().T
        return np.vstack([state[dim:], -grad])

    return rhs


def make_double_well_gaussian_bvp_rhs(potential) -> Callable:
    """Return the Gaussian-parametric BVP RHS used by ``double_well1d.ipynb``."""

    if not hasattr(potential, "alpha") or not hasattr(potential, "beta"):
        raise TypeError("Gaussian-parametric BVP currently requires a DoubleWellPotential.")

    alpha = float(potential.alpha)
    beta = float(potential.beta)

    def rhs(_t, state):
        mu, mu_prime, sigma, sigma_prime = state
        return np.vstack(
            [
                mu_prime,
                -alpha * mu**3 - mu * (beta + alpha * 3.0 * sigma**2),
                sigma_prime,
                -3.0 * alpha * sigma**3 - sigma * (beta + alpha * 3.0 * mu**2),
            ]
        )

    return rhs
