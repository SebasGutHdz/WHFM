"""Pointwise linear potentials used by Hamiltonian flow matching notebooks."""

from __future__ import annotations

from abc import abstractmethod
from typing import Callable, Dict, Mapping, Optional, Sequence

import torch
from torch import Tensor
from torch.nn import functional as F

from .potentials import Potential


class LinearPotential(Potential):
    """Potential of the form ``int V(x) rho(x) dx``."""

    @abstractmethod
    def energy(self, x: Tensor) -> Tensor:
        """Return pointwise potential energy with shape ``x.shape[:-1]``."""

    @abstractmethod
    def gradient(self, x: Tensor) -> Tensor:
        """Return ``grad_x V(x)`` with the same shape as ``x``."""


def node_index(x: int, y: int, side: int) -> int:
    return y * side + x


def von_neumann_edges(side: int):
    edges = []
    for y in range(side):
        for x in range(side):
            i = node_index(x, y, side)
            if x + 1 < side:
                edges.append((i, node_index(x + 1, y, side)))
            if y + 1 < side:
                edges.append((i, node_index(x, y + 1, side)))
    return edges


def von_neumann_grid_laplacian(side: int, *, device=None, dtype=None) -> Tensor:
    n = side**2
    lap = torch.zeros((n, n), device=device, dtype=dtype)
    for i, j in von_neumann_edges(side):
        lap[i, i] += 1.0
        lap[j, j] += 1.0
        lap[i, j] -= 1.0
        lap[j, i] -= 1.0
    return lap



def _as_like(value, x: Tensor) -> Tensor:
    return torch.as_tensor(value, dtype=x.dtype, device=x.device)


def harmonic_v(x: Tensor, U: Tensor) -> Tensor:
    """Return ``0.5 * x^T U x`` for a symmetric matrix ``U``."""

    U = _as_like(U, x)
    return 0.5 * (torch.matmul(x, U.T) * x).sum(dim=-1)


def harmonic_grad(x: Tensor, U: Tensor) -> Tensor:
    """Return the gradient of ``0.5 * x^T U x``."""

    U = _as_like(U, x)
    symmetric_U = 0.5 * (U + U.T)
    return torch.matmul(x, symmetric_U.T)


def double_well_v(x: Tensor, alpha: float = 0.15, beta: float = -3.0) -> Tensor:
    """Return ``sum_i alpha*x_i^4/4 + beta*x_i^2/2``."""

    alpha_t = _as_like(alpha, x)
    beta_t = _as_like(beta, x)
    return (alpha_t * x.pow(4) / 4.0 + beta_t * x.pow(2) / 2.0).sum(dim=-1)


def double_well_grad(x: Tensor, alpha: float = 0.15, beta: float = -3.0) -> Tensor:
    """Return the gradient of the double-well potential."""

    alpha_t = _as_like(alpha, x)
    beta_t = _as_like(beta, x)
    return alpha_t * x.pow(3) + beta_t * x


def hill_v(x: Tensor, alpha: float = 3.0) -> Tensor:
    """Return the hill potential ``-alpha * ||x||^2 / 2``."""

    alpha_t = _as_like(alpha, x)
    return -0.5 * alpha_t * x.pow(2).sum(dim=-1)


def hill_grad(x: Tensor, alpha: float = 3.0) -> Tensor:
    """Return the gradient of the hill potential."""

    alpha_t = _as_like(alpha, x)
    return -alpha_t * x


def obstacle_v(
    x: Tensor,
    alpha: float = 5.0,
    beta: float = 20.0,
    centers: Optional[Tensor] = None,
    radius_squared: float = 1.0,
) -> Tensor:
    """Return summed smooth Gaussian obstacle energies.

    The default centers match the two shifted obstacles used in
    ``examples_HF/obstacle_pot_2.ipynb``.
    """

    if centers is None:
        centers = torch.tensor([[0.0, -1.25], [0.0, 1.25]], dtype=x.dtype, device=x.device)
    else:
        centers = _as_like(centers, x)

    alpha_t = _as_like(alpha, x)
    beta_t = _as_like(beta, x)
    radius_t = _as_like(radius_squared, x)
    y = x.unsqueeze(-2) - centers
    r = y.pow(2).sum(dim=-1)
    width_t = torch.clamp_min(radius_t, torch.finfo(x.dtype).eps)
    profile = torch.exp(-0.5 * alpha_t * r / width_t)
    return (-beta_t * profile).sum(dim=-1)


def obstacle_grad(
    x: Tensor,
    alpha: float = 5.0,
    beta: float = 20.0,
    centers: Optional[Tensor] = None,
    radius_squared: float = 1.0,
) -> Tensor:
    """Return the gradient of the summed smooth Gaussian obstacle potential."""

    if centers is None:
        centers = torch.tensor([[0.0, -1.25], [0.0, 1.25]], dtype=x.dtype, device=x.device)
    else:
        centers = _as_like(centers, x)

    alpha_t = _as_like(alpha, x)
    beta_t = _as_like(beta, x)
    radius_t = _as_like(radius_squared, x)
    y = x.unsqueeze(-2) - centers
    r = y.pow(2).sum(dim=-1, keepdim=True)
    width_t = torch.clamp_min(radius_t, torch.finfo(x.dtype).eps)
    profile = torch.exp(-0.5 * alpha_t * r / width_t)
    scale = beta_t * alpha_t * profile / width_t
    return (scale * y).sum(dim=-2)


def _require_2d(x: Tensor, name: str) -> None:
    if x.shape[-1] != 2:
        raise ValueError(f"{name} expects points with last dimension 2.")


def _autograd_gradient(energy_fn: Callable[[Tensor], Tensor], x: Tensor) -> Tensor:
    with torch.enable_grad():
        x_req = x if x.requires_grad else x.detach().requires_grad_(True)
        energy = energy_fn(x_req)
        (grad,) = torch.autograd.grad(
            energy.sum(),
            x_req,
            create_graph=x.requires_grad,
            retain_graph=x.requires_grad,
        )
    return grad


def crowd_nav_obstacle_cfg_drunken_spider():
    xys = [[-7.0, 0.5], [-7.0, -7.5]]
    widths = [14.0, 14.0]
    heights = [7.0, 7.0]
    return xys, widths, heights


def crowd_nav_obstacle_drunken_spider_v(x: Tensor) -> Tensor:
    """Return the drunken-spider rectangular obstacle cost."""

    _require_2d(x, "drunken_spider")
    xys, widths, heights = crowd_nav_obstacle_cfg_drunken_spider()
    xys_t = _as_like(xys, x)
    widths_t = _as_like(widths, x)
    heights_t = _as_like(heights, x)

    x_coord = x[..., 0].unsqueeze(-1)
    y_coord = x[..., 1].unsqueeze(-1)
    x_lower = xys_t[:, 0]
    x_upper = x_lower + widths_t
    y_lower = xys_t[:, 1]
    y_upper = y_lower + heights_t

    a = -5.0 * (x_coord - x_lower) * (x_coord - x_upper)
    b = -5.0 * (y_coord - y_lower) * (y_coord - y_upper)
    cost = F.softplus(a, beta=20, threshold=1) * F.softplus(b, beta=20, threshold=1)
    return cost.sum(dim=-1)


def crowd_nav_obstacle_cfg_gmm():
    centers = [[6.0, 6.0], [6.0, -6.0], [-6.0, -6.0]]
    radius = 1.5
    return centers, radius


def crowd_nav_obstacle_gmm_v(x: Tensor) -> Tensor:
    """Return the three-bump GMM obstacle cost."""

    _require_2d(x, "gmm")
    centers, radius = crowd_nav_obstacle_cfg_gmm()
    centers_t = _as_like(centers, x)
    radius_t = _as_like(radius, x)
    dist = torch.linalg.vector_norm(x.unsqueeze(-2) - centers_t, dim=-1)
    return F.softplus(100.0 * (radius_t - dist), beta=1, threshold=20).sum(dim=-1)


def crowd_nav_obstacle_cfg_stunnel():
    a, b, c = 20.0, 1.0, 90.0
    centers = [[5.0, 6.0], [-5.0, -6.0]]
    return a, b, c, centers


def crowd_nav_obstacle_stunnel_v(x: Tensor) -> Tensor:
    """Return the soft tunnel obstacle cost."""

    _require_2d(x, "stunnel")
    a, b, c, centers = crowd_nav_obstacle_cfg_stunnel()
    a_t = _as_like(a, x)
    b_t = _as_like(b, x)
    c_t = _as_like(c, x)
    centers_t = _as_like(centers, x)

    diff = x.unsqueeze(-2) - centers_t
    d = a_t * diff[..., 0].pow(2) + b_t * diff[..., 1].pow(2)
    return F.softplus(c_t - d, beta=1, threshold=20).sum(dim=-1)


def crowd_nav_obstacle_cfg_vneck():
    c_sq = 0.36
    coef = 5.0
    return c_sq, coef


def crowd_nav_obstacle_vneck_v(x: Tensor) -> Tensor:
    """Return the soft v-neck obstacle cost."""

    _require_2d(x, "vneck")
    c_sq, coef = crowd_nav_obstacle_cfg_vneck()
    c_sq_t = _as_like(c_sq, x)
    coef_t = _as_like(coef, x)
    x_sq = x.pow(2)
    d = coef_t * x_sq[..., 0] - x_sq[..., 1]
    return F.softplus(-c_sq_t - d, beta=1, threshold=20)


_CROWD_NAV_OBSTACLE_COSTS: Dict[str, Callable[[Tensor], Tensor]] = {
    "gmm": crowd_nav_obstacle_gmm_v,
    "stunnel": crowd_nav_obstacle_stunnel_v,
    "vneck": crowd_nav_obstacle_vneck_v,
    "drunken_spider": crowd_nav_obstacle_drunken_spider_v,
}


def build_crowd_nav_obstacle_cost(name: str) -> Callable[[Tensor], Tensor]:
    """Return a named crowd-navigation obstacle cost function."""

    try:
        return _CROWD_NAV_OBSTACLE_COSTS[name]
    except KeyError as exc:
        options = ", ".join(sorted(_CROWD_NAV_OBSTACLE_COSTS))
        raise ValueError(f"Unknown crowd-navigation obstacle '{name}'. Options: {options}.") from exc


def crowd_nav_obstacle_v(x: Tensor, name: str) -> Tensor:
    """Return a named crowd-navigation obstacle cost."""

    return build_crowd_nav_obstacle_cost(name)(x)


def crowd_nav_obstacle_grad(x: Tensor, name: str) -> Tensor:
    """Return the gradient of a named crowd-navigation obstacle cost."""

    return _autograd_gradient(lambda y: crowd_nav_obstacle_v(y, name), x)


class HarmonicPotential(LinearPotential):
    """Linear potential with pointwise energy ``0.5 * x^T U x``."""

    def __init__(self, U: Tensor):
        self.U = torch.as_tensor(U, dtype=torch.get_default_dtype())

    def energy(self, x: Tensor) -> Tensor:
        return harmonic_v(x, self.U)

    def gradient(self, x: Tensor) -> Tensor:
        return harmonic_grad(x, self.U)


class DoubleWellPotential(LinearPotential):
    """Elementwise double-well potential."""

    def __init__(self, alpha: float = 0.15, beta: float = -3.0):
        self.alpha = alpha
        self.beta = beta

    def energy(self, x: Tensor) -> Tensor:
        return double_well_v(x, self.alpha, self.beta)

    def gradient(self, x: Tensor) -> Tensor:
        return double_well_grad(x, self.alpha, self.beta)


class DuffingDoubleWellPotential(LinearPotential):
    """Two-dimensional Duffing double-well potential from the notebook example."""

    def __init__(self, kappa: float = 5.0):
        self.kappa = float(kappa)

    def energy(self, x: Tensor) -> Tensor:
        if x.shape[-1] != 2:
            raise ValueError("duffing_double_well_2d expects points with last dimension 2.")
        q = x[..., 0]
        y = x[..., 1]
        return -0.5 * q.pow(2) + 0.25 * q.pow(4) + 0.5 * self.kappa * y.pow(2) + 0.25

    def gradient(self, x: Tensor) -> Tensor:
        if x.shape[-1] != 2:
            raise ValueError("duffing_double_well_2d expects points with last dimension 2.")
        q = x[..., 0]
        y = x[..., 1]
        return torch.stack([q.pow(3) - q, self.kappa * y], dim=-1)


class HillPotential(LinearPotential):
    """Linear hill potential ``-alpha * ||x||^2 / 2``."""

    def __init__(self, alpha: float = 3.0):
        self.alpha = alpha

    def energy(self, x: Tensor) -> Tensor:
        return hill_v(x, self.alpha)

    def gradient(self, x: Tensor) -> Tensor:
        return hill_grad(x, self.alpha)


class ObstaclePotential(LinearPotential):
    """Summed smooth Gaussian obstacle potential."""

    def __init__(
        self,
        alpha: float = 5.0,
        beta: float = 20.0,
        centers: Optional[Sequence[Sequence[float]]] = None,
        radius_squared: float = 1.0,
    ):
        self.alpha = alpha
        self.beta = beta
        self.centers = None if centers is None else torch.as_tensor(centers, dtype=torch.float32)
        self.radius_squared = radius_squared

    def energy(self, x: Tensor) -> Tensor:
        return obstacle_v(x, self.alpha, self.beta, self.centers, self.radius_squared)

    def gradient(self, x: Tensor) -> Tensor:
        return obstacle_grad(x, self.alpha, self.beta, self.centers, self.radius_squared)

class CrowdNavObstaclePotential(LinearPotential):
    """Named 2D obstacle potential from the crowd-navigation examples."""

    def __init__(self, name: str):
        self.name = name
        self._energy_fn = build_crowd_nav_obstacle_cost(name)

    def energy(self, x: Tensor) -> Tensor:
        return self._energy_fn(x)

    def gradient(self, x: Tensor) -> Tensor:
        return _autograd_gradient(self.energy, x)


class SmoothCoulombPotential(LinearPotential):
    def __init__(self, n_particles: int, charges, coulomb_constant: float = 1.0, epsilon: float = 0.15, particle_dim: int = 2):
        self.n_particles = int(n_particles)
        self.particle_dim = int(particle_dim)
        self.charges = torch.as_tensor(charges).reshape(self.n_particles)
        self.coulomb_constant = float(coulomb_constant)
        self.epsilon = float(epsilon)
        self.pairs = [(i, j) for i in range(self.n_particles) for j in range(i + 1, self.n_particles)]

    def _reshape(self, q: Tensor) -> Tensor:
        return q.reshape(q.shape[0], self.n_particles, self.particle_dim)

    def _charges_like(self, q_particles: Tensor) -> Tensor:
        return self.charges.to(device=q_particles.device, dtype=q_particles.dtype)

    def energy(self, q: Tensor) -> Tensor:
        q_particles = self._reshape(q)
        charges = self._charges_like(q_particles)
        energy = torch.zeros(q_particles.shape[0], device=q_particles.device, dtype=q_particles.dtype)
        for i, j in self.pairs:
            diff = q_particles[:, i] - q_particles[:, j]
            radius_sq = diff.pow(2).sum(dim=-1) + self.epsilon**2
            energy = energy + self.coulomb_constant * charges[i] * charges[j] * radius_sq.rsqrt()
        return energy

    def gradient(self, q: Tensor) -> Tensor:
        original_shape = q.shape
        q_particles = self._reshape(q)
        charges = self._charges_like(q_particles)
        grad = torch.zeros_like(q_particles)
        for i, j in self.pairs:
            diff = q_particles[:, i] - q_particles[:, j]
            radius_sq = diff.pow(2).sum(dim=-1, keepdim=True) + self.epsilon**2
            coeff = self.coulomb_constant * charges[i] * charges[j]
            pair_grad = -coeff * diff / radius_sq.pow(1.5)
            grad[:, i] = grad[:, i] + pair_grad
            grad[:, j] = grad[:, j] - pair_grad
        return grad.reshape(original_shape)

    def linear_gradient(self, q: Tensor) -> Tensor:
        return self.gradient(q)


class FixedCenterThreeBodyPotential(LinearPotential):
    def __init__(self, fixed_position, G: float = 1.0, central_mass: float = 1.0, moving_mass: float = 1.0, epsilon: float = 0.35, n_moving: int = 2, particle_dim: int = 2):
        self.fixed_position = torch.as_tensor(fixed_position).reshape(1, 1, particle_dim)
        self.G = float(G)
        self.central_mass = float(central_mass)
        self.moving_mass = float(moving_mass)
        self.epsilon = float(epsilon)
        self.n_moving = int(n_moving)
        self.particle_dim = int(particle_dim)

    def _reshape(self, q: Tensor) -> Tensor:
        return q.reshape(q.shape[0], self.n_moving, self.particle_dim)

    def _fixed_like(self, q_particles: Tensor) -> Tensor:
        return self.fixed_position.to(device=q_particles.device, dtype=q_particles.dtype)

    def _soft_inverse_distance(self, diff: Tensor) -> Tensor:
        radius_sq = diff.pow(2).sum(dim=-1) + self.epsilon**2
        return radius_sq.rsqrt()

    def energy(self, q: Tensor) -> Tensor:
        q_particles = self._reshape(q)
        center = self._fixed_like(q_particles)
        center_diff = q_particles - center
        pair_diff = q_particles[:, 0] - q_particles[:, 1]
        center_energy = -self.G * self.central_mass #* self.moving_mass
        center_energy = center_energy * self._soft_inverse_distance(center_diff).sum(dim=-1)
        pair_energy = -self.G * self.moving_mass#**2
        pair_energy = pair_energy * self._soft_inverse_distance(pair_diff)
        return center_energy + pair_energy

    def gradient(self, q: Tensor) -> Tensor:
        original_shape = q.shape
        q_particles = self._reshape(q)
        center = self._fixed_like(q_particles)
        grad = torch.zeros_like(q_particles)
        center_diff = q_particles - center
        center_radius_sq = center_diff.pow(2).sum(dim=-1, keepdim=True) + self.epsilon**2
        center_coeff = self.G * self.central_mass #* self.moving_mass
        grad = grad + center_coeff * center_diff / center_radius_sq.pow(1.5)
        pair_diff = q_particles[:, 0] - q_particles[:, 1]
        pair_radius_sq = pair_diff.pow(2).sum(dim=-1, keepdim=True) + self.epsilon**2
        pair_grad = self.G * self.moving_mass * pair_diff / pair_radius_sq.pow(1.5) #**2
        grad[:, 0] = grad[:, 0] + pair_grad
        grad[:, 1] = grad[:, 1] - pair_grad
        return grad.reshape(original_shape)

    def linear_gradient(self, q: Tensor) -> Tensor:
        return self.gradient(q)


class GridSpringPotential(LinearPotential):
    def __init__(self, grid_side: int, particle_dim: int = 2, kappa: float = 1.0):
        self.grid_side = int(grid_side)
        self.n_particles = self.grid_side**2
        self.particle_dim = int(particle_dim)
        self.kappa = float(kappa)
        self.edges = von_neumann_edges(self.grid_side)

    def _reshape(self, q: Tensor) -> Tensor:
        return q.reshape(q.shape[0], self.n_particles, self.particle_dim)

    def energy(self, q: Tensor) -> Tensor:
        q_particles = self._reshape(q)
        energy = torch.zeros(q.shape[0], device=q.device, dtype=q.dtype)
        for i, j in self.edges:
            diff = q_particles[:, j] - q_particles[:, i]
            energy = energy + 0.5 * self.kappa * diff.pow(2).sum(dim=1)
        return energy

    def gradient(self, q: Tensor) -> Tensor:
        original_shape = q.shape
        q_particles = self._reshape(q)
        grad = torch.zeros_like(q_particles)
        for i, j in self.edges:
            diff = q_particles[:, j] - q_particles[:, i]
            grad[:, i] -= self.kappa * diff
            grad[:, j] += self.kappa * diff
        return grad.reshape(original_shape)

    def linear_gradient(self, q: Tensor) -> Tensor:
        return self.gradient(q)


def _require_empty_parameters(name: str, parameters: Mapping[str, object]) -> None:
    if parameters:
        keys = ", ".join(sorted(parameters))
        raise ValueError(f"Linear potential '{name}' does not accept parameters: {keys}.")


def _build_coulomb_roots_potential(parameters: Mapping[str, object]) -> LinearPotential:
    values = {
        "n_particles": 6,
        "coulomb_constant": 2.0,
        "epsilon": 0.1,
        "particle_dim": 2,
    }
    values.update(parameters)
    if "charges" not in values:
        values["charges"] = torch.ones(int(values["n_particles"]))
    return SmoothCoulombPotential(**values)


def _build_fixed_three_body_potential(parameters: Mapping[str, object]) -> LinearPotential:
    values = {
        "fixed_position": [0.0, 0.0],
        "G": 1.0,
        "central_mass": 20.0,
        "moving_mass": 1.0,
        "epsilon": 0.01,
        "n_moving": 2,
        "particle_dim": 2,
    }
    values.update(parameters)
    return FixedCenterThreeBodyPotential(**values)


def _build_grid_spring_potential(parameters: Mapping[str, object]) -> LinearPotential:
    values = {"grid_side": 4, "particle_dim": 2, "kappa": 1.0}
    values.update(parameters)
    return GridSpringPotential(**values)


def _build_crowd_nav_potential(name: str, parameters: Mapping[str, object]) -> LinearPotential:
    _require_empty_parameters(name, parameters)
    return CrowdNavObstaclePotential(name)


def build_linear_potential(name: str, parameters: Optional[Mapping[str, object]] = None) -> LinearPotential:
    """Build a linear potential by name and optional constructor parameters."""

    parameters = {} if parameters is None else dict(parameters)
    builders: Dict[str, Callable[[Mapping[str, object]], LinearPotential]] = {
        "double_well": lambda params: DoubleWellPotential(**params),
        "coulomb_roots": _build_coulomb_roots_potential,
        "drunken_spider": lambda params: _build_crowd_nav_potential("drunken_spider", params),
        "duffing_double_well_2d": lambda params: DuffingDoubleWellPotential(**params),
        "fixed_three_body": _build_fixed_three_body_potential,
        "gmm": lambda params: _build_crowd_nav_potential("gmm", params),
        "grid_spring": _build_grid_spring_potential,
        "hill": lambda params: HillPotential(**params),
        "obstacle": lambda params: ObstaclePotential(**params),
        "stunnel": lambda params: _build_crowd_nav_potential("stunnel", params),
        "vneck": lambda params: _build_crowd_nav_potential("vneck", params),
    }
    try:
        return builders[name](parameters)
    except KeyError as exc:
        options = ", ".join(sorted(builders))
        raise ValueError(f"Unknown linear potential '{name}'. Options: {options}.") from exc
