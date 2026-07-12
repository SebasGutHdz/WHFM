"""Toy dataset generators for Hamiltonian flow matching."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, pi, sin, sqrt
from typing import Callable, Optional, Union

import torch
from sklearn import datasets as sklearn_datasets
from torch import Tensor


Key = Optional[Union[int, Tensor, torch.Generator]]
_MAX_SKLEARN_SEED = 2**31 - 1


@dataclass
class DatasetBundle:
    sample_mu: Callable
    sample_nu: Callable
    sample_interior: Callable


def _make_generator(key: Key) -> Optional[torch.Generator]:
    if key is None or isinstance(key, torch.Generator):
        return key
    if torch.is_tensor(key):
        key = int(key.detach().cpu().item())
    generator = torch.Generator()
    generator.manual_seed(int(key))
    return generator


def _sklearn_seed(key: Key) -> Optional[int]:
    if key is None:
        return None
    if isinstance(key, torch.Generator):
        return int(torch.randint(0, _MAX_SKLEARN_SEED, (), generator=key).item())
    if torch.is_tensor(key):
        key = int(key.detach().cpu().item())
    return int(key) % _MAX_SKLEARN_SEED


def _sample_sphere(n_samples: int, dim: int, radius: float, generator=None) -> Tensor:
    x = torch.randn((n_samples, dim), generator=generator)
    return float(radius) * x / (x.norm(dim=-1, keepdim=True) + 1e-8)


def _sample_beta(n_samples: int, a: float, b: float, generator=None) -> Tensor:
    alpha = torch.full((n_samples,), float(a))
    beta = torch.full((n_samples,), float(b))
    x = torch._standard_gamma(alpha, generator=generator)
    y = torch._standard_gamma(beta, generator=generator)
    return x / (x + y).clamp_min(1e-8)


def randnsphere(key: Key = None, dim: int = 3, radius: float = 1.0) -> Tensor:
    """Uniform sample on a sphere surface in R^dim with the given radius."""

    return _sample_sphere(1, dim=dim, radius=radius, generator=_make_generator(key))[0]


def generate_concentric_spheres(
    key: Key = None,
    n_samples: int = 100,
    noise: float = 1e-4,
    dim: int = 3,
    inner_radius: float = 0.5,
    outer_radius: float = 1.0,
):
    generator = _make_generator(key)
    n_inner = n_samples // 2
    n_outer = n_samples - n_inner

    inner = _sample_sphere(n_inner, dim, inner_radius, generator)
    outer = _sample_sphere(n_outer, dim, outer_radius, generator)
    x = torch.cat([inner, outer], dim=0)

    if noise > 0.0:
        x = x + float(noise) * torch.randn(x.shape, generator=generator)

    y = torch.cat(
        [
            torch.ones(n_inner, dtype=torch.long),
            torch.zeros(n_outer, dtype=torch.long),
        ]
    )
    return x, y


def generate_moons(key: Key = None, n_samples: int = 100, noise: float = 1e-4, **kwargs):
    """Generate moons with sklearn and return torch tensors."""

    x_np, y_np = sklearn_datasets.make_moons(
        n_samples=n_samples,
        noise=float(noise),
        random_state=_sklearn_seed(key),
    )
    x = torch.as_tensor(x_np, dtype=torch.float32)
    y = torch.as_tensor(y_np, dtype=torch.long)

    scale = float(kwargs.get("scale", kwargs.get("scaling_factor", 1.0)))
    if kwargs.get("standardize", False):
        x = (x - x.mean()) / (x.std(unbiased=False) + 1e-8)
        x = x * scale
    else:
        x = x * scale
        x[:, 0] = x[:, 0] + float(kwargs.get("x_shift", 0.0))

    return x, y


def generate_scurve(
    key: Key = None,
    n_samples: int = 100,
    noise: float = 0.05,
    scale: float = 1.5,
    **kwargs,
):
    """Generate a 2D S-curve by projecting sklearn's 3D S-curve to (x, z)."""

    del kwargs
    x3d, t = sklearn_datasets.make_s_curve(
        n_samples=n_samples,
        noise=float(noise),
        random_state=_sklearn_seed(key),
    )
    x = torch.as_tensor(x3d[:, [0, 2]], dtype=torch.float32) * float(scale)
    y = torch.as_tensor(t, dtype=torch.float32)
    return x, y


def generate_spirals(key: Key = None, n_samples: int = 100, noise: float = 1e-4, **kwargs):
    del kwargs
    if n_samples <= 0:
        return torch.empty((0, 2)), torch.empty((0,), dtype=torch.long)

    generator = _make_generator(key)
    n_half = max(1, n_samples // 2)
    n = torch.sqrt(torch.rand((n_half, 1), generator=generator)) * 780.0 * (2.0 * pi) / 360.0
    d1x = -torch.cos(n) * n + float(noise) * torch.rand((n_half, 1), generator=generator)
    d1y = torch.sin(n) * n + float(noise) * torch.rand((n_half, 1), generator=generator)

    arm1 = torch.cat([d1x, d1y], dim=1)
    arm2 = -arm1
    x = torch.cat([arm1, arm2, arm1[:1]], dim=0)[:n_samples]
    y = torch.cat(
        [
            torch.zeros(n_half, dtype=torch.long),
            torch.ones(n_half, dtype=torch.long),
            torch.zeros(1, dtype=torch.long),
        ]
    )[:n_samples]
    return x, y


def generate_checkers(
    key: Key = None,
    n_samples: int = 100,
    n_tiles: int = 4,
    cell_size: float = 2.0,
    noise: float = 0.0,
):
    """Sample a 2D checkerboard distribution on alternating occupied tiles."""

    if n_tiles <= 0:
        raise ValueError("n_tiles must be positive")
    if cell_size <= 0.0:
        raise ValueError("cell_size must be positive")

    generator = _make_generator(key)
    rows, cols = torch.meshgrid(torch.arange(n_tiles), torch.arange(n_tiles), indexing="ij")
    active = ((rows + cols) % 2) == 0
    active_rows = rows[active]
    active_cols = cols[active]

    comp = torch.randint(active_rows.numel(), (n_samples,), generator=generator)
    offsets = cell_size * torch.rand((n_samples, 2), generator=generator)
    origin = -0.5 * n_tiles * cell_size
    x = torch.stack(
        [
            origin + active_cols[comp].float() * cell_size,
            origin + active_rows[comp].float() * cell_size,
        ],
        dim=1,
    )
    x = x + offsets

    if noise > 0.0:
        x = x + float(noise) * torch.randn(x.shape, generator=generator)

    return x, comp.long()


def generate_gaussians(
    key: Key = None,
    n_samples: int = 100,
    n_gaussians: int = 7,
    dim: int = 2,
    radius: float = 0.5,
    std_gaussians: float = 0.1,
    noise: float = 1e-3,
    post_scale: float = 1.0,
):
    generator = _make_generator(key)
    comp = torch.randint(n_gaussians, (n_samples,), generator=generator)
    angles = 2.0 * pi * comp.float() / float(n_gaussians)
    centers = torch.stack([radius * torch.cos(angles), radius * torch.sin(angles)], dim=1)

    if dim > 2:
        centers = torch.cat([centers, torch.zeros(n_samples, dim - 2)], dim=1)
    else:
        centers = centers[:, :dim]

    x = centers + float(std_gaussians) * torch.randn((n_samples, dim), generator=generator)
    if noise > 0.0:
        x = x + float(noise) * torch.randn(x.shape, generator=generator)
    if post_scale != 1.0:
        x = x * float(post_scale)
    return x, comp.long()


def generate_2dgaussian(
    key: Key = None,
    n_samples: int = 100,
    mean: Optional[Tensor] = None,
    std: float = 1.0,
    noise: float = 0.0,
):
    del noise
    generator = _make_generator(key)
    mean = torch.zeros(2) if mean is None else torch.as_tensor(mean, dtype=torch.float32)
    x = mean + float(std) * torch.randn((n_samples, 2), generator=generator)
    return x, torch.zeros_like(x)


def generate_diagonal_gaussian(
    key: Key = None,
    n_samples: int = 100,
    mean: Optional[Tensor] = None,
    std: float = 1.0,
    dim: Optional[int] = None,
    noise: float = 0.0,
):
    """Sample a diagonal Gaussian in any dimension."""

    generator = _make_generator(key)
    if mean is None:
        if dim is None:
            raise ValueError("diagonal_gaussian requires either mean or dim.")
        mean_t = torch.zeros(int(dim), dtype=torch.float32)
    else:
        mean_t = torch.as_tensor(mean, dtype=torch.float32).reshape(-1)
        if dim is not None and int(dim) != mean_t.numel():
            raise ValueError("diagonal_gaussian dim must match mean length.")
    x = mean_t + float(std) * torch.randn((n_samples, mean_t.numel()), generator=generator)
    if noise > 0.0:
        x = x + float(noise) * torch.randn(x.shape, generator=generator)
    return x, torch.zeros(n_samples, dtype=torch.long)


def generate_gaussian_mixture(
    key: Key = None,
    n_samples: int = 100,
    means=None,
    std: float = 1.0,
    weights=None,
    noise: float = 0.0,
):
    """Sample a Gaussian mixture with scalar component standard deviation."""

    if means is None:
        raise ValueError("gaussian_mixture requires means.")
    generator = _make_generator(key)
    means_t = torch.as_tensor(means, dtype=torch.float32)
    if means_t.ndim == 1:
        means_t = means_t[:, None]
    if means_t.ndim != 2:
        raise ValueError("gaussian_mixture means must have shape (components, dim).")
    if weights is None:
        weights_t = torch.ones(means_t.shape[0], dtype=torch.float32)
    else:
        weights_t = torch.as_tensor(weights, dtype=torch.float32).reshape(-1)
        if weights_t.numel() != means_t.shape[0]:
            raise ValueError("gaussian_mixture weights must match number of means.")
    comp = torch.multinomial(weights_t / weights_t.sum(), n_samples, replacement=True, generator=generator)
    x = means_t[comp] + float(std) * torch.randn((n_samples, means_t.shape[1]), generator=generator)
    if noise > 0.0:
        x = x + float(noise) * torch.randn(x.shape, generator=generator)
    return x, comp.long()


def _rotated_positions(positions: Tensor, theta: float) -> Tensor:
    rotation = torch.tensor(
        [[cos(float(theta)), -sin(float(theta))], [sin(float(theta)), cos(float(theta))]],
        dtype=positions.dtype,
    )
    return positions @ rotation.T


def generate_coulomb_roots(
    key: Key = None,
    n_samples: int = 100,
    endpoint: str = "source",
    n_particles: int = 6,
    particle_dim: int = 2,
    radius: float = 2.0,
    theta: float = pi / 3.0,
    std: float = 0.1,
    noise: float = 0.0,
):
    """Endpoint Gaussian from the Coulomb roots bridge notebook."""

    if int(particle_dim) != 2:
        raise ValueError("coulomb_roots currently supports particle_dim=2.")
    if endpoint not in {"source", "target"}:
        raise ValueError("coulomb_roots endpoint must be 'source' or 'target'.")
    generator = _make_generator(key)
    angles = 2.0 * pi * torch.arange(int(n_particles), dtype=torch.float32) / float(n_particles)
    source_positions = float(radius) * torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
    positions = source_positions if endpoint == "source" else _rotated_positions(source_positions, theta)
    mean = positions.reshape(-1)
    x = mean + float(std) * torch.randn((n_samples, mean.numel()), generator=generator)
    if noise > 0.0:
        x = x + float(noise) * torch.randn(x.shape, generator=generator)
    return x, torch.zeros(n_samples, dtype=torch.long)


def generate_fixed_three_body(
    key: Key = None,
    n_samples: int = 100,
    endpoint: str = "source",
    r: float = 5.0,
    alpha: float = pi / 2.0,
    std: float = 0.5,
    noise: float = 0.0,
):
    """Endpoint Gaussian from the fixed-center three-body bridge notebook."""

    if endpoint not in {"source", "target"}:
        raise ValueError("fixed_three_body endpoint must be 'source' or 'target'.")
    generator = _make_generator(key)
    source_positions = torch.tensor([[-2.0 * float(r), 0.0], [-float(r), 0.0]], dtype=torch.float32)
    positions = source_positions if endpoint == "source" else _rotated_positions(source_positions, alpha)
    mean = positions.reshape(-1)
    x = mean + float(std) * torch.randn((n_samples, mean.numel()), generator=generator)
    if noise > 0.0:
        x = x + float(noise) * torch.randn(x.shape, generator=generator)
    return x, torch.zeros(n_samples, dtype=torch.long)


def generate_grid_spring(
    key: Key = None,
    n_samples: int = 100,
    endpoint: str = "source",
    grid_side: int = 4,
    particle_dim: int = 2,
    grid_spacing: float = 3.0,
    bend_x_coeff: float = 0.6,
    bend_y_coeff: float = 0.3,
    covariance: float = 0.1,
    std: Optional[float] = None,
    noise: float = 0.0,
):
    """Endpoint Gaussian from the open-boundary grid spring notebook."""

    if int(particle_dim) != 2:
        raise ValueError("grid_spring currently supports particle_dim=2.")
    if endpoint not in {"source", "target"}:
        raise ValueError("grid_spring endpoint must be 'source' or 'target'.")
    generator = _make_generator(key)
    side = int(grid_side)
    ys, xs = torch.meshgrid(torch.arange(side, dtype=torch.float32), torch.arange(side, dtype=torch.float32), indexing="ij")
    source_positions = float(grid_spacing) * torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=1)
    grid_extent = float(grid_spacing) * float(side - 1)
    xc = 1.5 * source_positions[:, 0] / grid_extent - 0.5
    yc = 1.5 * source_positions[:, 1] / grid_extent - 0.5
    target_positions = torch.stack(
        [
            source_positions[:, 0] + float(bend_x_coeff) * yc.pow(2),
            source_positions[:, 1] + float(bend_y_coeff) * xc.pow(2),
        ],
        dim=1,
    )
    positions = source_positions if endpoint == "source" else target_positions
    mean = positions.reshape(-1)
    sample_std = sqrt(float(covariance)) if std is None else float(std)
    x = mean + sample_std * torch.randn((n_samples, mean.numel()), generator=generator)
    if noise > 0.0:
        x = x + float(noise) * torch.randn(x.shape, generator=generator)
    return x, torch.zeros(n_samples, dtype=torch.long)


def generate_gaussians_spiral(
    key: Key = None,
    n_samples: int = 100,
    n_gaussians: int = 7,
    n_gaussians_per_loop: int = 4,
    dim: int = 2,
    radius_start: float = 1.0,
    radius_end: float = 0.2,
    std_gaussians_start: float = 0.3,
    std_gaussians_end: float = 0.1,
    noise: float = 1e-3,
):
    generator = _make_generator(key)
    comp = torch.randint(n_gaussians, (n_samples,), generator=generator)
    angles = 2.0 * pi * comp.float() / float(n_gaussians_per_loop)
    radii = torch.linspace(radius_start, radius_end, n_gaussians)[comp]
    stds = torch.linspace(std_gaussians_start, std_gaussians_end, n_gaussians)[comp]
    centers = torch.stack([radii * torch.cos(angles), radii * torch.sin(angles)], dim=1)

    if dim > 2:
        centers = torch.cat([centers, torch.zeros(n_samples, dim - 2)], dim=1)
    else:
        centers = centers[:, :dim]

    x = centers + stds[:, None] * torch.randn((n_samples, dim), generator=generator)
    if noise > 0.0:
        x = x + float(noise) * torch.randn(x.shape, generator=generator)
    return x, comp.long()


_GENERATORS = {
    "randnsphere": None,
    "concentric_spheres": generate_concentric_spheres,
    "checkers": generate_checkers,
    "moons": generate_moons,
    "scurve": generate_scurve,
    "spirals": generate_spirals,
    "gaussians": generate_gaussians,
    "gaussians_spiral": generate_gaussians_spiral,
    "2d_gaussian": generate_2dgaussian,
    "diagonal_gaussian": generate_diagonal_gaussian,
    "gaussian_mixture": generate_gaussian_mixture,
    "coulomb_roots": generate_coulomb_roots,
    "fixed_three_body": generate_fixed_three_body,
    "grid_spring": generate_grid_spring,
}

_VALID_OTCFM_TAGS = {
    "gaussian_to_moons",
    "gaussian_to_8gaussians",
    "moons_to_8gaussians",
}


def _canonical_otcfm_tag(tag):
    if tag is None:
        return None
    tag = str(tag)
    if tag == "moons_to_8guassians":
        return "moons_to_8gaussians"
    if tag not in _VALID_OTCFM_TAGS:
        raise ValueError(
            f"Unknown otcfm_tag: {tag}. Expected one of {sorted(_VALID_OTCFM_TAGS)} or None."
        )
    return tag


def _default_kwargs_for_none(name: str):
    if name == "gaussians":
        return {"dim": 2, "radius": 5.0, "std_gaussians": 0.5, "n_gaussians": 8}
    return {}


def _otcfm_tag_defaults(name: str, tag: Optional[str]):
    if tag is None:
        return {}
    if name == "moons":
        if tag == "gaussian_to_moons":
            return {"standardize": False, "scale": 2.0, "x_shift": -1.0, "noise": 0.05}
        if tag == "moons_to_8gaussians":
            return {"standardize": True, "scale": 7.0, "noise": 0.1}
    if name == "gaussians":
        if tag == "gaussian_to_8gaussians":
            return {"dim": 2, "radius": 5.0, "std_gaussians": 1.0, "n_gaussians": 8}
        if tag == "moons_to_8gaussians":
            return {
                "dim": 2,
                "radius": 4.0,
                "std_gaussians": 0.5,
                "n_gaussians": 8,
                "post_scale": 3.0,
            }
    return {}


def _resolve_dataset_kwargs(name: str, ds_kwargs: Optional[dict]):
    resolved = _default_kwargs_for_none(name)
    explicit = {} if ds_kwargs is None else dict(ds_kwargs)
    resolved.update(_otcfm_tag_defaults(name, _canonical_otcfm_tag(explicit.pop("otcfm_tag", None))))
    resolved.update(explicit)
    return resolved


def _dataset_samples(name: str, key: Key, n_samples: int, ds_kwargs: Optional[dict]):
    kwargs = _resolve_dataset_kwargs(name, ds_kwargs)
    if name == "randnsphere":
        dim = int(kwargs.get("dim", 3))
        radius = float(kwargs.get("radius", 1.0))
        x = _sample_sphere(n_samples, dim, radius, _make_generator(key))
        return x, torch.zeros(n_samples, dtype=torch.long)
    if name not in _GENERATORS:
        raise ValueError(f"Unknown dataset name: {name}")
    return _GENERATORS[name](key, n_samples=n_samples, **kwargs)


def get_sampler_functions(
    mu_name: str,
    nu_name: str,
    mu_kwargs: Optional[dict] = None,
    nu_kwargs: Optional[dict] = None,
):
    def sample_mu(key: Key, n: int):
        return _dataset_samples(mu_name, key, n, mu_kwargs)[0]

    def sample_nu(key: Key, n: int):
        return _dataset_samples(nu_name, key, n, nu_kwargs)[0]

    return sample_mu, sample_nu


def _make_sample_interior(sample_mu, t_uniform_mix_prob, t_beta_a, t_beta_b):
    def sample_interior(key: Key, n: int, mixture: bool = False):
        generator = _make_generator(key)
        z = sample_mu(generator, n)
        t_uniform = torch.rand(n, generator=generator)

        if not mixture:
            return z, t_uniform

        use_uniform = torch.rand(n, generator=generator) < float(t_uniform_mix_prob)
        t_beta = _sample_beta(n, t_beta_a, t_beta_b, generator=generator)
        return z, torch.where(use_uniform, t_uniform, t_beta)

    return sample_interior


def make_dataset(
    mu_name: str,
    nu_name: str,
    mu_kwargs: Optional[dict] = None,
    nu_kwargs: Optional[dict] = None,
    mu_sample_noise: float = 0.0,
    nu_sample_noise: float = 0.0,
    t_uniform_mix_prob: float = 2.0 / 3.0,
    t_beta_a: float = 0.5,
    t_beta_b: float = 0.5,
):
    sample_mu_raw, sample_nu_raw = get_sampler_functions(mu_name, nu_name, mu_kwargs, nu_kwargs)

    def sample_mu(key: Key, n: int):
        generator = _make_generator(key)
        x = sample_mu_raw(generator, n)
        if mu_sample_noise > 0.0:
            x = x + float(mu_sample_noise) * torch.randn(x.shape, generator=generator)
        return x

    def sample_nu(key: Key, n: int):
        generator = _make_generator(key)
        x = sample_nu_raw(generator, n)
        if nu_sample_noise > 0.0:
            x = x + float(nu_sample_noise) * torch.randn(x.shape, generator=generator)
        return x

    return DatasetBundle(
        sample_mu=sample_mu,
        sample_nu=sample_nu,
        sample_interior=_make_sample_interior(sample_mu, t_uniform_mix_prob, t_beta_a, t_beta_b),
    )
