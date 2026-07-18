from __future__ import annotations

import importlib
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EntropyPotential = importlib.import_module(
    "torchcfm.WHFM-standalone.Potentials.internal_potentials"
).EntropyPotential
ConfiguredPotential = importlib.import_module(
    "torchcfm.WHFM-standalone.Potentials.configured_potential"
).ConfiguredPotential
BridgeSolverConfig = importlib.import_module(
    "torchcfm.WHFM-standalone.config"
).BridgeSolverConfig
_gaussian_paths = importlib.import_module(
    "torchcfm.WHFM-standalone.gaussian_paths"
)
_make_mean_std_bvp_rhs = _gaussian_paths._make_mean_std_bvp_rhs
MeanStdBVPGaussianPath = _gaussian_paths.MeanStdBVPGaussianPath


def test_entropy_score_single_component_matches_gaussian_score():
    entropy = EntropyPotential()
    x = torch.tensor([[2.0, -1.0]], dtype=torch.float64)
    means = torch.tensor([[0.5, 3.0]], dtype=torch.float64)
    stds = torch.tensor([[2.0]], dtype=torch.float64)

    score = entropy.score_from_gaussian_mixture(x, means, stds)

    expected = (means - x) / stds.pow(2)
    assert torch.allclose(score, expected)


def test_entropy_score_symmetric_mixture_zero_at_midpoint():
    entropy = EntropyPotential()
    x = torch.tensor([[0.0]], dtype=torch.float64)
    means = torch.tensor([[-1.0], [1.0]], dtype=torch.float64)
    stds = torch.ones(2, 1, dtype=torch.float64)

    score = entropy.score_from_gaussian_mixture(x, means, stds)

    assert torch.allclose(score, torch.zeros_like(score), atol=1e-12)


def test_entropy_score_finite_for_small_sigma():
    entropy = EntropyPotential()
    x = torch.tensor([[1.0]], dtype=torch.float64)
    means = torch.tensor([[1.0]], dtype=torch.float64)
    stds = torch.tensor([[1e-10]], dtype=torch.float64)

    score = entropy.score_from_gaussian_mixture(x, means, stds)

    assert torch.isfinite(score).all()


def test_entropy_only_rhs_uses_configured_score_projection():
    potential = ConfiguredPotential({"linear": None, "internal": ("entropy", 2.0), "interaction": None})
    rhs = _make_mean_std_bvp_rhs(
        potential,
        dim=1,
        eps=np.array([[0.0]], dtype=float),
        weights=np.array([1.0], dtype=float),
        reference_t_grid=np.array([0.0, 1.0], dtype=float),
        reference_means=np.array([[[0.0]], [[0.0]]], dtype=float),
        reference_stds=np.array([[[1.0]], [[1.0]]], dtype=float),
    )
    state = np.array([[2.0], [0.0], [1.0], [0.0]], dtype=float)

    out = rhs(np.array([0.5]), state)

    assert np.allclose(out[:, 0], np.array([0.0, 4.0, 0.0, 0.0]))


def test_bridge_solver_density_fields_parse():
    cfg = BridgeSolverConfig.from_dict(
        {
            "kind": "scipy",
            "sigma": 1e-2,
            "bridge_steps": 3,
            "tol": 1e-2,
            "max_nodes": 20,
            "quadrature_order": 2,
            "n_density_samples": 7,
            "n_reference_grid": 5,
            "failure_policy": "skip_pair",
        }
    )

    assert cfg.n_density_samples == 7
    assert cfg.n_reference_grid == 5


def test_entropy_density_std_floor_applies_only_to_reference_stds():
    potential = ConfiguredPotential({"linear": None, "internal": ("entropy", 1.0), "interaction": None})
    path = MeanStdBVPGaussianPath(
        potential,
        sigma_source=1e-4,
        sigma_target=1e-3,
        n_steps=2,
        n_density_samples=3,
        entropy_density_std_floor=0.05,
    )
    x0 = torch.zeros(3, 1, dtype=torch.float64)
    x1 = torch.ones(3, 1, dtype=torch.float64)
    mu_guess = np.zeros((3, 3, 1), dtype=float)
    mu_dot_guess = np.zeros((3, 3, 1), dtype=float)
    sigma_guess = np.full((3, 3), 1e-4, dtype=float)
    sigma_dot_guess = np.zeros((3, 3), dtype=float)

    _, reference_samples, _, reference_stds = path._build_reference_samples(
        x0,
        x1,
        mu_guess,
        mu_dot_guess,
        sigma_guess,
        sigma_dot_guess,
        build_samples=True,
    )

    assert reference_samples is not None
    assert np.allclose(reference_stds, 0.05)
    assert np.allclose(path.reference_stds, 0.05)
    assert np.allclose(
        reference_samples,
        path.reference_means + 1e-4 * path.reference_noise[None, :, :],
    )


def test_negative_entropy_rhs_accelerates_toward_mixture_center():
    potential = ConfiguredPotential({"linear": None, "internal": ("entropy", -1.0), "interaction": None})
    rhs = _make_mean_std_bvp_rhs(
        potential,
        dim=1,
        eps=np.array([[0.0]], dtype=float),
        weights=np.array([1.0], dtype=float),
        reference_t_grid=np.array([0.0, 1.0], dtype=float),
        reference_means=np.array([[[-1.0], [1.0]], [[-1.0], [1.0]]], dtype=float),
        reference_stds=np.ones((2, 2, 1), dtype=float),
    )
    right_component_state = np.array([[1.0], [0.0], [1.0], [0.0]], dtype=float)

    out = rhs(np.array([0.5]), right_component_state)

    assert out[1, 0] < 0.0


def test_bridge_solver_entropy_floor_fields_parse_and_validate():
    cfg = BridgeSolverConfig.from_dict(
        {
            "kind": "scipy",
            "sigma": 1e-2,
            "bridge_steps": 3,
            "tol": 1e-2,
            "max_nodes": 20,
            "quadrature_order": 2,
            "n_density_samples": 7,
            "n_reference_grid": 5,
            "entropy_density_std_floor": 0.05,
            "failure_policy": "skip_pair",
        }
    )
    default_cfg = BridgeSolverConfig.from_dict(
        {
            "kind": "scipy",
            "sigma": 1e-2,
            "bridge_steps": 3,
            "tol": 1e-2,
            "max_nodes": 20,
            "quadrature_order": 2,
            "failure_policy": "skip_pair",
        }
    )

    assert cfg.entropy_density_std_floor == 0.05
    assert default_cfg.entropy_density_std_floor is None
    try:
        BridgeSolverConfig.from_dict(
            {
                "kind": "scipy",
                "sigma": 1e-2,
                "bridge_steps": 3,
                "tol": 1e-2,
                "max_nodes": 20,
                "quadrature_order": 2,
                "entropy_density_std_floor": 0.0,
                "failure_policy": "skip_pair",
            }
        )
    except ValueError as exc:
        assert "entropy_density_std_floor" in str(exc)
    else:
        raise AssertionError("nonpositive entropy_density_std_floor should be rejected")


if __name__ == "__main__":
    test_entropy_score_single_component_matches_gaussian_score()
    test_entropy_score_symmetric_mixture_zero_at_midpoint()
    test_entropy_score_finite_for_small_sigma()
    test_entropy_only_rhs_uses_configured_score_projection()
    test_bridge_solver_density_fields_parse()
    test_entropy_density_std_floor_applies_only_to_reference_stds()
    test_negative_entropy_rhs_accelerates_toward_mixture_center()
    test_bridge_solver_entropy_floor_fields_parse_and_validate()
    print("entropy internal smoke checks passed")
