from __future__ import annotations

import numpy as np
import pytest
import torch

from whfm.gaussian_paths import MeanStdBVPGaussianPath, _make_mean_std_bvp_rhs


class ZeroPotential:
    has_interaction = False
    has_internal = False

    @staticmethod
    def linear_gradient(x):
        return torch.zeros_like(x)


def test_zero_terminal_bvp_keeps_preterminal_scale_positive():
    path = MeanStdBVPGaussianPath(
        ZeroPotential(), sigma_source=0.2, sigma_target=0.0,
        n_steps=5, tol=1e-5, max_nodes=100, quadrature_order=1,
        num_workers=1,
    )
    x0 = torch.tensor([[0.0]], dtype=torch.float64)
    x1 = torch.tensor([[1.0]], dtype=torch.float64)
    states = path.batch_solve(x0, x1)
    assert states.shape == (1, 6, 4)
    assert torch.all(states[0, :-1, 2] > 0)
    torch.testing.assert_close(states[0, -1, 2], torch.tensor(0.0, dtype=torch.float64))
    with pytest.raises(RuntimeError, match="undefined at the zero-scale terminal"):
        path.compute_ut(x0, x1, torch.ones(1, 1, dtype=torch.float64), x1)


def test_zero_terminal_reference_particles_collapse_without_density_floor():
    path = MeanStdBVPGaussianPath(
        ZeroPotential(), sigma_source=0.2, sigma_target=0.0,
        n_steps=2, quadrature_order=1, n_density_samples=1, num_workers=1,
    )
    x0 = torch.tensor([[0.0]], dtype=torch.float64)
    x1 = torch.tensor([[1.0]], dtype=torch.float64)
    means = np.array([[[0.0], [0.5], [1.0]]])
    scales = np.array([[0.2, 0.1, 0.0]])
    _, samples, reference_means, reference_scales = path._build_reference_samples(
        x0, x1, means, np.ones_like(means), scales,
        np.full_like(scales, -0.2), build_samples=True,
        sample_indices=np.array([0]),
    )
    np.testing.assert_allclose(reference_scales[-1], 0)
    np.testing.assert_allclose(samples[-1], reference_means[-1])


def test_gaussian_scale_force_retains_inverse_dimension_mass():
    class HarmonicPotential:
        has_internal = False

        @staticmethod
        def linear_gradient(x):
            return x

    eps = np.array([[-1.0, -1.0], [1.0, 1.0]])
    rhs = _make_mean_std_bvp_rhs(
        HarmonicPotential(), dim=2, eps=eps, weights=np.array([0.5, 0.5]),
    )
    state = np.array([[0.0], [0.0], [0.0], [0.0], [0.3], [0.0]])
    np.testing.assert_allclose(rhs(np.array([0.5]), state)[:, 0], [0, 0, 0, 0, 0, -0.3])
