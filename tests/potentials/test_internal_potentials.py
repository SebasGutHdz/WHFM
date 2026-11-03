from __future__ import annotations

import math

import numpy as np
import pytest
import torch
import torch.distributions as td

from whfm.bridge import GaussianBridgeSolver
from whfm.gaussian_paths import MeanStdBVPGaussianPath
from whfm.losses import flow_matching_loss
from whfm.Potentials import ConfiguredPotential
from whfm.Potentials.internal_potentials import (
    EntropyCostPotential,
    EntropyPotential,
    build_internal_potential,
    entropy_cost,
    gaussian_mixture_log_density,
)


class DummyGaussianPath:
    def __init__(self, means: torch.Tensor, stds: torch.Tensor):
        self.means = means
        self.stds = stds

    def mean(self, _t):
        return self.means

    def gamma(self, _t):
        return self.stds


def supplied_entropy_cost(xt, t, gpath):
    batch, samples, n_time, dim = xt.shape
    assert t.shape == (n_time,)
    mean_t = gpath.mean(t).detach()
    gamma_t = gpath.gamma(t).detach()
    normals = td.Normal(
        mean_t.reshape(batch * n_time, dim),
        gamma_t.reshape(batch * n_time, 1)
        * torch.ones(batch * n_time, dim, dtype=xt.dtype),
    )
    indep_normals = td.Independent(normals, 1)
    expanded = xt.unsqueeze(2).expand(-1, -1, batch, -1, -1)
    log_components = indep_normals.log_prob(
        expanded.reshape(batch * samples, batch * n_time, dim)
    ).reshape(batch * samples, batch, n_time)
    return log_components.exp().mean(dim=1).log().reshape(batch, samples, n_time)


def test_entropy_cost_matches_supplied_formula_and_detaches_path():
    means = torch.tensor(
        [[[0.0, 0.5], [0.5, 1.0]], [[1.0, -0.5], [1.5, 0.0]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    stds = torch.full((2, 2, 1), 0.7, dtype=torch.float64, requires_grad=True)
    path = DummyGaussianPath(means, stds)
    xt = torch.tensor(
        [
            [[[0.1, 0.2], [0.4, 0.8]], [[0.9, -0.1], [1.2, 0.1]]],
            [[[0.3, 0.4], [0.7, 0.9]], [[1.1, -0.4], [1.4, -0.1]]],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    t = torch.tensor([0.0, 1.0], dtype=torch.float64)

    actual = entropy_cost(xt, t, path)
    expected = supplied_entropy_cost(xt, t, path)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)

    actual.sum().backward()
    assert xt.grad is not None and torch.isfinite(xt.grad).all()
    assert means.grad is None
    assert stds.grad is None


def test_gaussian_mixture_log_density_is_finite_in_low_density_region():
    x = torch.tensor([[1.0e6]], dtype=torch.float64)
    means = torch.tensor([[0.0], [1.0]], dtype=torch.float64)
    stds = torch.ones(2, 1, dtype=torch.float64)

    value = gaussian_mixture_log_density(x, means, stds)

    assert value.shape == (1,)
    assert torch.isfinite(value).all()
    assert value.item() < -1.0e10


def test_entropy_batch_energy_uses_raw_log_density_and_configured_bandwidth():
    x = torch.tensor(
        [[-1.0, 0.2], [0.5, -0.3], [1.5, 0.7]], dtype=torch.float64
    )
    bandwidth = 1.5
    potential = EntropyPotential(bandwidth=bandwidth)
    stds = torch.full((x.shape[0], 1), bandwidth, dtype=x.dtype)
    expected = gaussian_mixture_log_density(x, x, stds)

    torch.testing.assert_close(potential.batch_energy(x), expected)
    assert potential.bandwidth == bandwidth
    assert not hasattr(potential, "beta")
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        EntropyPotential(beta=2.0)


def test_configured_entropy_applies_coefficient_once_and_forwards_bandwidth():
    coefficient = -10.0
    bandwidth = 1.5
    configured = ConfiguredPotential(
        {
            "linear": None,
            "internal": ("entropy", coefficient, {"bandwidth": bandwidth}),
            "interaction": None,
        }
    )
    x = torch.tensor(
        [[-1.0, 0.2], [0.5, -0.3], [1.5, 0.7]], dtype=torch.float64
    )
    means = torch.tensor(
        [[[-0.5, 0.0], [1.0, 0.5]]] * x.shape[0], dtype=x.dtype
    )
    stds = torch.full((x.shape[0], 2, 1), 0.8, dtype=x.dtype)

    raw_energy = configured.internal.batch_energy(x)
    raw_score = configured.internal.score_from_gaussian_mixture(x, means, stds)
    torch.testing.assert_close(configured.internal_energy(x), coefficient * raw_energy)
    torch.testing.assert_close(
        configured.internal_gradient_from_gaussian_mixture(x, means, stds),
        coefficient * raw_score,
    )
    assert configured.internal.bandwidth == bandwidth


@pytest.mark.parametrize("potential_cls", [EntropyPotential, EntropyCostPotential])
def test_entropy_scores_have_expected_shape(potential_cls):
    x = torch.randn(3, 5, 2, dtype=torch.float64)
    means = torch.randn(3, 4, 2, dtype=torch.float64)
    stds = torch.rand(3, 4, 1, dtype=torch.float64) + 0.5

    score = potential_cls().score_from_gaussian_mixture(x, means, stds)

    assert score.shape == x.shape
    assert torch.isfinite(score).all()


def test_analytic_score_equals_autograd_of_proposed_cost():
    generator = torch.Generator().manual_seed(17)
    x = torch.randn(3, 5, 2, generator=generator, dtype=torch.float64)
    means = torch.randn(3, 4, 2, generator=generator, dtype=torch.float64)
    stds = torch.rand(3, 4, 1, generator=generator, dtype=torch.float64) + 0.3

    analytic = EntropyPotential().score_from_gaussian_mixture(x, means, stds)
    autograd_score = EntropyCostPotential().score_from_gaussian_mixture(x, means, stds)

    torch.testing.assert_close(analytic, autograd_score, atol=1e-11, rtol=1e-11)
    assert isinstance(build_internal_potential("entropy_cost"), EntropyCostPotential)


def _configured_entropy(name: str):
    return ConfiguredPotential(
        {"linear": None, "internal": (name, -0.2), "interaction": None}
    )


def _solve_small_entropy_path(name: str):
    path = MeanStdBVPGaussianPath(
        _configured_entropy(name),
        sigma=0.7,
        n_steps=4,
        tol=1e-5,
        max_nodes=80,
        quadrature_order=2,
        n_density_samples=2,
        entropy_reference_max_iterations=1,
        num_workers=1,
    )
    x0 = torch.tensor([[-1.0], [0.5]], dtype=torch.float64)
    x1 = torch.tensor([[0.25], [1.5]], dtype=torch.float64)
    states = path.batch_solve(x0, x1)
    return path, states, x0, x1


def test_analytic_and_proposed_cost_produce_same_bridge_and_optimizer_step():
    analytic_path, analytic_states, x0, x1 = _solve_small_entropy_path("entropy")
    cost_path, cost_states, _, _ = _solve_small_entropy_path("entropy_cost")
    torch.testing.assert_close(analytic_states, cost_states, atol=2e-7, rtol=2e-7)

    t = torch.tensor([[0.3], [0.7]], dtype=torch.float64)
    epsilon = torch.tensor([[0.2], [-0.4]], dtype=torch.float64)
    analytic_xt = analytic_path.sample_xt(x0, x1, t, epsilon)
    cost_xt = cost_path.sample_xt(x0, x1, t, epsilon)
    analytic_target = analytic_path.compute_ut(x0, x1, t, analytic_xt)
    cost_target = cost_path.compute_ut(x0, x1, t, cost_xt)
    torch.testing.assert_close(analytic_xt, cost_xt, atol=2e-7, rtol=2e-7)
    torch.testing.assert_close(analytic_target, cost_target, atol=2e-7, rtol=2e-7)

    torch.manual_seed(29)
    analytic_model = torch.nn.Linear(2, 1, dtype=torch.float64)
    cost_model = torch.nn.Linear(2, 1, dtype=torch.float64)
    cost_model.load_state_dict(analytic_model.state_dict())
    analytic_optimizer = torch.optim.SGD(analytic_model.parameters(), lr=1e-2)
    cost_optimizer = torch.optim.SGD(cost_model.parameters(), lr=1e-2)
    losses = []
    for model, optimizer, xt, target in (
        (analytic_model, analytic_optimizer, analytic_xt, analytic_target),
        (cost_model, cost_optimizer, cost_xt, cost_target),
    ):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(torch.cat([xt, t], dim=-1))
        loss = flow_matching_loss(prediction, target)
        losses.append(loss.detach())
        loss.backward()
        optimizer.step()
    torch.testing.assert_close(losses[0], losses[1], atol=2e-9, rtol=2e-9)
    for analytic_parameter, cost_parameter in zip(
        analytic_model.parameters(), cost_model.parameters()
    ):
        torch.testing.assert_close(
            analytic_parameter, cost_parameter, atol=2e-9, rtol=2e-9
        )


def test_invalid_entropy_inputs_are_rejected():
    with pytest.raises(ValueError, match="bandwidth"):
        EntropyPotential(bandwidth=0.0)
    with pytest.raises(ValueError, match="bandwidth"):
        EntropyPotential().batch_energy(torch.zeros(2, 1), bandwidth=-1.0)
    with pytest.raises(TypeError, match="mapping"):
        build_internal_potential("entropy", parameters=1.5)
    with pytest.raises(ValueError, match="sigma_floor"):
        gaussian_mixture_log_density(
            torch.zeros(1, 1), torch.zeros(1, 1), torch.ones(1, 1), sigma_floor=0.0
        )
    with pytest.raises(ValueError, match="final dimension 1"):
        gaussian_mixture_log_density(
            torch.zeros(1, 2), torch.zeros(1, 2), torch.ones(1, 2)
        )
    with pytest.raises(ValueError, match="positive values"):
        gaussian_mixture_log_density(
            torch.zeros(1, 1), torch.zeros(1, 1), torch.zeros(1, 1)
        )
    path = DummyGaussianPath(torch.zeros(1, 2, 1), torch.ones(1, 2, 1))
    with pytest.raises(ValueError, match="t must have shape"):
        entropy_cost(torch.zeros(1, 1, 2, 1), torch.zeros(1), path)


def _fake_result(job, *, shift=0.0, failure=None):
    state = job["guess"].T.copy()
    if state.shape[0] > 2:
        state[1:-1, 0] += shift
    state[0, 0] = job["start"][0]
    state[-1, 0] = job["end"][0]
    return {
        "index": int(job["index"]),
        "state": None if failure is not None else state,
        "failure_message": failure,
        "solve_time": 0.01,
        "iterations": None if failure is not None else 1,
        "mesh_nodes": None if failure is not None else state.shape[0],
    }


def _fixed_point_path(max_iterations=3, tolerance=1e-12, relaxation=0.5):
    return MeanStdBVPGaussianPath(
        _configured_entropy("entropy"),
        sigma=0.7,
        n_steps=3,
        quadrature_order=1,
        n_density_samples=2,
        entropy_reference_max_iterations=max_iterations,
        entropy_reference_tolerance=tolerance,
        entropy_reference_relaxation=relaxation,
        num_workers=1,
    )


def test_fixed_point_rebuilds_references_from_relaxed_solved_paths(monkeypatch):
    path = _fixed_point_path(max_iterations=2, tolerance=1e-15, relaxation=0.5)
    references = []

    def fake_solve_jobs(jobs):
        references.append(jobs[0]["reference_means"].copy())
        return [_fake_result(job, shift=0.5) for job in jobs]

    monkeypatch.setattr(path, "_solve_pair_jobs", fake_solve_jobs)
    x0 = torch.tensor([[-1.0], [0.0]], dtype=torch.float64)
    x1 = torch.tensor([[1.0], [2.0]], dtype=torch.float64)

    states = path.batch_solve(x0, x1)

    assert states.shape == (2, 4, 4)
    assert len(references) == 2
    np.testing.assert_allclose(
        references[1][1:-1, :, 0] - references[0][1:-1, :, 0],
        0.25,
    )
    assert path.solve_metadata["entropy_reference_iterations"] == 2
    assert path.solve_metadata["entropy_reference_converged"] is False
    assert len(path.solve_metadata["entropy_reference_delta_history"]) == 2
    assert path.solve_metadata["entropy_reference_component_indices"] == [0, 1]


def test_fixed_point_stops_when_reference_paths_converge(monkeypatch):
    path = _fixed_point_path(max_iterations=3, tolerance=1e-12)
    calls = 0

    def fake_solve_jobs(jobs):
        nonlocal calls
        calls += 1
        return [_fake_result(job) for job in jobs]

    monkeypatch.setattr(path, "_solve_pair_jobs", fake_solve_jobs)
    path.batch_solve(
        torch.tensor([[-1.0], [0.0]], dtype=torch.float64),
        torch.tensor([[1.0], [2.0]], dtype=torch.float64),
    )

    assert calls == 1
    assert path.solve_metadata["entropy_reference_iterations"] == 1
    assert path.solve_metadata["entropy_reference_converged"] is True
    assert path.solve_metadata["entropy_reference_final_delta"] == pytest.approx(0.0)


def test_fixed_point_removes_failed_pairs_and_preserves_original_indices(monkeypatch):
    path = MeanStdBVPGaussianPath(
        _configured_entropy("entropy"),
        sigma=0.7,
        n_steps=3,
        quadrature_order=1,
        n_density_samples=3,
        entropy_reference_max_iterations=2,
        entropy_reference_tolerance=1e-15,
        entropy_reference_relaxation=0.5,
        num_workers=1,
    )
    batch_sizes = []

    def fake_solve_jobs(jobs):
        batch_sizes.append(len(jobs))
        if len(batch_sizes) == 1:
            return [
                _fake_result(job, shift=0.5, failure="forced failure")
                if int(job["index"]) == 1
                else _fake_result(job, shift=0.5)
                for job in jobs
            ]
        return [_fake_result(job, shift=0.5) for job in jobs]

    monkeypatch.setattr(path, "_solve_pair_jobs", fake_solve_jobs)
    states = path.batch_solve(
        torch.tensor([[-1.0], [0.0], [1.0]], dtype=torch.float64),
        torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float64),
    )

    assert batch_sizes == [3, 2]
    assert states.shape[0] == 2
    assert path.success_indices.tolist() == [0, 2]
    assert path.failed_indices.tolist() == [1]
    assert "iteration 1" in path.failure_messages[1]
    assert path.solve_metadata["entropy_reference_component_fallbacks"] == 0




def _fixed_point_bridge_solver(failure_policy):
    return GaussianBridgeSolver(
        _configured_entropy("entropy"),
        sigma=0.7,
        bridge_steps=3,
        tol=1e-3,
        max_nodes=20,
        quadrature_order=1,
        n_density_samples=2,
        entropy_reference_max_iterations=3,
        entropy_reference_tolerance=1e-12,
        entropy_reference_relaxation=0.5,
        num_workers=1,
        failure_policy=failure_policy,
    )


def test_total_failure_preserves_fixed_point_diagnostics_under_skip(monkeypatch):
    def fail_all(_path, jobs):
        return [_fake_result(job, failure="forced total failure") for job in jobs]

    monkeypatch.setattr(MeanStdBVPGaussianPath, "_solve_pair_jobs", fail_all)
    solution = _fixed_point_bridge_solver("skip_pair").solve_batch(
        torch.tensor([[-1.0], [0.0]], dtype=torch.float64),
        torch.tensor([[1.0], [2.0]], dtype=torch.float64),
    )

    assert solution.num_successful == 0
    assert solution.failed_indices.tolist() == [0, 1]
    assert solution.entropy_reference_iterations == 1
    assert solution.entropy_reference_converged is False
    assert math.isfinite(solution.solve_time_seconds)
    assert solution.solve_time_seconds >= 0.0
    assert len(solution.failure_messages) == 2


def test_raise_policy_stops_before_another_reference_iteration(monkeypatch):
    calls = 0

    def fail_one(_path, jobs):
        nonlocal calls
        calls += 1
        return [
            _fake_result(job, failure="forced partial failure")
            if index == 0
            else _fake_result(job, shift=0.5)
            for index, job in enumerate(jobs)
        ]

    monkeypatch.setattr(MeanStdBVPGaussianPath, "_solve_pair_jobs", fail_one)
    with pytest.raises(RuntimeError, match="reference iteration 1"):
        _fixed_point_bridge_solver("raise").solve_batch(
            torch.tensor([[-1.0], [0.0]], dtype=torch.float64),
            torch.tensor([[1.0], [2.0]], dtype=torch.float64),
        )
    assert calls == 1


def test_fixed_point_configuration_defaults_and_validation():
    from whfm.config import BridgeSolverConfig

    base = {
        "kind": "scipy",
        "sigma": 0.2,
        "bridge_steps": 3,
        "tol": 1e-3,
        "max_nodes": 20,
        "quadrature_order": 1,
        "failure_policy": "skip_pair",
    }
    config = BridgeSolverConfig.from_dict(base)
    assert config.entropy_reference_max_iterations == 1
    assert config.entropy_reference_tolerance == pytest.approx(1e-3)
    assert config.entropy_reference_relaxation == pytest.approx(0.5)

    invalid_values = [
        ("entropy_reference_max_iterations", 0),
        ("entropy_reference_tolerance", 0.0),
        ("entropy_reference_relaxation", 0.0),
        ("entropy_reference_relaxation", 1.1),
    ]
    for key, value in invalid_values:
        with pytest.raises(ValueError, match=key):
            BridgeSolverConfig.from_dict({**base, key: value})
