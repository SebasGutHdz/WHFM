from __future__ import annotations

import ast
import importlib
import types
from pathlib import Path

import pytest
import torch


class ZeroPotential:
    has_interaction = False

    def linear_gradient(self, x):
        return torch.zeros_like(x)

    def gradient(self, x):
        return torch.zeros_like(x)


def _load_modules(monkeypatch):
    gaussian_paths = importlib.import_module("whfm.gaussian_paths")
    bridge = importlib.import_module("whfm.bridge")
    return gaussian_paths, bridge


def _state(n_steps: int, start: float, end: float, sigma: float = 0.2):
    t = torch.linspace(0.0, 1.0, n_steps + 1)
    mu = ((1.0 - t) * start + t * end).reshape(-1, 1)
    mu_dot = torch.full_like(mu, end - start)
    std = torch.full_like(mu, sigma)
    std_dot = torch.zeros_like(mu)
    return torch.cat([mu, mu_dot, std, std_dot], dim=1).numpy()


def test_mean_std_bvp_parallel_matches_serial(monkeypatch):
    gaussian_paths, potentials = _load_real_modules(monkeypatch)
    potential = potentials.ConfiguredPotential(
        {"linear": ("stunnel", 0.0), "internal": None, "interaction": None}
    )
    x0 = torch.tensor([[0.0, 0.0], [1.0, -0.5], [-1.0, 0.25]], dtype=torch.float64)
    x1 = torch.tensor([[1.0, 0.5], [2.0, 0.5], [0.5, -0.25]], dtype=torch.float64)

    serial = gaussian_paths.MeanStdBVPGaussianPath(
        potential,
        sigma=0.2,
        n_steps=5,
        tol=1e-4,
        max_nodes=50,
        quadrature_order=1,
        num_workers=1,
    )
    parallel = gaussian_paths.MeanStdBVPGaussianPath(
        potential,
        sigma=0.2,
        n_steps=5,
        tol=1e-4,
        max_nodes=50,
        quadrature_order=1,
        num_workers=2,
    )

    serial_states = serial.batch_solve(x0, x1)
    parallel_states = parallel.batch_solve(x0, x1)

    assert torch.allclose(parallel_states, serial_states, atol=1e-7, rtol=1e-7)
    assert parallel.success_indices.tolist() == [0, 1, 2]
    assert parallel.failed_indices.tolist() == []
    assert len(parallel.solve_metadata["pair_solve_time_seconds"]) == 3
    assert all(nodes is not None for nodes in parallel.solve_metadata["pair_solver_mesh_nodes"])


def test_mean_std_bvp_partial_failure_preserves_original_indices(monkeypatch):
    gaussian_paths, _ = _load_modules(monkeypatch)

    def fake_solve_pair(job):
        index = int(job["index"])
        if index == 1:
            return {
                "index": index,
                "state": None,
                "failure_message": "forced failure",
                "solve_time": 0.25,
                "iterations": None,
                "mesh_nodes": None,
            }
        return {
            "index": index,
            "state": _state(3, float(job["start"][0]), float(job["end"][0])),
            "failure_message": None,
            "solve_time": 0.5 + index,
            "iterations": 2 + index,
            "mesh_nodes": 4 + index,
        }

    monkeypatch.setattr(gaussian_paths, "_solve_mean_std_bvp_pair", fake_solve_pair)
    path = gaussian_paths.MeanStdBVPGaussianPath(
        ZeroPotential(), sigma=0.2, n_steps=3, quadrature_order=1, num_workers=1
    )

    states = path.batch_solve(
        torch.tensor([[0.0], [1.0], [2.0]]),
        torch.tensor([[1.0], [2.0], [3.0]]),
    )

    assert states.shape == (2, 4, 4)
    assert path.success_indices.tolist() == [0, 2]
    assert path.failed_indices.tolist() == [1]
    assert path.failure_messages == {1: "forced failure"}
    assert path.solve_metadata["pair_solve_time_seconds"] == [0.5, 0.25, 2.5]
    assert path.solve_metadata["pair_solver_iterations"] == [2, None, 4]
    assert path.solve_metadata["pair_solver_mesh_nodes"] == [4, None, 6]


def test_bridge_solver_skip_pair_returns_empty_solution_when_all_pairs_fail(monkeypatch):
    gaussian_paths, bridge = _load_modules(monkeypatch)

    def fail_pair(job):
        return {
            "index": int(job["index"]),
            "state": None,
            "failure_message": "forced failure",
            "solve_time": 0.1,
            "iterations": None,
            "mesh_nodes": None,
        }

    monkeypatch.setattr(gaussian_paths, "_solve_mean_std_bvp_pair", fail_pair)
    solver = bridge.GaussianBridgeSolver(
        ZeroPotential(),
        sigma=0.2,
        bridge_steps=3,
        tol=1e-3,
        max_nodes=20,
        quadrature_order=1,
        failure_policy="skip_pair",
    )

    solution = solver.solve_batch(torch.tensor([[0.0], [1.0]]), torch.tensor([[1.0], [2.0]]))

    assert solution.num_pairs == 2
    assert solution.num_successful == 0
    assert solution.success_mask.tolist() == [False, False]
    assert solution.failed_indices.tolist() == [0, 1]
    assert solution.failure_messages == {0: "forced failure", 1: "forced failure"}


def test_bridge_solver_raise_policy_preserves_all_failure_error(monkeypatch):
    gaussian_paths, bridge = _load_modules(monkeypatch)

    def fail_pair(job):
        return {
            "index": int(job["index"]),
            "state": None,
            "failure_message": "forced failure",
            "solve_time": 0.1,
            "iterations": None,
            "mesh_nodes": None,
        }

    monkeypatch.setattr(gaussian_paths, "_solve_mean_std_bvp_pair", fail_pair)
    solver = bridge.GaussianBridgeSolver(
        ZeroPotential(),
        sigma=0.2,
        bridge_steps=3,
        tol=1e-3,
        max_nodes=20,
        quadrature_order=1,
        failure_policy="raise",
    )

    with pytest.raises(RuntimeError, match="forced failure"):
        solver.solve_batch(torch.tensor([[0.0]]), torch.tensor([[1.0]]))



def _load_real_modules(monkeypatch):
    gaussian_paths = importlib.import_module("whfm.gaussian_paths")
    potentials = importlib.import_module("whfm.Potentials")
    return gaussian_paths, potentials


def test_mean_std_bvp_parallel_uses_nonfork_context(monkeypatch):
    gaussian_paths, _ = _load_modules(monkeypatch)
    captured = {}

    class FakeContext:
        def __init__(self, method):
            self.method = method

        def get_start_method(self):
            return self.method

    class FakeExecutor:
        def __init__(self, **kwargs):
            captured["executor_kwargs"] = kwargs

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def map(self, fn, jobs):
            return [fn(job) for job in jobs]

    def fake_get_context(method):
        captured.setdefault("requested_methods", []).append(method)
        if method == "forkserver":
            return FakeContext(method)
        raise AssertionError(f"unexpected start method request: {method}")

    monkeypatch.setattr(gaussian_paths.multiprocessing, "get_context", fake_get_context)
    monkeypatch.setattr(gaussian_paths, "ProcessPoolExecutor", FakeExecutor)
    path = gaussian_paths.MeanStdBVPGaussianPath(
        ZeroPotential(),
        sigma=0.2,
        n_steps=3,
        tol=1e-4,
        max_nodes=50,
        quadrature_order=1,
        num_workers=2,
    )

    states = path.batch_solve(
        torch.tensor([[0.0], [1.0]], dtype=torch.float64),
        torch.tensor([[1.0], [2.0]], dtype=torch.float64),
    )

    assert states.shape == (2, 4, 4)
    assert captured["requested_methods"] == ["forkserver"]
    executor_kwargs = captured["executor_kwargs"]
    assert executor_kwargs["mp_context"].get_start_method() == "forkserver"
    assert executor_kwargs["initializer"] is gaussian_paths._mean_std_bvp_worker_init
    assert executor_kwargs["max_workers"] == 2


def test_mean_std_bvp_parallel_supports_stunnel_autograd_gradient(monkeypatch):
    gaussian_paths, potentials = _load_real_modules(monkeypatch)
    potential = potentials.ConfiguredPotential(
        {"linear": ("stunnel", 0.0), "internal": None, "interaction": None}
    )
    path = gaussian_paths.MeanStdBVPGaussianPath(
        potential,
        sigma=0.2,
        n_steps=3,
        tol=1e-4,
        max_nodes=50,
        quadrature_order=1,
        num_workers=2,
    )

    states = path.batch_solve(
        torch.tensor([[-1.0, -0.5], [0.25, -0.25]], dtype=torch.float64),
        torch.tensor([[1.0, 0.5], [0.75, 0.25]], dtype=torch.float64),
    )

    assert states.shape == (2, 4, 6)
    assert path.success_indices.tolist() == [0, 1]
    assert path.failed_indices.tolist() == []
    assert not any("autograd" in message for message in path.failure_messages.values())


def test_deterministic_bvp_zero_potential_samples_straight_line(monkeypatch):
    gaussian_paths, _ = _load_modules(monkeypatch)
    x0 = torch.tensor([[0.0], [2.0]], dtype=torch.float64)
    x1 = torch.tensor([[1.0], [5.0]], dtype=torch.float64)
    path = gaussian_paths.DeterministicBVPPath(
        ZeroPotential(),
        n_steps=4,
        tol=1e-5,
        max_nodes=50,
    )

    states = path.batch_solve(x0, x1)

    t_grid = torch.linspace(0.0, 1.0, 5, dtype=torch.float64)
    expected_gamma = (1.0 - t_grid[None, :, None]) * x0[:, None, :] + t_grid[
        None, :, None
    ] * x1[:, None, :]
    expected_gamma_prime = (x1 - x0)[:, None, :].expand_as(expected_gamma)
    assert states.shape == (2, 5, 2)
    assert torch.allclose(states[:, :, :1], expected_gamma, atol=1e-7, rtol=1e-7)
    assert torch.allclose(states[:, :, 1:], expected_gamma_prime, atol=1e-7, rtol=1e-7)

    t = torch.tensor([[0.25], [0.75]], dtype=torch.float64)
    epsilon_a = torch.full_like(x0, 10.0)
    epsilon_b = torch.full_like(x0, -10.0)
    xt_a = path.sample_xt(x0, x1, t, epsilon_a)
    xt_b = path.sample_xt(x0, x1, t, epsilon_b)
    expected_xt = (1.0 - t) * x0 + t * x1
    assert torch.allclose(xt_a, expected_xt, atol=1e-7, rtol=1e-7)
    assert torch.allclose(xt_b, expected_xt, atol=1e-7, rtol=1e-7)

    ut = path.compute_ut(x0, x1, t, xt_a)
    assert torch.allclose(ut, x1 - x0, atol=1e-7, rtol=1e-7)


def test_deterministic_bvp_partial_failure_preserves_original_indices(monkeypatch):
    gaussian_paths, _ = _load_modules(monkeypatch)
    calls = []

    def fake_solve_bvp(rhs, bc, grid, guess, *, tol, max_nodes, to_tensor):
        index = len(calls)
        calls.append(index)
        raw = types.SimpleNamespace(x=grid, niter=None, sol=lambda t: guess)
        if index == 1:
            return types.SimpleNamespace(
                success=False,
                message="forced deterministic failure",
                raw=raw,
            )
        raw.niter = 3 + index
        return types.SimpleNamespace(success=True, message="", raw=raw)

    monkeypatch.setattr(gaussian_paths, "scipy_solve_bvp", fake_solve_bvp)
    path = gaussian_paths.DeterministicBVPPath(ZeroPotential(), n_steps=3, tol=1e-4)

    states = path.batch_solve(
        torch.tensor([[0.0], [1.0], [2.0]]),
        torch.tensor([[1.0], [2.0], [3.0]]),
    )

    assert states.shape == (2, 4, 2)
    assert path.success_indices.tolist() == [0, 2]
    assert path.failed_indices.tolist() == [1]
    assert path.failure_messages == {1: "forced deterministic failure"}
    assert path.solve_metadata["pair_solver_iterations"] == [3, None, 5]
    assert path.solve_metadata["pair_solver_mesh_nodes"] == [4, 4, 4]


def test_package_surface_exports_only_active_gaussian_paths():
    root = Path(__file__).resolve().parents[2]
    module = ast.parse((root / "src" / "whfm" / "__init__.py").read_text())
    all_assignment = next(
        node for node in module.body if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets)
    )
    exported = {item.value for item in all_assignment.value.elts}

    assert {"GaussianPath", "MeanStdBVPGaussianPath", "DeterministicBVPPath"} <= exported
    assert exported.isdisjoint(
        {
            "HarmonicGaussianPath",
            "HillGaussianPath",
            "DensityGaussianPath",
            "InteractionGaussianPath",
            "ParticleBVPGaussianPath",
            "ParametricBVPGaussianPath",
        }
    )

