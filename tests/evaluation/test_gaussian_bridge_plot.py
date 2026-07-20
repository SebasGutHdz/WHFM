from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


def _load_gaussian_bridge_plot(monkeypatch):
    return importlib.import_module("whfm.evaluation.gaussian_bridge_plot")


def _synthetic_solution():
    mean = torch.tensor(
        [
            [
                [0.0, 0.0],
                [0.4, 0.15],
                [0.8, 0.35],
                [1.0, 0.6],
            ]
        ],
        dtype=torch.float32,
    )
    std = torch.tensor([[[0.1], [0.2], [0.15], [0.08]]], dtype=torch.float32)
    return SimpleNamespace(
        num_successful=1,
        mean=mean,
        std=std,
        x0=mean[:, 0],
        x1=mean[:, -1],
        time_grid=torch.linspace(0.0, 1.0, mean.shape[1]),
        endpoint_errors=torch.zeros(1, 4),
        solve_time_seconds=0.0,
    )


def test_parser_defaults_and_num_frames(monkeypatch):
    bridge_plot = _load_gaussian_bridge_plot(monkeypatch)

    args = bridge_plot.build_parser().parse_args(["results/stunnel/run"])
    assert args.run_dir == "results/stunnel/run"
    assert args.checkpoint is None
    assert args.direction == "all"
    assert args.model_kind == "all"
    assert args.sample_index == 0
    assert args.node_steps is None
    assert args.num_frames == 8
    assert args.num_cloud_samples == 300
    assert args.density_levels == "1,2,3"
    assert args.visual_seed == 12345
    assert args.std_visual_scale == 1.0
    assert args.no_potential_background is False

    args = bridge_plot.build_parser().parse_args([
        "results/stunnel/run",
        "--num-frames",
        "10",
        "--num-cloud-samples",
        "25",
        "--density-levels",
        "0.5,1,2",
        "--visual-seed",
        "999",
        "--std-visual-scale",
        "50",
    ])
    assert args.num_frames == 10
    assert args.num_cloud_samples == 25
    assert args.density_levels == "0.5,1,2"
    assert args.visual_seed == 999
    assert args.std_visual_scale == 50.0


def test_snapshot_indices_are_evenly_spaced(monkeypatch):
    bridge_plot = _load_gaussian_bridge_plot(monkeypatch)

    assert bridge_plot._snapshot_indices(11, 6) == [0, 2, 4, 6, 8, 10]
    assert bridge_plot._snapshot_indices(4, 8) == [0, 1, 2, 3]
    with pytest.raises(ValueError, match="num_frames must be positive"):
        bridge_plot._snapshot_indices(4, 0)


def test_density_level_parsing_and_validation(monkeypatch):
    bridge_plot = _load_gaussian_bridge_plot(monkeypatch)

    assert bridge_plot._parse_density_levels("2,1,1,0.5") == [0.5, 1.0, 2.0]
    assert bridge_plot._parse_density_levels([3, 1]) == [1.0, 3.0]
    with pytest.raises(ValueError, match="positive finite"):
        bridge_plot._parse_density_levels("1,0")
    with pytest.raises(ValueError, match="at least one"):
        bridge_plot._parse_density_levels("")


def test_gaussian_cloud_sampling_is_deterministic(monkeypatch):
    bridge_plot = _load_gaussian_bridge_plot(monkeypatch)
    mean_plot = np.array([[[0.0, 0.0], [1.0, 1.0]], [[0.5, 0.5], [1.5, 1.5]]])
    std = np.array([0.1, 0.2])

    first = bridge_plot._sample_gaussian_clouds(mean_plot, std, num_cloud_samples=5, seed=123)
    second = bridge_plot._sample_gaussian_clouds(mean_plot, std, num_cloud_samples=5, seed=123)
    different = bridge_plot._sample_gaussian_clouds(mean_plot, std, num_cloud_samples=5, seed=124)

    assert first.shape == (2, 10, 2)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, different)


def test_gaussian_cloud_sampling_respects_visual_scale(monkeypatch):
    bridge_plot = _load_gaussian_bridge_plot(monkeypatch)
    mean_plot = np.zeros((1, 1, 2), dtype=float)
    std = np.array([0.1])

    base = bridge_plot._sample_gaussian_clouds(
        mean_plot, std, num_cloud_samples=100, seed=123, std_visual_scale=1.0
    )
    scaled = bridge_plot._sample_gaussian_clouds(
        mean_plot, std, num_cloud_samples=100, seed=123, std_visual_scale=10.0
    )

    assert scaled.std() > base.std() * 5.0


def test_gaussian_cloud_sampling_handles_particle_shapes(monkeypatch):
    bridge_plot = _load_gaussian_bridge_plot(monkeypatch)
    mean_plot = np.zeros((3, 2, 2), dtype=float)
    std = np.ones(3, dtype=float) * 0.1

    clouds = bridge_plot._sample_gaussian_clouds(mean_plot, std, num_cloud_samples=4, seed=123)

    assert clouds.shape == (3, 8, 2)


def test_save_gaussian_bridge_static_plot_smoke(monkeypatch, tmp_path):
    pytest.importorskip("matplotlib")
    bridge_plot = _load_gaussian_bridge_plot(monkeypatch)

    output_path = tmp_path / "bridge.png"
    eval_config = SimpleNamespace(plot_dir1=0, plot_dir2=1, plot_xlim=[0.0, 0.0], plot_ylim=[0.0, 0.0])
    potential = SimpleNamespace(has_linear=False)

    bridge_plot._save_gaussian_bridge_static_plot(
        output_path=output_path,
        tag="smoke",
        solution=_synthetic_solution(),
        potential=potential,
        evaluation_config=eval_config,
        num_frames=4,
        density_levels=[1.0, 2.0, 3.0],
        std_visual_scale=2.0,
        potential_background=False,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_save_gaussian_bridge_gif_smoke(monkeypatch, tmp_path):
    pytest.importorskip("matplotlib")
    pytest.importorskip("PIL")
    bridge_plot = _load_gaussian_bridge_plot(monkeypatch)

    gif_path = tmp_path / "bridge.gif"
    eval_config = SimpleNamespace(plot_dir1=0, plot_dir2=1, plot_xlim=[0.0, 0.0], plot_ylim=[0.0, 0.0])
    potential = SimpleNamespace(has_linear=False)

    bridge_plot._save_gaussian_bridge_gif(
        gif_path=gif_path,
        tag="smoke",
        solution=_synthetic_solution(),
        potential=potential,
        evaluation_config=eval_config,
        fps=4,
        dpi=60,
        num_cloud_samples=20,
        visual_seed=123,
        std_visual_scale=2.0,
        potential_background=False,
    )

    assert gif_path.exists()
    assert gif_path.stat().st_size > 0


def test_create_gaussian_bridge_visualizations_uses_learned_guess(monkeypatch, tmp_path):
    bridge_plot = _load_gaussian_bridge_plot(monkeypatch)
    captured = {}

    class DummyModel(torch.nn.Module):
        def forward(self, x):
            return x

    class DummyNodeSolver:
        def __init__(self, method, node_steps):
            captured["node_solver"] = (method, node_steps)

        def integrate(self, model, x0):
            states = torch.stack([x0, x0 + 0.5, x0 + 1.0], dim=0)
            velocities = torch.ones_like(states)
            return SimpleNamespace(states=states, velocities=velocities)

        def prepare_bridge_guess(self, states, velocities, x0, x1, *, reverse, bridge_steps):
            captured["guess_inputs"] = {
                "states": states.clone(),
                "velocities": velocities.clone(),
                "x0": x0.clone(),
                "x1": x1.clone(),
                "reverse": reverse,
                "bridge_steps": bridge_steps,
            }
            return np.full((1, bridge_steps + 1, 2), 3.0), np.full((1, bridge_steps + 1, 2), 4.0)

    class DummyBridgeSolver:
        def __init__(self, *args, **kwargs):
            captured["bridge_kwargs"] = kwargs

        def solve_batch(self, x0, x1, *, mean_guess=None, mean_velocity_guess=None):
            captured["solve_batch"] = {
                "x0": x0.clone(),
                "x1": x1.clone(),
                "mean_guess": mean_guess.copy(),
                "mean_velocity_guess": mean_velocity_guess.copy(),
            }
            return _synthetic_solution()

    train_config = SimpleNamespace(
        dtype="float32",
        device="cpu",
        node_solver=SimpleNamespace(method="euler", node_steps=5),
        bridge_solver=SimpleNamespace(
            sigma_source=0.1,
            sigma_target=0.1,
            bridge_steps=2,
            tol=1e-3,
            max_nodes=20,
            quadrature_order=3,
            use_monte_carlo=True,
            monte_carlo_samples=5,
            failure_policy="skip_pair",
        ),
    )
    problem_config = SimpleNamespace(
        dimension=2,
        evaluation=SimpleNamespace(plot_dir1=0, plot_dir2=1, plot_xlim=[0.0, 0.0], plot_ylim=[0.0, 0.0]),
        functional=SimpleNamespace(to_potential_cfg=lambda: {"linear": None, "internal": None, "interaction": None}),
    )
    monkeypatch.setattr(bridge_plot, "load_train_config", lambda path: train_config)
    monkeypatch.setattr(bridge_plot, "load_problem_config", lambda path: problem_config)
    monkeypatch.setattr(bridge_plot, "resolve_checkpoint_path", lambda run_dir, checkpoint=None: tmp_path / "final.pt")
    monkeypatch.setattr(bridge_plot.torch, "load", lambda *args, **kwargs: {"directions": {"forward": {"model": {}}}})
    monkeypatch.setattr(bridge_plot, "available_model_specs", lambda *args, **kwargs: [("forward", "online")])
    monkeypatch.setattr(bridge_plot, "ConfiguredPotential", lambda cfg: SimpleNamespace(has_linear=False))
    monkeypatch.setattr(bridge_plot, "NodeSolver", DummyNodeSolver)
    monkeypatch.setattr(bridge_plot, "GaussianBridgeSolver", DummyBridgeSolver)
    monkeypatch.setattr(bridge_plot, "_build_model", lambda *args, **kwargs: DummyModel())
    monkeypatch.setattr(bridge_plot, "_model_state_dict", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        bridge_plot,
        "_prepare_evaluation_split",
        lambda *args, **kwargs: (torch.tensor([[1.0, 2.0]]), torch.tensor([[9.0, 10.0]])),
    )
    monkeypatch.setattr(bridge_plot, "_save_gaussian_bridge_static_plot", lambda **kwargs: kwargs["output_path"])
    monkeypatch.setattr(bridge_plot, "_save_gaussian_bridge_gif", lambda **kwargs: kwargs["gif_path"])

    rows = bridge_plot.create_gaussian_bridge_visualizations(tmp_path, node_steps=7)

    assert len(rows) == 1
    assert captured["node_solver"] == ("euler", 7)
    assert captured["guess_inputs"]["reverse"] is False
    assert captured["guess_inputs"]["bridge_steps"] == 2
    assert np.all(captured["solve_batch"]["mean_guess"] == 3.0)
    assert np.all(captured["solve_batch"]["mean_velocity_guess"] == 4.0)
    assert torch.equal(captured["solve_batch"]["x0"], torch.tensor([[1.0, 2.0]]))
    assert torch.equal(captured["solve_batch"]["x1"], torch.tensor([[2.0, 3.0]]))
