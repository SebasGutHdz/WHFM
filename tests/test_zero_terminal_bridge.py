from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch
import yaml

from whfm.config import NodeSolverConfig, config_to_plain_dict
from whfm.losses import gaussian_sample_and_target
from whfm.node import NodeSolver
from whfm.trainer_v2 import BridgeTargetSet, TrainV2Config

ROOT = Path(__file__).resolve().parents[1]


def _v2_config():
    return yaml.safe_load((ROOT / "configs/train_v2.yaml").read_text())


def test_zero_terminal_config_derives_tau_horizon():
    config = TrainV2Config.from_dict(_v2_config())
    assert config.bridge_solver.sigma_source > 0
    assert config.bridge_solver.sigma_target == 0
    assert config.tau > 0
    assert config.node_solver.time_end == pytest.approx(1 - config.tau)


def test_zero_terminal_config_rejects_missing_cutoff_and_wrong_horizon():
    no_cutoff = _v2_config()
    no_cutoff["tau"] = 0
    with pytest.raises(ValueError, match="requires tau > 0"):
        TrainV2Config.from_dict(no_cutoff)
    wrong_horizon = copy.deepcopy(_v2_config())
    wrong_horizon["node_solver"]["time_end"] = 1.0
    with pytest.raises(ValueError, match="must equal 1 - tau"):
        TrainV2Config.from_dict(wrong_horizon)


def test_legacy_zero_xi_loads_but_nonzero_is_rejected():
    config = NodeSolverConfig.from_dict({"method": "euler", "node_steps": 4, "xi": 0})
    assert config.method == "euler"
    assert not hasattr(config, "xi")
    with pytest.raises(ValueError, match="only legacy xi: 0"):
        NodeSolverConfig.from_dict({"method": "euler", "node_steps": 4, "xi": 0.1})


def test_legacy_zero_xi_is_omitted_from_resolved_v2_config():
    data = _v2_config()
    data["node_solver"]["xi"] = 0
    config = TrainV2Config.from_dict(data)
    assert "xi" not in config_to_plain_dict(config)["node_solver"]


def test_node_guess_extends_tau_tail_to_exact_target():
    solver = NodeSolver("euler", node_steps=4, time_end=0.8)
    grid = solver.time_grid(device="cpu", dtype=torch.float64)
    states = grid.reshape(-1, 1, 1)
    velocities = torch.ones_like(states)
    mean, velocity = solver.prepare_bridge_guess(
        states, velocities, torch.tensor([[0.0]], dtype=torch.float64),
        torch.tensor([[2.0]], dtype=torch.float64), reverse=False,
        bridge_steps=10, time_grid=grid,
    )
    torch.testing.assert_close(torch.from_numpy(mean[0, :9, 0]), torch.linspace(0, 0.8, 9, dtype=torch.float64))
    torch.testing.assert_close(torch.from_numpy(mean[0, 9:, 0]), torch.tensor([1.4, 2.0], dtype=torch.float64))
    torch.testing.assert_close(torch.from_numpy(velocity[0, 9:, 0]), torch.tensor([6.0, 6.0], dtype=torch.float64))


def test_endpoint_correction_toggle_changes_initializer_terminal():
    solver = NodeSolver("euler", node_steps=4, time_end=0.8)
    grid = solver.time_grid(device="cpu", dtype=torch.float64)
    states = grid.reshape(-1, 1, 1)
    velocities = torch.ones_like(states)
    x0 = torch.tensor([[0.0]], dtype=torch.float64)
    x1 = torch.tensor([[2.0]], dtype=torch.float64)

    corrected, _ = solver.prepare_bridge_guess(
        states, velocities, x0, x1, reverse=False,
        bridge_steps=4, time_grid=grid, correct_terminal_guess=True,
    )
    raw, _ = solver.prepare_bridge_guess(
        states, velocities, x0, x1, reverse=False,
        bridge_steps=4, time_grid=grid, correct_terminal_guess=False,
    )
    assert corrected[0, -1, 0] == pytest.approx(2.0)
    assert raw[0, -1, 0] == pytest.approx(0.8)


def test_cached_targets_sample_before_tau_cutoff():
    targets = BridgeTargetSet(
        x0=torch.zeros(1024, 1), x1=torch.ones(1024, 1),
        mean=torch.zeros(1024, 3, 1), mean_velocity=torch.ones(1024, 3, 1),
        std=torch.ones(1024, 3, 1), std_velocity=torch.zeros(1024, 3, 1),
        time_grid=torch.linspace(0, 1, 3),
    )
    _, times, _ = targets.sample_batch(
        torch.arange(1024), device=torch.device("cpu"), dtype=torch.float32,
        time_end=0.7,
    )
    assert torch.all((times >= 0) & (times < 0.7))


def test_deterministic_target_remains_stable_when_state_noise_rounds_away():
    mean = torch.tensor([[10.0]], dtype=torch.float32)
    xt, target = gaussian_sample_and_target(
        mean, torch.tensor([[2.0]]), torch.tensor([[1e-8]]),
        torch.tensor([[-3.0]]), torch.tensor([[0.5]]),
    )
    torch.testing.assert_close(xt, mean)
    torch.testing.assert_close(target, torch.tensor([[0.5]]))
