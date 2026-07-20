from __future__ import annotations

import importlib
from pathlib import Path

import pytest
import torch


def _load_model(monkeypatch):
    return importlib.import_module("whfm.evaluation.model")


def test_rectification_residual_exact_values(monkeypatch):
    model = _load_model(monkeypatch)
    previous = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
    current = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    assert model.rectification_residual(previous, previous) == pytest.approx(0.0)
    assert model.rectification_residual(current, previous) == pytest.approx(7.0)


def test_rectification_residual_rejects_mismatched_shapes(monkeypatch):
    model = _load_model(monkeypatch)

    with pytest.raises(ValueError, match="matching terminal shapes"):
        model.rectification_residual(torch.zeros(2, 2), torch.zeros(3, 2))


def test_rectification_residual_reads_previous_terminal_file(monkeypatch, tmp_path):
    model = _load_model(monkeypatch)
    previous_path = tmp_path / "previous_terminal.pt"
    previous = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    current = torch.tensor([[1.0, 0.0], [1.0, 3.0]])
    torch.save({"generated_terminal": previous}, previous_path)

    assert torch.equal(model.load_terminal_output(previous_path), previous)
    assert model.rectification_residual_from_file(current, previous_path) == pytest.approx(2.5)
    assert model.rectification_residual_from_file(current, None) != model.rectification_residual_from_file(current, None)


def test_save_evaluation_artifacts_splits_trajectory_and_terminal(monkeypatch, tmp_path):
    model = _load_model(monkeypatch)
    time_grid = torch.linspace(0.0, 1.0, 3)
    states = torch.arange(12, dtype=torch.float32).reshape(3, 2, 2)
    start = states[0]
    terminal = states[-1]
    reference = terminal + 1.0

    paths = model.save_evaluation_artifacts(
        samples_dir=tmp_path,
        tag="warmup_epoch0_forward_online",
        stage="warmup",
        direction="forward",
        model_kind="online",
        epoch=0,
        time_grid=time_grid,
        states=states,
        start=start,
        generated_terminal=terminal,
        reference_terminal=reference,
    )

    trajectory_path = Path(paths["trajectory_path"])
    terminal_path = Path(paths["terminal_path"])
    assert trajectory_path != terminal_path
    assert trajectory_path.name == "warmup_epoch0_forward_online_trajectory.pt"
    assert terminal_path.name == "warmup_epoch0_forward_online_terminal.pt"

    trajectory_payload = torch.load(trajectory_path, map_location="cpu", weights_only=False)
    terminal_payload = torch.load(terminal_path, map_location="cpu", weights_only=False)
    assert torch.equal(trajectory_payload["states"], states)
    assert torch.equal(trajectory_payload["start"], start)
    assert torch.equal(trajectory_payload["reference_terminal"], reference)
    assert "generated_terminal" not in trajectory_payload
    assert torch.equal(terminal_payload["generated_terminal"], terminal)
    assert terminal_payload["num_eval_samples"] == 2
    assert terminal_payload["stage"] == "warmup"
    assert terminal_payload["epoch"] == 0


def test_warmup_evaluation_can_save_artifacts_without_plots(monkeypatch, tmp_path):
    from types import SimpleNamespace

    model_module = _load_model(monkeypatch)
    states = torch.tensor(
        [
            [[0.0, 0.0], [1.0, 1.0]],
            [[0.5, 0.0], [1.5, 1.0]],
            [[1.0, 0.0], [2.0, 1.0]],
        ]
    )
    traj = SimpleNamespace(time_grid=torch.linspace(0.0, 1.0, 3), states=states)
    monkeypatch.setattr(model_module, "integrate_model", lambda model, node_solver, x0: traj)
    monkeypatch.setattr(model_module, "distribution_summary", lambda *args, **kwargs: {"sliced_w2": 0.0, "sinkhorn": 0.0, "mmd": 0.0})

    def fail_if_called(*args, **kwargs):
        raise AssertionError("warmup plots should not be generated")

    monkeypatch.setattr(model_module, "save_warmup_plots", fail_if_called)

    row, terminal_path = model_module.evaluate_warmup_model_with_terminal(
        model=object(),
        model_kind="online",
        direction="forward",
        node_solver=object(),
        potential=object(),
        source_test=torch.zeros(2, 2),
        target_test=torch.ones(2, 2),
        evaluation_config=SimpleNamespace(max_metric_samples=2, num_sliced_projections=8),
        latest_warmup_loss=1.25,
        figures_dir=tmp_path / "figures",
        samples_dir=tmp_path / "samples",
        save_plots=False,
    )

    assert Path(row["trajectory_path"]).name == "warmup_final_forward_online_trajectory.pt"
    assert Path(row["terminal_path"]).name == "warmup_final_forward_online_terminal.pt"
    assert terminal_path == Path(row["terminal_path"])
    assert row["trajectory_plot"] == ""
    assert row["linear_potential_plot"] == ""
    assert row["terminal_scatter_plot"] == ""
    assert "terminal_mean_error" not in row
    assert "terminal_cov_error" not in row
    assert "terminal_displacement_mean" not in row
    assert torch.equal(model_module.load_terminal_output(terminal_path), states[-1])


def test_warmup_evaluation_can_save_only_trajectory_plot(monkeypatch, tmp_path):
    from types import SimpleNamespace

    model_module = _load_model(monkeypatch)
    states = torch.tensor(
        [
            [[0.0, 0.0], [1.0, 1.0]],
            [[0.5, 0.0], [1.5, 1.0]],
            [[1.0, 0.0], [2.0, 1.0]],
        ]
    )
    traj = SimpleNamespace(time_grid=torch.linspace(0.0, 1.0, 3), states=states)
    monkeypatch.setattr(model_module, "integrate_model", lambda model, node_solver, x0: traj)
    monkeypatch.setattr(model_module, "distribution_summary", lambda *args, **kwargs: {"sliced_w2": 0.0, "sinkhorn": 0.0, "mmd": 0.0})
    plot_calls = []

    def fake_save_warmup_plots(**kwargs):
        plot_calls.append(kwargs)
        assert kwargs["plot_mode"] == "trajectory"
        return {
            "trajectory_plot": str(tmp_path / "figures" / f"{kwargs['tag']}_trajectories.png"),
            "linear_potential_plot": "",
            "terminal_scatter_plot": "",
        }

    monkeypatch.setattr(model_module, "save_warmup_plots", fake_save_warmup_plots)

    row, terminal_path = model_module.evaluate_warmup_model_with_terminal(
        model=object(),
        model_kind="ema",
        direction="forward",
        node_solver=object(),
        potential=object(),
        source_test=torch.zeros(2, 2),
        target_test=torch.ones(2, 2),
        evaluation_config=SimpleNamespace(max_metric_samples=2, num_sliced_projections=8),
        latest_warmup_loss=1.25,
        figures_dir=tmp_path / "figures",
        samples_dir=tmp_path / "samples",
        save_plots=True,
        plot_mode="trajectory",
    )

    assert len(plot_calls) == 1
    assert Path(row["trajectory_path"]).name == "warmup_final_forward_ema_trajectory.pt"
    assert Path(row["terminal_path"]).name == "warmup_final_forward_ema_terminal.pt"
    assert Path(row["trajectory_plot"]).name == "warmup_final_forward_ema_trajectories.png"
    assert row["linear_potential_plot"] == ""
    assert row["terminal_scatter_plot"] == ""
    assert terminal_path == Path(row["terminal_path"])
    assert "terminal_mean_error" not in row
    assert "terminal_cov_error" not in row
    assert "terminal_displacement_mean" not in row


def test_save_warmup_plots_trajectory_mode_writes_only_trajectory(monkeypatch, tmp_path):
    from types import SimpleNamespace
    import importlib

    _load_model(monkeypatch)
    plotting = importlib.import_module("whfm.evaluation.plotting")
    traj = torch.tensor(
        [
            [[0.0, 0.0], [1.0, 1.0]],
            [[0.5, 0.2], [1.4, 1.2]],
            [[1.0, 0.3], [2.0, 1.3]],
        ]
    )
    cfg = SimpleNamespace(
        plot_trajectory_count=2,
        plot_dir1=0,
        plot_dir2=1,
        plot_xlim=[0.0, 0.0],
        plot_ylim=[0.0, 0.0],
    )

    paths = plotting.save_warmup_plots(
        figures_dir=tmp_path,
        tag="warmup_final_forward_online",
        traj=traj,
        time_grid=torch.linspace(0.0, 1.0, 3),
        generated=traj[-1],
        reference=traj[-1] + 0.1,
        potential=object(),
        source_reference=traj[0],
        evaluation_config=cfg,
        plot_mode="trajectory",
    )

    assert Path(paths["trajectory_plot"]).name == "warmup_final_forward_online_trajectories.png"
    assert Path(paths["trajectory_plot"]).exists()
    assert paths["linear_potential_plot"] == ""
    assert paths["terminal_scatter_plot"] == ""
    assert not (tmp_path / "warmup_final_forward_online_linear_potential.png").exists()
    assert not (tmp_path / "warmup_final_forward_online_terminal_scatter.png").exists()
