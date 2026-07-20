from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

def load_evaluation_module(name: str):
    return importlib.import_module(f"whfm.evaluation.{name}")


def test_metric_fieldnames_exclude_terminal_error_metrics():
    io = load_evaluation_module("io")
    removed = {
        "terminal_mean_error",
        "terminal_cov_error",
        "terminal_displacement_mean",
    }
    assert removed.isdisjoint(io.METRIC_FIELDNAMES)
    assert removed.isdisjoint(io.WARMUP_METRIC_FIELDNAMES)


def test_metric_series_plots_all_configured_metrics(tmp_path):
    pytest.importorskip("matplotlib")
    postprocess = load_evaluation_module("postprocess")
    rows = []
    for rectification in range(2):
        rows.append(
            {
                "rectification_index": rectification,
                "direction": "forward",
                "model_kind": "ema",
                "sliced_w2": 2.0 - rectification,
                "hamiltonian_drift_integral_mean": 0.1 + rectification,
                "action_mean": 1.0 + rectification,
                "kinetic_integral_mean": 2.0 + rectification,
                "potential_integral_mean": 0.5 + rectification,
                "linear_potential_integral_mean": 0.4 + rectification,
                "rectification_residual": 0.25 + rectification,
                "latest_loss": 0.75 + rectification,
            }
        )

    paths = postprocess.plot_rectification_metric_series(tmp_path, rows)

    assert set(paths) == set(postprocess.RECTIFICATION_SERIES_METRICS)
    for path in paths.values():
        assert Path(path).is_file()
        assert Path(path).suffix == ".png"


def test_metric_ylabels_use_math_notation():
    postprocess = load_evaluation_module("postprocess")
    assert postprocess.METRIC_YLABELS == {
        "sliced_w2": r"$SW_2(\nu,\tilde{\nu})$",
        "hamiltonian_drift_integral_mean": r"$\mathbb{E}\,\mathcal{D}_H[Z]$",
        "action_mean": r"$\mathbb{E}\,\mathcal{A}[Z]$",
        "kinetic_integral_mean": r"$\int_0^1 \mathbb{E}_{x\sim\mu}\left[\frac{1}{2}\|Z_t'(x)\|^2\right]\,dt$",
        "potential_integral_mean": r"$\int_0^1 \mathcal{F}[\rho_t]\,dt$",
        "linear_potential_integral_mean": r"$\mathbb{E}\int_0^1 V(Z_t(x))\,dt$",
        "rectification_residual": r"$\|\pi-\mathcal{R}^H(\pi)\|$",
        "latest_loss": r"$\mathcal{L}_{CFM}$",
    }


def test_residual_sample_plot_uses_terminal_artifacts(tmp_path):
    pytest.importorskip("matplotlib")
    postprocess = load_evaluation_module("postprocess")
    sample_dir = tmp_path / "samples"
    sample_dir.mkdir()
    previous_path = sample_dir / "previous.pt"
    current_path = sample_dir / "current.pt"
    torch.save({"generated_terminal": torch.tensor([[0.0, 0.0], [1.0, 1.0]])}, previous_path)
    torch.save({"generated_terminal": torch.tensor([[0.5, 0.0], [1.5, 1.0]])}, current_path)
    rows = [
        {
            "rectification_index": 1,
            "direction": "forward",
            "model_kind": "ema",
            "terminal_path": str(current_path),
            "previous_terminal_path": str(previous_path),
        }
    ]
    evaluation_config = SimpleNamespace(
        plot_trajectory_count=2,
        plot_dir1=0,
        plot_dir2=1,
        plot_xlim=[0.0, 0.0],
        plot_ylim=[0.0, 0.0],
    )

    paths = postprocess.plot_rectification_residual_samples(
        tmp_path,
        rows,
        potential=object(),
        evaluation_config=evaluation_config,
    )

    assert list(paths) == ["r1_forward_ema"]
    assert Path(paths["r1_forward_ema"]).is_file()
    assert Path(paths["r1_forward_ema"]).parent.name == "rectification_residual_samples"


def test_best_metric_filter_ignores_non_finite_values():
    postprocess = load_evaluation_module("postprocess")
    values = [float("nan"), "", None, "inf", 0.4, "0.3"]
    finite = [postprocess._as_finite_float(value) for value in values]
    assert finite == [None, None, None, None, 0.4, 0.3]
    assert min(value for value in finite if value is not None) == 0.3
