"""CSV schemas and append helpers for evaluation outputs."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict


METRIC_FIELDNAMES = [
    "timestamp",
    "rectification_index",
    "direction",
    "model_kind",
    "num_eval_samples",
    "w2",
    "sliced_w2",
    "mmd2_rbf",
    "hamiltonian_drift_integral_mean",
    "hamiltonian_drift_integral_max",
    "action_mean",
    "kinetic_integral_mean",
    "potential_integral_mean",
    "terminal_mean_error",
    "terminal_cov_error",
    "terminal_displacement_mean",
    "latest_loss",
    "bridge_success_rate",
    "bridge_failed_pairs",
    "trajectory_plot",
    "linear_potential_plot",
    "terminal_scatter_plot",
    "hamiltonian_histogram_plot",
]

WARMUP_METRIC_FIELDNAMES = [
    "timestamp",
    "direction",
    "model_kind",
    "num_eval_samples",
    "w2",
    "sliced_w2",
    "mmd2_rbf",
    "terminal_mean_error",
    "terminal_cov_error",
    "terminal_displacement_mean",
    "latest_warmup_loss",
    "trajectory_plot",
    "linear_potential_plot",
    "terminal_scatter_plot",
    "sample_path",
]


def append_metrics_row(path: Path, row: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDNAMES)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in METRIC_FIELDNAMES})


def append_warmup_metrics_row(path: Path, row: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=WARMUP_METRIC_FIELDNAMES)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in WARMUP_METRIC_FIELDNAMES})
