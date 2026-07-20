from __future__ import annotations

import importlib

import pytest
import yaml


def _load_config_module(monkeypatch):
    return importlib.import_module("whfm.config")


def _problem_dict(functional_linear):
    return {
        "name": "stunnel",
        "dimension": 2,
        "boundaries": {
            "source": {
                "kind": "dataset",
                "name": "2d_gaussian",
                "parameters": {"mean": [-11.0, -1.0], "std": 0.5},
                "sample_noise": 0.0,
            },
            "target": {
                "kind": "dataset",
                "name": "2d_gaussian",
                "parameters": {"mean": [11.0, 1.0], "std": 0.5},
                "sample_noise": 0.0,
            },
        },
        "functional": {
            "linear": functional_linear,
            "internal": None,
            "interaction": None,
        },
        "evaluation": {
            "num_samples": 128,
            "max_metric_samples": 64,
            "num_sliced_projections": 8,
            "mmd_bandwidth": None,
            "plot_trajectory_count": 8,
            "plot_dir1": 0,
            "plot_dir2": 1,
            "xaxis_hist": 0.4,
        },
    }


def test_functional_config_accepts_short_list(monkeypatch):
    config = _load_config_module(monkeypatch)

    problem = config.ProblemConfig.from_dict(_problem_dict(["stunnel", -35.0]))

    assert problem.functional.linear == ("stunnel", -35.0, {})


def test_functional_config_accepts_old_resolved_three_item_list(monkeypatch):
    config = _load_config_module(monkeypatch)

    problem = config.ProblemConfig.from_dict(_problem_dict(["stunnel", -35.0, {}]))

    assert problem.functional.linear == ("stunnel", -35.0, {})


def test_functional_config_accepts_mapping(monkeypatch):
    config = _load_config_module(monkeypatch)

    problem = config.ProblemConfig.from_dict(
        _problem_dict(
            {
                "name": "stunnel",
                "coefficient": -35.0,
                "parameters": {},
            }
        )
    )

    assert problem.functional.linear == ("stunnel", -35.0, {})


def test_problem_config_dump_resolved_yaml_round_trips_functional_mapping(monkeypatch, tmp_path):
    config = _load_config_module(monkeypatch)
    problem = config.ProblemConfig.from_dict(_problem_dict(["stunnel", -35.0]))
    path = tmp_path / "resolved_problem.yaml"

    config.dump_resolved_yaml(path, problem)

    dumped = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert dumped["functional"]["linear"] == {
        "name": "stunnel",
        "coefficient": -35.0,
        "parameters": {},
    }
    reloaded = config.load_problem_config(path)
    assert reloaded == problem


def test_evaluation_axis_limits_default_to_auto_when_missing(monkeypatch):
    config = _load_config_module(monkeypatch)

    problem = config.ProblemConfig.from_dict(_problem_dict(["stunnel", -35.0]))

    assert problem.evaluation.plot_xlim == [0.0, 0.0]
    assert problem.evaluation.plot_ylim == [0.0, 0.0]


def test_evaluation_axis_limits_accept_nonzero_bounds(monkeypatch):
    config = _load_config_module(monkeypatch)
    data = _problem_dict(["stunnel", -35.0])
    data["evaluation"]["plot_xlim"] = [-12.0, 12.0]
    data["evaluation"]["plot_ylim"] = [-2.0, 2.0]

    problem = config.ProblemConfig.from_dict(data)

    assert problem.evaluation.plot_xlim == [-12.0, 12.0]
    assert problem.evaluation.plot_ylim == [-2.0, 2.0]


def test_evaluation_axis_limits_reject_non_increasing_bounds(monkeypatch):
    config = _load_config_module(monkeypatch)
    data = _problem_dict(["stunnel", -35.0])
    data["evaluation"]["plot_xlim"] = [1.0, 1.0]

    with pytest.raises(ValueError, match="plot_xlim"):
        config.ProblemConfig.from_dict(data)


def _train_dict_with_bridge_solver(**bridge_overrides):
    bridge_solver = {
        "kind": "scipy",
        "sigma": 0.2,
        "bridge_steps": 3,
        "tol": 1e-3,
        "max_nodes": 20,
        "quadrature_order": 1,
        "use_monte_carlo": False,
        "monte_carlo_samples": 5,
        "failure_policy": "skip_pair",
    }
    bridge_solver.update(bridge_overrides)
    return {
        "seed": 1,
        "device": "cpu",
        "dtype": "float32",
        "data": {
            "batch_size": 4,
            "num_workers": 0,
            "shuffle": True,
            "drop_last": True,
            "total_samples": 16,
            "test_fraction": 0.25,
            "n_dataset": 8,
        },
        "model": {"width": 8, "hidden": 1, "fourier_modes": 2},
        "initial_fit": {"coupling": "ot", "epochs": 1, "steps_per_batch": 1, "noise_std": 0.0},
        "rectification": {
            "num_rectifications": 1,
            "direction_order": ["forward"],
            "coupling_generation": "own_ema",
            "steps_per_batch": 1,
            "ema_target_refresh": "per_batch",
        },
        "node_solver": {"method": "euler", "node_steps": 3},
        "bridge_solver": bridge_solver,
        "optimization": {
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "scheduler": {"kind": "none", "min_lr": 0.0},
        },
        "ema": {"mode": "fixed", "decay": 0.99, "gamma": 1.0},
        "output": {
            "root": "results",
            "run_name": None,
            "checkpoint_every_direction_pass": False,
            "save_figures": False,
            "diagnostic_plot_count": 0,
        },
    }


def test_bridge_solver_num_workers_defaults_to_one(monkeypatch):
    config = _load_config_module(monkeypatch)

    train = config.TrainConfig.from_dict(_train_dict_with_bridge_solver())

    assert train.bridge_solver.num_workers == 1


def test_bridge_solver_num_workers_accepts_zero_as_serial(monkeypatch):
    config = _load_config_module(monkeypatch)

    train = config.TrainConfig.from_dict(_train_dict_with_bridge_solver(num_workers=0))

    assert train.bridge_solver.num_workers == 0


def test_bridge_solver_num_workers_rejects_negative(monkeypatch):
    config = _load_config_module(monkeypatch)

    with pytest.raises(ValueError, match="num_workers"):
        config.TrainConfig.from_dict(_train_dict_with_bridge_solver(num_workers=-1))
