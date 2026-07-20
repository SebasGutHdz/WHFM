from __future__ import annotations

import csv
import importlib
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]


def load_sweep_module():
    return importlib.import_module("whfm.sweep")


def train_dict(root: Path):
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
        "initial_fit": {
            "coupling": "independent",
            "epochs": 1,
            "steps_per_batch": 1,
            "noise_std": 0.0,
        },
        "rectification": {
            "num_rectifications": 1,
            "direction_order": ["forward"],
            "coupling_generation": "own_ema",
            "steps_per_batch": 1,
            "ema_target_refresh": "per_batch",
        },
        "node_solver": {"method": "euler", "node_steps": 3},
        "bridge_solver": {
            "kind": "scipy",
            "sigma": 0.2,
            "bridge_steps": 3,
            "tol": 1e-3,
            "max_nodes": 20,
            "quadrature_order": 1,
            "use_monte_carlo": False,
            "monte_carlo_samples": 5,
            "failure_policy": "skip_pair",
        },
        "optimization": {
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "scheduler": {"kind": "none", "min_lr": 0.0},
        },
        "ema": {"mode": "fixed", "decay": 0.99, "gamma": 1.0},
        "output": {
            "root": str(root),
            "run_name": None,
            "checkpoint_every_direction_pass": False,
            "save_figures": False,
            "diagnostic_plot_count": 0,
        },
    }


def problem_dict():
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
        "functional": {"linear": ["stunnel", -35.0], "internal": None, "interaction": None},
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


def write_yaml(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def write_configs(tmp_path: Path):
    train_path = tmp_path / "train.yaml"
    problem_path = tmp_path / "problems" / "stunnel.yaml"
    write_yaml(train_path, train_dict(tmp_path / "results"))
    write_yaml(problem_path, problem_dict())
    return train_path, problem_path


def test_sweep_config_rejects_missing_and_unknown_keys(tmp_path):
    sweep = load_sweep_module()
    train_path, problem_path = write_configs(tmp_path)

    with pytest.raises(ValueError, match="missing: gpus"):
        sweep.SweepConfig.from_dict(
            {
                "name": "x",
                "train_config": str(train_path),
                "problem_configs": [str(problem_path)],
                "seeds": [1],
            },
            base_dir=tmp_path,
        )

    with pytest.raises(ValueError, match="unknown: extra"):
        sweep.SweepConfig.from_dict(
            {
                "name": "x",
                "train_config": str(train_path),
                "problem_configs": [str(problem_path)],
                "seeds": [1],
                "gpus": [0],
                "extra": True,
            },
            base_dir=tmp_path,
        )


def test_job_expansion_builds_seed_run_names_and_valid_train_configs(tmp_path):
    sweep = load_sweep_module()
    train_path, problem_path = write_configs(tmp_path)
    config = sweep.SweepConfig.from_dict(
        {
            "name": "stat_run",
            "train_config": str(train_path),
            "problem_configs": [str(problem_path)],
            "seeds": [10, 11],
            "gpus": [2],
            "train_overrides": {"bridge_solver.num_workers": 4},
        },
        base_dir=tmp_path,
    )

    jobs = sweep.expand_jobs(config)

    assert [job.seed for job in jobs] == [10, 11]
    assert [job.effective_train["device"] for job in jobs] == ["cuda:0", "cuda:0"]
    assert [job.effective_train["cuda_device"] for job in jobs] == [0, 0]
    assert [job.effective_train["output"]["run_name"] for job in jobs] == [
        "stat_run/seed_10",
        "stat_run/seed_11",
    ]
    assert [job.effective_train["bridge_solver"]["num_workers"] for job in jobs] == [4, 4]
    assert jobs[0].run_dir == tmp_path / "results" / "stunnel" / "stat_run" / "seed_10"


def test_scheduler_assigns_cuda_masks_with_mocked_subprocess(tmp_path):
    sweep = load_sweep_module()
    train_path, problem_path = write_configs(tmp_path)
    config = sweep.SweepConfig.from_dict(
        {
            "name": "stat_run",
            "train_config": str(train_path),
            "problem_configs": [str(problem_path)],
            "seeds": [0, 1, 2],
            "gpus": [0, 1],
        },
        base_dir=tmp_path,
    )
    jobs = sweep.expand_jobs(config)
    calls = []

    class FakeProcess:
        next_pid = 100

        def __init__(self):
            self.pid = FakeProcess.next_pid
            FakeProcess.next_pid += 1

        def poll(self):
            return 0

    def fake_popen(cmd, env, stdout, stderr, text):
        calls.append({"cmd": cmd, "env": env, "text": text})
        return FakeProcess()

    records = sweep.run_scheduled_jobs(
        jobs,
        config=config,
        popen_factory=fake_popen,
        sleep_fn=lambda seconds: None,
        poll_interval=0.0,
        finalize_logs=False,
    )

    assert [call["env"]["CUDA_VISIBLE_DEVICES"] for call in calls] == ["0", "1", "0"]
    assert [call["cmd"][:3] for call in calls] == [[sweep.sys.executable, "-m", "whfm.sweep"]] * 3
    assert [call["cmd"][-2:] for call in calls] == [["--trainer", "v1"]] * 3
    assert {record["status"] for record in records.values()} == {"completed"}
    assert all(Path(job.train_config_path).is_file() for job in jobs)


def test_aggregator_computes_final_ema_statistics(tmp_path):
    sweep = load_sweep_module()
    train_path, problem_path = write_configs(tmp_path)
    config = sweep.SweepConfig.from_dict(
        {
            "name": "stat_run",
            "train_config": str(train_path),
            "problem_configs": [str(problem_path)],
            "seeds": [0, 1, 2],
            "gpus": [0],
        },
        base_dir=tmp_path,
    )
    jobs = sweep.expand_jobs(config)
    for job, value in zip(jobs[:2], [1.0, 3.0]):
        metrics_dir = job.run_dir / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        with (metrics_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "rectification_index",
                    "direction",
                    "model_kind",
                    "sliced_w2",
                    "sinkhorn",
                    "mmd",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "rectification_index": 0,
                    "direction": "forward",
                    "model_kind": "ema",
                    "sliced_w2": value + 10.0,
                    "sinkhorn": 5.0,
                    "mmd": 0.5,
                }
            )
            writer.writerow(
                {
                    "rectification_index": 1,
                    "direction": "forward",
                    "model_kind": "ema",
                    "sliced_w2": value,
                    "sinkhorn": 7.0,
                    "mmd": 0.25,
                }
            )

    summaries = sweep.aggregate_runs(jobs, ["sliced_w2"])
    rows = summaries[tmp_path / "results" / "stunnel" / "stat_run"]

    assert len(rows) == 1
    assert rows[0]["n"] == 2
    assert rows[0]["mean"] == 2.0
    assert rows[0]["std"] == pytest.approx(2**0.5)
    assert rows[0]["sem"] == pytest.approx(1.0)
    assert rows[0]["min"] == 1.0
    assert rows[0]["max"] == 3.0
    assert rows[0]["seeds"] == [0, 1]
    assert rows[0]["missing_runs"] == [
        {"job_id": "p0_stunnel_seed_2", "seed": 2, "run_dir": str(jobs[2].run_dir)}
    ]
    assert (tmp_path / "results" / "stunnel" / "stat_run" / "aggregate_metrics.csv").is_file()
    assert (tmp_path / "results" / "stunnel" / "stat_run" / "aggregate_metrics.json").is_file()



def train_v2_dict(root: Path):
    data = train_dict(root)
    data["cuda_device"] = 3
    data["data"]["drop_last"] = False
    data["rectification"] = {
        "num_rectifications": 1,
        "direction_order": ["forward"],
        "coupling_generation": "own_ema",
        "epochs": 2,
        "ema_target_refresh": "per_direction_pass",
    }
    return data


def test_sweep_config_trainer_defaults_accepts_and_rejects(tmp_path):
    sweep = load_sweep_module()
    train_path, problem_path = write_configs(tmp_path)
    base = {
        "name": "stat_run",
        "train_config": str(train_path),
        "problem_configs": [str(problem_path)],
        "seeds": [1],
        "gpus": [0],
    }

    assert sweep.SweepConfig.from_dict(base, base_dir=tmp_path).trainer == "v1"
    assert sweep.SweepConfig.from_dict({**base, "trainer": "v2"}, base_dir=tmp_path).trainer == "v2"
    with pytest.raises(ValueError, match="trainer"):
        sweep.SweepConfig.from_dict({**base, "trainer": "other"}, base_dir=tmp_path)


def test_v2_job_expansion_uses_train_v2_validation_and_gpu_normalization(monkeypatch, tmp_path):
    sweep = load_sweep_module()
    train_path = tmp_path / "train_v2.yaml"
    problem_path = tmp_path / "problems" / "stunnel.yaml"
    write_yaml(train_path, train_v2_dict(tmp_path / "results"))
    write_yaml(problem_path, problem_dict())
    calls = {"v2": [], "problem": []}

    class FakeConfigModule:
        @staticmethod
        def load_problem_config(path):
            calls["problem"].append(Path(path))
            return object()

    class FakeTrainV2Config:
        @staticmethod
        def from_dict(data):
            calls["v2"].append(data)
            return object()

    class FakeTrainerV2Module:
        TrainV2Config = FakeTrainV2Config

    def fake_load_runtime_module(name):
        if name == "config":
            return FakeConfigModule
        if name == "trainer_v2":
            return FakeTrainerV2Module
        raise AssertionError(name)

    monkeypatch.setattr(sweep, "_load_runtime_module", fake_load_runtime_module)
    config = sweep.SweepConfig.from_dict(
        {
            "name": "stat_run_v2",
            "trainer": "v2",
            "train_config": str(train_path),
            "problem_configs": [str(problem_path)],
            "seeds": [10],
            "gpus": [2],
            "train_overrides": {"bridge_solver.num_workers": 4},
        },
        base_dir=tmp_path,
    )

    jobs = sweep.expand_jobs(config)

    assert jobs[0].trainer == "v2"
    assert jobs[0].effective_train["device"] == "cuda:0"
    assert jobs[0].effective_train["cuda_device"] == 0
    assert jobs[0].effective_train["rectification"]["epochs"] == 2
    assert "steps_per_batch" not in jobs[0].effective_train["rectification"]
    assert jobs[0].effective_train["bridge_solver"]["num_workers"] == 4
    assert calls["v2"] == [jobs[0].effective_train]
    assert calls["problem"] == [problem_path]


def test_scheduler_passes_v2_trainer_to_child_process(tmp_path):
    sweep = load_sweep_module()
    train_path = tmp_path / "train_v2.yaml"
    problem_path = tmp_path / "problems" / "stunnel.yaml"
    write_yaml(train_path, train_v2_dict(tmp_path / "results"))
    write_yaml(problem_path, problem_dict())
    config = sweep.SweepConfig.from_dict(
        {
            "name": "stat_run_v2",
            "trainer": "v2",
            "train_config": str(train_path),
            "problem_configs": [str(problem_path)],
            "seeds": [0],
            "gpus": [0],
        },
        base_dir=tmp_path,
    )
    jobs = sweep.expand_jobs(config, validate=False)
    calls = []

    class FakeProcess:
        pid = 101

        def poll(self):
            return 0

    def fake_popen(cmd, env, stdout, stderr, text):
        calls.append({"cmd": cmd, "env": env})
        return FakeProcess()

    sweep.run_scheduled_jobs(
        jobs,
        config=config,
        popen_factory=fake_popen,
        sleep_fn=lambda seconds: None,
        poll_interval=0.0,
        finalize_logs=False,
    )

    assert calls[0]["cmd"][:3] == [sweep.sys.executable, "-m", "whfm.sweep"]
    assert calls[0]["cmd"][-2:] == ["--trainer", "v2"]
    assert calls[0]["env"]["CUDA_VISIBLE_DEVICES"] == "0"


def test_run_training_job_dispatches_to_v1_and_v2(monkeypatch, tmp_path):
    sweep = load_sweep_module()
    trained = []

    class FakeTrainer:
        def __init__(self, train_config, problem_config):
            self.train_config = train_config
            self.problem_config = problem_config
            self.run_dir = Path("run") / str(train_config)

        def train(self):
            trained.append((self.train_config, self.problem_config))

    class FakeConfigModule:
        @staticmethod
        def load_train_config(path):
            return "v1_train"

        @staticmethod
        def load_problem_config(path):
            return "problem"

    class FakeTrainerModule:
        HamiltonianTrainer = FakeTrainer

    class FakeTrainerV2Module:
        @staticmethod
        def load_train_v2_config(path):
            return "v2_train"

        HamiltonianTrainerV2 = FakeTrainer

    def fake_load_runtime_module(name):
        if name == "config":
            return FakeConfigModule
        if name == "trainer":
            return FakeTrainerModule
        if name == "trainer_v2":
            return FakeTrainerV2Module
        raise AssertionError(name)

    monkeypatch.setattr(sweep, "_load_runtime_module", fake_load_runtime_module)

    assert sweep.run_training_job(tmp_path / "train.yaml", tmp_path / "problem.yaml") == Path("run/v1_train")
    assert sweep.run_training_job(
        tmp_path / "train_v2.yaml",
        tmp_path / "problem.yaml",
        trainer_name="v2",
    ) == Path("run/v2_train")
    assert trained == [("v1_train", "problem"), ("v2_train", "problem")]
