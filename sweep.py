"""Seeded multi-GPU sweep launcher for WHFM standalone experiments."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import yaml


ROOT = Path(__file__).resolve().parent
PACKAGE_NAME = "torchcfm.WHFM_standalone"
DEFAULT_AGGREGATE_METRICS = ("sliced_w2", "sinkhorn", "mmd")
METRIC_FILES = ("metrics.csv", "metrics_forward.csv", "metrics_backward.csv")


def _load_runtime_module(name: str):
    """Load WHFM-standalone modules when this file is run as a script."""
    if "torchcfm" not in sys.modules:
        torchcfm = types.ModuleType("torchcfm")
        torchcfm.__path__ = [str(ROOT.parent)]
        sys.modules["torchcfm"] = torchcfm
    if PACKAGE_NAME not in sys.modules:
        package = types.ModuleType(PACKAGE_NAME)
        package.__path__ = [str(ROOT)]
        sys.modules[PACKAGE_NAME] = package
    return importlib.import_module(f"{PACKAGE_NAME}.{name}")


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping.")
    return value


def _load_yaml(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        raise ValueError(f"{path} is empty.")
    return data


def _dump_yaml(path: str | Path, data: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def _resolve_path(path: str | Path, base_dir: Path) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    return (base_dir / value).resolve()


def _as_nonempty_string(value: Any, label: str) -> str:
    text = str(value)
    if not text:
        raise ValueError(f"{label} must be nonempty.")
    if "/" in text or "\\" in text:
        raise ValueError(f"{label} must not contain path separators.")
    return text


def _as_int_list(value: Any, label: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a nonempty list.")
    result = []
    for item in value:
        try:
            result.append(int(item))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} entries must be integers.") from exc
    return result


def _as_gpu_list(value: Any) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError("gpus must be a nonempty list.")
    result = []
    for item in value:
        text = str(item)
        if text == "cpu":
            result.append(text)
            continue
        try:
            gpu_id = int(text)
        except ValueError as exc:
            raise ValueError("gpus entries must be integer CUDA IDs or 'cpu'.") from exc
        if gpu_id < 0:
            raise ValueError("gpus entries must be nonnegative.")
        result.append(str(gpu_id))
    return result


@dataclass(frozen=True)
class SweepConfig:
    name: str
    train_config: Path
    problem_configs: tuple[Path, ...]
    seeds: tuple[int, ...]
    gpus: tuple[str, ...]
    max_parallel_per_gpu: int = 1
    train_overrides: Mapping[str, Any] = field(default_factory=dict)
    aggregate_metrics: tuple[str, ...] = DEFAULT_AGGREGATE_METRICS

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, base_dir: Path) -> "SweepConfig":
        data = _require_mapping(data, "sweep config")
        allowed = {
            "name",
            "train_config",
            "problem_configs",
            "seeds",
            "gpus",
            "max_parallel_per_gpu",
            "train_overrides",
            "aggregate_metrics",
        }
        required = {"name", "train_config", "problem_configs", "seeds", "gpus"}
        extra = set(data) - allowed
        missing = required - set(data)
        if extra or missing:
            details = []
            if missing:
                details.append("missing: " + ", ".join(sorted(missing)))
            if extra:
                details.append("unknown: " + ", ".join(sorted(extra)))
            raise ValueError(f"sweep config has invalid keys ({'; '.join(details)}).")

        problem_values = data["problem_configs"]
        if not isinstance(problem_values, list) or not problem_values:
            raise ValueError("problem_configs must be a nonempty list.")
        overrides = data.get("train_overrides", {})
        if overrides is None:
            overrides = {}
        if not isinstance(overrides, Mapping):
            raise ValueError("train_overrides must be a mapping.")
        aggregate_metrics = data.get("aggregate_metrics", list(DEFAULT_AGGREGATE_METRICS))
        if not isinstance(aggregate_metrics, list) or not aggregate_metrics:
            raise ValueError("aggregate_metrics must be a nonempty list.")
        max_parallel = int(data.get("max_parallel_per_gpu", 1))
        if max_parallel <= 0:
            raise ValueError("max_parallel_per_gpu must be positive.")

        return cls(
            name=_as_nonempty_string(data["name"], "name"),
            train_config=_resolve_path(data["train_config"], base_dir),
            problem_configs=tuple(_resolve_path(path, base_dir) for path in problem_values),
            seeds=tuple(_as_int_list(data["seeds"], "seeds")),
            gpus=tuple(_as_gpu_list(data["gpus"])),
            max_parallel_per_gpu=max_parallel,
            train_overrides=dict(overrides),
            aggregate_metrics=tuple(str(metric) for metric in aggregate_metrics),
        )


@dataclass(frozen=True)
class SweepJob:
    job_id: str
    sweep_name: str
    problem_index: int
    problem_name: str
    problem_config: Path
    seed: int
    run_name: str
    run_dir: Path
    control_dir: Path
    train_config_path: Path
    effective_train: Mapping[str, Any]


def load_sweep_config(path: str | Path) -> SweepConfig:
    path = Path(path).resolve()
    return SweepConfig.from_dict(_load_yaml(path), base_dir=path.parent)


def _set_nested(data: dict[str, Any], dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    if any(not key for key in keys):
        raise ValueError(f"invalid override path: {dotted_key!r}")
    cursor: dict[str, Any] = data
    for key in keys[:-1]:
        child = cursor.get(key)
        if not isinstance(child, dict):
            raise ValueError(f"override path {dotted_key!r} crosses a non-mapping key.")
        cursor = child
    cursor[keys[-1]] = value


def build_effective_train_dict(
    base_train: Mapping[str, Any],
    *,
    seed: int,
    sweep_name: str,
    train_overrides: Mapping[str, Any],
    gpu: str,
) -> dict[str, Any]:
    effective = json.loads(json.dumps(base_train))
    for key, value in train_overrides.items():
        _set_nested(effective, str(key), value)
    effective["seed"] = int(seed)
    effective["device"] = "cpu" if gpu == "cpu" else "cuda:0"
    output = effective.get("output")
    if not isinstance(output, dict):
        raise ValueError("train config output must be a mapping.")
    output["run_name"] = f"{sweep_name}/seed_{int(seed)}"
    return effective


def _problem_name(path: Path) -> str:
    data = _require_mapping(_load_yaml(path), f"problem config {path}")
    if "name" not in data:
        raise ValueError(f"problem config {path} is missing name.")
    return str(data["name"])


def _run_dir(train_dict: Mapping[str, Any], problem_name: str) -> Path:
    output = _require_mapping(train_dict.get("output"), "train config output")
    root = Path(str(output.get("root", "results")))
    run_name = str(output["run_name"])
    return root / problem_name / run_name


def expand_jobs(config: SweepConfig, *, validate: bool = True) -> list[SweepJob]:
    base_train = _require_mapping(_load_yaml(config.train_config), "train config")
    config_module = _load_runtime_module("config") if validate else None
    jobs: list[SweepJob] = []
    for problem_index, problem_config in enumerate(config.problem_configs):
        problem_name = _problem_name(problem_config)
        for seed_index, seed in enumerate(config.seeds):
            gpu = config.gpus[seed_index % len(config.gpus)]
            effective = build_effective_train_dict(
                base_train,
                seed=seed,
                sweep_name=config.name,
                train_overrides=config.train_overrides,
                gpu=gpu,
            )
            if validate:
                config_module.TrainConfig.from_dict(effective)
                config_module.load_problem_config(problem_config)
            run_dir = _run_dir(effective, problem_name)
            control_dir = run_dir.parent / "_sweep_control"
            train_config_path = control_dir / "configs" / f"p{problem_index}_{problem_config.stem}_seed_{seed}.yaml"
            jobs.append(
                SweepJob(
                    job_id=f"p{problem_index}_{problem_config.stem}_seed_{seed}",
                    sweep_name=config.name,
                    problem_index=problem_index,
                    problem_name=problem_name,
                    problem_config=problem_config,
                    seed=seed,
                    run_name=str(effective["output"]["run_name"]),
                    run_dir=run_dir,
                    control_dir=control_dir,
                    train_config_path=train_config_path,
                    effective_train=effective,
                )
            )
    return jobs


def _job_record(job: SweepJob, *, status: str, gpu: str | None = None) -> dict[str, Any]:
    return {
        "job_id": job.job_id,
        "sweep_name": job.sweep_name,
        "problem_index": job.problem_index,
        "problem_name": job.problem_name,
        "problem_config": str(job.problem_config),
        "seed": job.seed,
        "gpu": gpu,
        "run_name": job.run_name,
        "run_dir": str(job.run_dir),
        "train_config": str(job.train_config_path),
        "pid": None,
        "status": status,
        "exit_code": None,
        "start_time": None,
        "end_time": None,
        "stdout_path": None,
        "stderr_path": None,
    }


def _status_paths(jobs: Iterable[SweepJob]) -> list[Path]:
    paths = sorted({job.control_dir.parent / "sweep_status.json" for job in jobs})
    return paths


def _write_status(records: Mapping[str, dict[str, Any]], jobs: Iterable[SweepJob]) -> None:
    for path in _status_paths(jobs):
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            records[job.job_id]
            for job in jobs
            if job.control_dir.parent == path.parent and job.job_id in records
        ]
        with path.open("w", encoding="utf-8") as handle:
            json.dump(rows, handle, indent=2)


def _is_complete(job: SweepJob) -> bool:
    return (job.run_dir / "metrics" / "metrics.json").is_file()


def _preflight_jobs(jobs: Iterable[SweepJob], *, resume: bool) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for job in jobs:
        record = _job_record(job, status="pending")
        if resume and _is_complete(job):
            record["status"] = "skipped"
            record["exit_code"] = 0
            record["end_time"] = _timestamp()
        elif job.run_dir.exists():
            if _is_complete(job):
                raise FileExistsError(
                    f"run directory already contains completed metrics: {job.run_dir}. "
                    "Use --resume to skip completed runs."
                )
            raise FileExistsError(
                f"run directory exists but is incomplete: {job.run_dir}. "
                "Use a new sweep name or clear the incomplete run directory."
            )
        records[job.job_id] = record
    return records


def _timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _job_slots(config: SweepConfig) -> list[str]:
    slots: list[str] = []
    for gpu in config.gpus:
        slots.extend([gpu] * config.max_parallel_per_gpu)
    return slots


def _subprocess_env(gpu: str) -> dict[str, str]:
    env = dict(os.environ)
    if gpu == "cpu":
        env.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    return env


def _launch_job(
    job: SweepJob,
    *,
    gpu: str,
    popen_factory: Callable[..., subprocess.Popen] = subprocess.Popen,
):
    _dump_yaml(job.train_config_path, job.effective_train)
    log_dir = job.control_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / f"{job.job_id}.stdout.log"
    stderr_path = log_dir / f"{job.job_id}.stderr.log"
    stdout_handle = stdout_path.open("w", encoding="utf-8")
    stderr_handle = stderr_path.open("w", encoding="utf-8")
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--run-job",
        str(job.train_config_path),
        "--problem-config",
        str(job.problem_config),
    ]
    process = popen_factory(
        cmd,
        cwd=str(ROOT),
        env=_subprocess_env(gpu),
        stdout=stdout_handle,
        stderr=stderr_handle,
        text=True,
    )
    return process, stdout_handle, stderr_handle, stdout_path, stderr_path


def _finalize_logs(job: SweepJob, stdout_path: Path, stderr_path: Path) -> tuple[Path, Path]:
    final_dir = job.run_dir / "logs"
    if not final_dir.exists():
        final_dir = job.control_dir / "logs"
    final_stdout = final_dir / "stdout.log"
    final_stderr = final_dir / "stderr.log"
    if stdout_path != final_stdout:
        final_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(stdout_path, final_stdout)
    if stderr_path != final_stderr:
        final_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(stderr_path, final_stderr)
    return final_stdout, final_stderr


def run_scheduled_jobs(
    jobs: list[SweepJob],
    *,
    config: SweepConfig,
    resume: bool = False,
    popen_factory: Callable[..., subprocess.Popen] = subprocess.Popen,
    sleep_fn: Callable[[float], None] = time.sleep,
    poll_interval: float = 5.0,
    finalize_logs: bool = True,
) -> dict[str, dict[str, Any]]:
    records = _preflight_jobs(jobs, resume=resume)
    _write_status(records, jobs)
    pending = [job for job in jobs if records[job.job_id]["status"] == "pending"]
    slots = _job_slots(config)
    available = list(slots)
    active: list[dict[str, Any]] = []

    while pending or active:
        while pending and available:
            job = pending.pop(0)
            gpu = available.pop(0)
            process, stdout_handle, stderr_handle, stdout_path, stderr_path = _launch_job(
                job, gpu=gpu, popen_factory=popen_factory
            )
            record = records[job.job_id]
            record.update(
                {
                    "gpu": gpu,
                    "pid": process.pid,
                    "status": "running",
                    "start_time": _timestamp(),
                    "stdout_path": str(stdout_path),
                    "stderr_path": str(stderr_path),
                }
            )
            active.append(
                {
                    "job": job,
                    "gpu": gpu,
                    "process": process,
                    "stdout_handle": stdout_handle,
                    "stderr_handle": stderr_handle,
                    "stdout_path": stdout_path,
                    "stderr_path": stderr_path,
                }
            )
            _write_status(records, jobs)

        still_active = []
        for item in active:
            process = item["process"]
            exit_code = process.poll()
            if exit_code is None:
                still_active.append(item)
                continue
            item["stdout_handle"].close()
            item["stderr_handle"].close()
            job = item["job"]
            stdout_path = item["stdout_path"]
            stderr_path = item["stderr_path"]
            if finalize_logs:
                stdout_path, stderr_path = _finalize_logs(job, stdout_path, stderr_path)
            records[job.job_id].update(
                {
                    "status": "completed" if exit_code == 0 else "failed",
                    "exit_code": int(exit_code),
                    "end_time": _timestamp(),
                    "stdout_path": str(stdout_path),
                    "stderr_path": str(stderr_path),
                }
            )
            available.append(item["gpu"])
            _write_status(records, jobs)
        active = still_active
        if pending or active:
            sleep_fn(poll_interval)

    return records


def _as_finite_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _load_metric_rows(run_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    metrics_dir = run_dir / "metrics"
    for filename in METRIC_FILES:
        path = metrics_dir / filename
        if not path.is_file():
            continue
        with path.open("r", newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def _final_ema_rows(rows: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    best: dict[str, Mapping[str, Any]] = {}
    best_index: dict[str, int] = {}
    for row in rows:
        if row.get("model_kind") != "ema":
            continue
        direction = str(row.get("direction", ""))
        rectification = _as_int(row.get("rectification_index"))
        if not direction or rectification is None:
            continue
        if direction not in best_index or rectification >= best_index[direction]:
            best[direction] = row
            best_index[direction] = rectification
    return [best[key] for key in sorted(best)]


def _summary(values: list[float]) -> dict[str, Any]:
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": None, "std": None, "sem": None, "min": None, "max": None}
    mean = sum(values) / n
    if n > 1:
        variance = sum((value - mean) ** 2 for value in values) / (n - 1)
        std = math.sqrt(variance)
        sem = std / math.sqrt(n)
    else:
        std = None
        sem = None
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "sem": sem,
        "min": min(values),
        "max": max(values),
    }


def aggregate_runs(jobs: list[SweepJob], metrics: Iterable[str]) -> dict[Path, list[dict[str, Any]]]:
    metric_names = tuple(metrics)
    grouped_values: dict[tuple[Path, str, str], list[float]] = {}
    grouped_seeds: dict[tuple[Path, str, str], list[int]] = {}
    missing: dict[Path, list[dict[str, Any]]] = {}

    for job in jobs:
        aggregate_dir = job.control_dir.parent
        rows = _load_metric_rows(job.run_dir)
        final_rows = _final_ema_rows(rows)
        if not final_rows:
            missing.setdefault(aggregate_dir, []).append(
                {"job_id": job.job_id, "seed": job.seed, "run_dir": str(job.run_dir)}
            )
            continue
        for row in final_rows:
            direction = str(row.get("direction", ""))
            for metric in metric_names:
                value = _as_finite_float(row.get(metric))
                if value is None:
                    continue
                key = (aggregate_dir, direction, metric)
                grouped_values.setdefault(key, []).append(value)
                grouped_seeds.setdefault(key, []).append(job.seed)

    output: dict[Path, list[dict[str, Any]]] = {}
    aggregate_dirs = sorted({job.control_dir.parent for job in jobs})
    for aggregate_dir in aggregate_dirs:
        summaries = []
        for direction in sorted({key[1] for key in grouped_values if key[0] == aggregate_dir}):
            for metric in metric_names:
                key = (aggregate_dir, direction, metric)
                values = grouped_values.get(key, [])
                row = {
                    "problem_name": next(job.problem_name for job in jobs if job.control_dir.parent == aggregate_dir),
                    "sweep_name": next(job.sweep_name for job in jobs if job.control_dir.parent == aggregate_dir),
                    "direction": direction,
                    "model_kind": "ema",
                    "metric": metric,
                    "seeds": grouped_seeds.get(key, []),
                    "missing_runs": missing.get(aggregate_dir, []),
                }
                row.update(_summary(values))
                summaries.append(row)
        output[aggregate_dir] = summaries
        _write_aggregate_outputs(aggregate_dir, summaries)
    return output


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        return json.dumps(value, sort_keys=True)
    return value


def _write_aggregate_outputs(aggregate_dir: Path, rows: list[dict[str, Any]]) -> None:
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    json_path = aggregate_dir / "aggregate_metrics.json"
    csv_path = aggregate_dir / "aggregate_metrics.csv"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
    fieldnames = [
        "problem_name",
        "sweep_name",
        "direction",
        "model_kind",
        "metric",
        "n",
        "mean",
        "std",
        "sem",
        "min",
        "max",
        "seeds",
        "missing_runs",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def run_training_job(train_config_path: str | Path, problem_config_path: str | Path) -> Path:
    config_module = _load_runtime_module("config")
    trainer_module = _load_runtime_module("trainer")
    train_config = config_module.load_train_config(train_config_path)
    problem_config = config_module.load_problem_config(problem_config_path)
    trainer = trainer_module.HamiltonianTrainer(train_config, problem_config)
    trainer.train()
    print(f"Run directory: {trainer.run_dir}")
    return trainer.run_dir


def _print_dry_run(jobs: list[SweepJob], config: SweepConfig) -> None:
    print(f"Sweep: {config.name}")
    print(f"Jobs: {len(jobs)}")
    print(f"GPU slots: {', '.join(_job_slots(config))}")
    for job in jobs:
        print(
            f"{job.job_id}: problem={job.problem_name} seed={job.seed} "
            f"run_dir={job.run_dir}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run seeded WHFM sweeps across local GPUs.")
    parser.add_argument("--sweep-config", help="Path to the sweep YAML manifest.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print jobs without launching training.")
    parser.add_argument("--resume", action="store_true", help="Skip completed seed run directories.")
    parser.add_argument("--aggregate-only", action="store_true", help="Aggregate completed runs without launching training.")
    parser.add_argument("--run-job", help=argparse.SUPPRESS)
    parser.add_argument("--problem-config", help=argparse.SUPPRESS)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.run_job:
        if not args.problem_config:
            parser.error("--problem-config is required with --run-job")
        run_training_job(args.run_job, args.problem_config)
        return 0

    if not args.sweep_config:
        parser.error("--sweep-config is required")

    config = load_sweep_config(args.sweep_config)
    jobs = expand_jobs(config)
    if args.dry_run:
        _print_dry_run(jobs, config)
        return 0

    if not args.aggregate_only:
        records = run_scheduled_jobs(jobs, config=config, resume=args.resume)
        failed = [record for record in records.values() if record["status"] == "failed"]
        if failed:
            print(f"{len(failed)} sweep job(s) failed; aggregating completed runs.")
    summaries = aggregate_runs(jobs, config.aggregate_metrics)
    for aggregate_dir, rows in summaries.items():
        print(f"Aggregate rows: {len(rows)} -> {aggregate_dir / 'aggregate_metrics.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
