"""Package-local HFM trainer with cached GBVP rectification epochs."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import torch
from torch import Tensor

import tqdm as tqdm
import yaml

from .bridge import BridgeSolution
from .config import (
    BridgeSolverConfig,
    DataConfig,
    EMAConfig,
    InitialFitConfig,
    ModelConfig,
    NodeSolverConfig,
    OptimizationConfig,
    OutputConfig,
    ProblemConfig,
    RectificationConfig,
    TrainConfig,
    dump_resolved_yaml,
    load_problem_config,
)
from .directions import Direction, DirectionState
from .losses import flow_matching_loss
from .node import call_velocity_model
from .trainer import HamiltonianTrainer


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    return value


def _strict_dataclass(cls, data: Mapping[str, Any], name: str):
    data = _require_mapping(data, name)
    allowed = {field.name for field in fields(cls)}
    keys = set(data)
    missing = allowed - keys
    extra = keys - allowed
    details = []
    if missing:
        details.append("missing: " + ", ".join(sorted(missing)))
    if extra:
        details.append("unknown: " + ", ".join(sorted(extra)))
    if details:
        raise ValueError(f"{name} has invalid keys ({'; '.join(details)}).")
    return cls(**data)


def _load_yaml(path: str | Path) -> Mapping[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        raise ValueError(f"{path} is empty.")
    return _require_mapping(data, str(path))


def _resolve_cuda_device(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("cuda_device must be an integer CUDA device index or null.")
    try:
        index = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("cuda_device must be an integer CUDA device index or null.") from exc
    if index < 0:
        raise ValueError("cuda_device must be nonnegative.")
    return index


@dataclass(frozen=True)
class RectificationV2Config:
    num_rectifications: int = 1
    direction_order: List[str] = None
    coupling_generation: str = "own_ema"
    epochs: int = 1
    ema_target_refresh: str = "per_direction_pass"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RectificationV2Config":
        values = dict(_require_mapping(data, "rectification"))
        if "steps_per_batch" in values:
            raise ValueError("trainer_v2 rectification uses epochs; remove steps_per_batch.")
        if values.get("direction_order") is None:
            values["direction_order"] = ["forward"]
        if values.get("ema_target_refresh") is None:
            values["ema_target_refresh"] = "per_direction_pass"
        cfg = _strict_dataclass(cls, values, "rectification")
        if cfg.num_rectifications < 0:
            raise ValueError("rectification.num_rectifications must be nonnegative.")
        if cfg.epochs <= 0:
            raise ValueError("rectification.epochs must be positive.")
        if cfg.coupling_generation not in {"opposite_ema", "own_ema"}:
            raise ValueError("rectification.coupling_generation must be 'opposite_ema' or 'own_ema'.")
        if cfg.ema_target_refresh != "per_direction_pass":
            raise ValueError("trainer_v2 requires rectification.ema_target_refresh: per_direction_pass.")
        if any(direction not in {"forward", "backward"} for direction in cfg.direction_order):
            raise ValueError("rectification.direction_order values must be 'forward' or 'backward'.")
        if cfg.coupling_generation == "own_ema" and any(
            direction != "forward" for direction in cfg.direction_order
        ):
            raise ValueError("own_ema supports only forward direction_order entries.")
        return cfg


@dataclass(frozen=True)
class TrainV2Config:
    seed: int
    device: str
    dtype: str
    data: DataConfig
    model: ModelConfig
    initial_fit: InitialFitConfig
    rectification: RectificationV2Config
    node_solver: NodeSolverConfig
    bridge_solver: BridgeSolverConfig
    optimization: OptimizationConfig
    ema: EMAConfig
    output: OutputConfig
    cuda_device: int | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TrainV2Config":
        data = dict(_require_mapping(data, "train_v2 config"))
        data.setdefault("cuda_device", None)
        allowed = {field.name for field in fields(cls)}
        extra = set(data) - allowed
        missing = allowed - set(data)
        if extra or missing:
            details = []
            if missing:
                details.append("missing: " + ", ".join(sorted(missing)))
            if extra:
                details.append("unknown: " + ", ".join(sorted(extra)))
            raise ValueError(f"train_v2 config has invalid keys ({'; '.join(details)}).")
        dtype = str(data["dtype"])
        if dtype not in {"float32", "float64"}:
            raise ValueError("dtype must be 'float32' or 'float64'.")
        cuda_device = _resolve_cuda_device(data["cuda_device"])
        device = f"cuda:{cuda_device}" if cuda_device is not None else str(data["device"])
        cfg = cls(
            seed=int(data["seed"]),
            device=device,
            dtype=dtype,
            data=DataConfig.from_dict(data["data"]),
            model=ModelConfig.from_dict(data["model"]),
            initial_fit=InitialFitConfig.from_dict(data["initial_fit"]),
            rectification=RectificationV2Config.from_dict(data["rectification"]),
            node_solver=NodeSolverConfig.from_dict(data["node_solver"]),
            bridge_solver=BridgeSolverConfig.from_dict(data["bridge_solver"]),
            optimization=OptimizationConfig.from_dict(data["optimization"]),
            ema=EMAConfig.from_dict(data["ema"]),
            output=OutputConfig.from_dict(data["output"]),
            cuda_device=cuda_device,
        )
        if cfg.bridge_solver.bridge_steps > cfg.node_solver.node_steps:
            raise ValueError("bridge_solver.bridge_steps must be <= node_solver.node_steps.")
        return cfg


def load_train_v2_config(path: str | Path) -> TrainV2Config:
    return TrainV2Config.from_dict(_load_yaml(path))


@dataclass
class BridgeTargetSet:
    """CPU-backed cache of successful GBVP targets for one direction pass."""

    x0: Tensor
    x1: Tensor
    mean: Tensor
    mean_velocity: Tensor
    std: Tensor
    std_velocity: Tensor
    time_grid: Tensor

    @classmethod
    def from_solutions(cls, solutions: Sequence[BridgeSolution]) -> "BridgeTargetSet":
        successful = [solution for solution in solutions if solution.num_successful > 0]
        if not successful:
            raise ValueError("BridgeTargetSet requires at least one successful bridge solution.")
        time_grid = successful[0].time_grid.detach().cpu()
        for solution in successful[1:]:
            current_grid = solution.time_grid.detach().cpu()
            if current_grid.shape != time_grid.shape or not torch.allclose(current_grid, time_grid):
                raise ValueError("Cannot aggregate bridge solutions with different time grids.")
        return cls(
            x0=torch.cat([solution.x0.detach().cpu() for solution in successful], dim=0),
            x1=torch.cat([solution.x1.detach().cpu() for solution in successful], dim=0),
            mean=torch.cat([solution.mean.detach().cpu() for solution in successful], dim=0),
            mean_velocity=torch.cat(
                [solution.mean_velocity.detach().cpu() for solution in successful], dim=0
            ),
            std=torch.cat([solution.std.detach().cpu() for solution in successful], dim=0),
            std_velocity=torch.cat([solution.std_velocity.detach().cpu() for solution in successful], dim=0),
            time_grid=time_grid,
        )

    @property
    def num_pairs(self) -> int:
        return int(self.x0.shape[0])

    def num_batches(self, batch_size: int, *, drop_last: bool) -> int:
        if drop_last:
            return self.num_pairs // int(batch_size)
        return math.ceil(self.num_pairs / int(batch_size))

    def iter_batch_indices(self, batch_size: int, *, drop_last: bool):
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if drop_last:
            usable = (self.num_pairs // batch_size) * batch_size
        else:
            usable = self.num_pairs
        if usable <= 0:
            return
        perm = torch.randperm(self.num_pairs)[:usable]
        for start in range(0, usable, batch_size):
            yield perm[start : start + batch_size]

    def sample_batch(self, indices: Tensor, *, device: torch.device, dtype: torch.dtype):
        indices = indices.to(device=self.mean.device)
        mean = self.mean.index_select(0, indices).to(device=device, dtype=dtype)
        mean_velocity = self.mean_velocity.index_select(0, indices).to(device=device, dtype=dtype)
        std = self.std.index_select(0, indices).to(device=device, dtype=dtype)
        std_velocity = self.std_velocity.index_select(0, indices).to(device=device, dtype=dtype)
        time_grid = self.time_grid.to(device=device, dtype=dtype)

        t = torch.rand((mean.shape[0], 1), device=device, dtype=dtype)
        epsilon = torch.randn_like(mean[:, 0, :])
        mu_t = self._interpolate(mean, time_grid, t)
        mu_dot_t = self._interpolate(mean_velocity, time_grid, t)
        sigma_t = self._interpolate(std, time_grid, t)
        sigma_dot_t = self._interpolate(std_velocity, time_grid, t)
        if torch.any(sigma_t <= 0):
            raise RuntimeError("Cached mean/std BVP contains a nonpositive sigma value.")
        xt = mu_t + sigma_t * epsilon
        target = mu_dot_t + (sigma_dot_t / sigma_t) * (xt - mu_t)
        return xt, t, target

    @staticmethod
    def _interpolate(values: Tensor, time_grid: Tensor, t: Tensor) -> Tensor:
        t_flat = t.reshape(-1).clamp(0.0, 1.0)
        right = torch.searchsorted(time_grid, t_flat, right=False).clamp(1, time_grid.numel() - 1)
        left = right - 1
        weight = ((t_flat - time_grid[left]) / (time_grid[right] - time_grid[left]).clamp_min(1e-12))
        view_shape = (weight.shape[0],) + (1,) * (values.dim() - 2)
        weight = weight.reshape(view_shape)
        batch = torch.arange(values.shape[0], device=values.device)
        return values[batch, left] + weight * (values[batch, right] - values[batch, left])


class HamiltonianTrainerV2(HamiltonianTrainer):
    """HFM trainer that solves one cached GBVP dataset per rectification pass."""

    train_config: TrainV2Config

    def __init__(self, train_config: TrainV2Config, problem_config: ProblemConfig):
        super().__init__(train_config, problem_config)
        dump_resolved_yaml(self.run_dir / "resolved_train_v2.yaml", train_config)
        dump_resolved_yaml(self.run_dir / "resolved_train.yaml", self._v1_compatible_config(train_config))

    def run_direction_pass(self, direction: Direction, rectification_index: int) -> None:
        x_train, y_train = self._rectification_subdataset(rectification_index)
        source_loader, target_loader = self._make_loaders(x_train, y_train)
        train_state = self.states[direction]
        generation_state = self._generation_state_for(direction)
        solutions = []
        pbar = tqdm.tqdm(
            zip(source_loader, target_loader),
            desc=f"Solving cached {direction.value} GBVP targets",
        )
        for batch_index, (x_source, y_target) in enumerate(pbar):
            x_source = x_source[0].to(self.device, dtype=self.dtype)
            y_target = y_target[0].to(self.device, dtype=self.dtype)
            if self.mode == "own_ema":
                solution = self._generate_own_forward_bridge(
                    x_source, y_target, generation_state.generation_model
                )
            elif direction is Direction.FORWARD:
                solution = self._generate_forward_bridge(
                    x_source, y_target, generation_state.generation_model
                )
            else:
                solution = self._generate_backward_bridge(
                    x_source, y_target, generation_state.generation_model
                )
            self._record_bridge_metrics(direction, rectification_index, batch_index, solution)
            print("\n")
            print(f"Solved {solution.num_successful} GBVP successfully")
            if solution.num_successful == 0:
                if self.train_config.bridge_solver.failure_policy == "raise":
                    raise RuntimeError(f"{direction.value}: no successful bridge pairs.")
                continue
            solutions.append(solution)

        if not solutions:
            if self.train_config.bridge_solver.failure_policy == "raise":
                raise RuntimeError(f"{direction.value}: no successful bridge pairs.")
            print(f"{direction.value}: no successful cached GBVP targets; skipping direction pass training.")
        else:
            target_set = BridgeTargetSet.from_solutions(solutions)
            self._train_bridge_target_set(train_state, target_set)

        self.completed_passes.append(
            {
                "stage": "rectification",
                "rectification_index": rectification_index,
                "direction": direction.value,
            }
        )

    def _train_bridge_target_set(self, state: DirectionState, target_set: BridgeTargetSet) -> None:
        cfg = self.train_config
        batch_size = cfg.data.batch_size
        drop_last = cfg.data.drop_last
        batches_per_epoch = target_set.num_batches(batch_size, drop_last=drop_last)
        if batches_per_epoch <= 0:
            print(
                f"{state.direction.value}: cached {target_set.num_pairs} successful bridges, "
                "but drop_last leaves no training batches."
            )
            return
        print(
            f"{state.direction.value}: training on {target_set.num_pairs} cached GBVP targets "
            f"for {cfg.rectification.epochs} epochs ({batches_per_epoch} batches/epoch)."
        )
        state.model.train()
        for epoch in range(cfg.rectification.epochs):
            pbar = tqdm.tqdm(
                target_set.iter_batch_indices(batch_size, drop_last=drop_last),
                total=batches_per_epoch,
                desc=f"Training cached {state.direction.value} epoch {epoch}",
            )
            for indices in pbar:
                state.optimizer.zero_grad(set_to_none=True)
                xt, t, target = target_set.sample_batch(indices, device=self.device, dtype=self.dtype)
                prediction = call_velocity_model(state.model, xt, t)
                loss = flow_matching_loss(prediction, target)
                self._check_finite(loss, "cached bridge loss")
                loss.backward()
                state.optimizer.step()
                state.scheduler_step()
                if state.ema is not None:
                    state.ema.update(state.model)
                self.metrics["losses"][state.direction.value].append(float(loss.detach().cpu()))

    def _rectification_steps_for_direction(self, direction: Direction) -> int:
        rect_cfg = self.train_config.rectification
        rect_batches = self._num_batches(self.train_config.data.n_dataset)
        rect_occurrences = sum(
            1 for value in rect_cfg.direction_order if value == direction.value
        )
        return max(1, rect_batches * rect_occurrences * rect_cfg.epochs)

    def _estimate_total_steps(self, directions) -> Dict[Direction, int]:
        cfg = self.train_config
        train_count = self._train_count()
        warmup_batches = self._num_batches(train_count)
        rect_batches = self._num_batches(cfg.data.n_dataset)
        totals = {}
        for direction in directions:
            warmup_steps = warmup_batches * cfg.initial_fit.epochs * cfg.initial_fit.steps_per_batch
            rect_occurrences = sum(
                1 for value in cfg.rectification.direction_order if value == direction.value
            )
            rect_steps = (
                rect_batches
                * cfg.rectification.num_rectifications
                * rect_occurrences
                * cfg.rectification.epochs
            )
            totals[direction] = max(1, warmup_steps + rect_steps)
        return totals

    @staticmethod
    def _v1_compatible_config(config: TrainV2Config) -> TrainConfig:
        return TrainConfig(
            seed=config.seed,
            device=config.device,
            dtype=config.dtype,
            data=config.data,
            model=config.model,
            initial_fit=config.initial_fit,
            rectification=RectificationConfig(
                num_rectifications=config.rectification.num_rectifications,
                direction_order=list(config.rectification.direction_order),
                coupling_generation=config.rectification.coupling_generation,
                steps_per_batch=config.rectification.epochs,
                ema_target_refresh=config.rectification.ema_target_refresh,
            ),
            node_solver=config.node_solver,
            bridge_solver=config.bridge_solver,
            optimization=config.optimization,
            ema=config.ema,
            output=config.output,
            cuda_device=config.cuda_device,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Hamiltonian Flow Matching v2.")
    parser.add_argument("--train-config", required=True, help="Path to the v2 training YAML config.")
    parser.add_argument("--problem-config", required=True, help="Path to the problem YAML config.")
    parser.add_argument("--cuda-device", type=int, default=None, help="CUDA device index to use, overriding the training YAML.")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    train_config = load_train_v2_config(args.train_config)
    if args.cuda_device is not None:
        if args.cuda_device < 0:
            raise ValueError("--cuda-device must be nonnegative.")
        train_config = replace(
            train_config,
            device=f"cuda:{args.cuda_device}",
            cuda_device=args.cuda_device,
        )
    problem_config = load_problem_config(args.problem_config)
    trainer = HamiltonianTrainerV2(train_config, problem_config)
    trainer.train()
    print(f"Run directory: {trainer.run_dir}")


if __name__ == "__main__":
    main()
