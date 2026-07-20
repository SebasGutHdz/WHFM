"""Strict configuration loading for Hamiltonian flow matching training."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import yaml


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


def _plain(value: Any) -> Any:
    if value.__class__.__name__ == "FunctionalConfig":
        return {
            "linear": _plain_functional_component(value.linear),
            "internal": _plain_functional_component(value.internal),
            "interaction": _plain_functional_component(value.interaction),
        }
    if is_dataclass(value):
        return {field.name: _plain(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    if isinstance(value, list):
        return [_plain(item) for item in value]
    return value


def config_to_plain_dict(config: Any) -> Dict[str, Any]:
    """Return a YAML-compatible plain dictionary for a config dataclass."""

    plain = _plain(config)
    if not isinstance(plain, Mapping):
        raise TypeError("config_to_plain_dict expects a dataclass or mapping config.")
    return dict(plain)


def _plain_functional_component(value: Any) -> Optional[Dict[str, Any]]:
    if value is None:
        return None
    if not isinstance(value, tuple) or len(value) not in {2, 3}:
        return _plain(value)
    component_name, coefficient = value[:2]
    parameters = value[2] if len(value) == 3 else {}
    return {
        "name": _plain(component_name),
        "coefficient": _plain(coefficient),
        "parameters": _plain({} if parameters is None else parameters),
    }


def _resolve_cuda_device(value: Any) -> Optional[int]:
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
class DataConfig:
    batch_size: int = 256
    num_workers: int = 0
    shuffle: bool = True
    drop_last: bool = True
    total_samples: int = 10240
    test_fraction: float = 0.2
    n_dataset: int = 2048

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DataConfig":
        cfg = _strict_dataclass(cls, data, "data")
        if cfg.batch_size <= 0:
            raise ValueError("data.batch_size must be positive.")
        if cfg.total_samples <= 1 or cfg.n_dataset <= 0:
            raise ValueError("data.total_samples must exceed 1 and data.n_dataset must be positive.")
        if not 0.0 <= cfg.test_fraction < 1.0:
            raise ValueError("data.test_fraction must be in [0, 1).")
        if cfg.num_workers < 0:
            raise ValueError("data.num_workers must be nonnegative.")
        train_count = int(round(cfg.total_samples * (1.0 - cfg.test_fraction)))
        if train_count <= 0:
            raise ValueError("data.test_fraction leaves no training samples.")
        if cfg.n_dataset > train_count:
            raise ValueError("data.n_dataset must be no larger than the training split size.")
        return cfg


@dataclass(frozen=True)
class ModelConfig:
    width: int = 256
    hidden: int = 4
    fourier_modes: int = 6

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ModelConfig":
        cfg = _strict_dataclass(cls, data, "model")
        if cfg.width <= 0 or cfg.hidden < 0 or cfg.fourier_modes <= 0:
            raise ValueError("model width, hidden, and fourier_modes must be valid positive sizes.")
        return cfg


@dataclass(frozen=True)
class InitialFitConfig:
    coupling: str = "ot"
    epochs: int = 1
    steps_per_batch: int = 1
    noise_std: float = 1.0e-3

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "InitialFitConfig":
        cfg = _strict_dataclass(cls, data, "initial_fit")
        if cfg.coupling not in {"independent", "ot"}:
            raise ValueError("initial_fit.coupling must be 'independent' or 'ot'.")
        if cfg.epochs < 0 or cfg.steps_per_batch <= 0:
            raise ValueError("initial_fit.epochs must be nonnegative and steps_per_batch positive.")
        if cfg.noise_std < 0.0:
            raise ValueError("initial_fit.noise_std must be nonnegative.")
        return cfg


@dataclass(frozen=True)
class RectificationConfig:
    num_rectifications: int = 1
    direction_order: List[str] = None
    coupling_generation: str = "own_ema"
    steps_per_batch: int = 1
    ema_target_refresh: str = "per_batch"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RectificationConfig":
        values = dict(_require_mapping(data, "rectification"))
        if values.get("direction_order") is None:
            values["direction_order"] = ["forward"]
        cfg = _strict_dataclass(cls, values, "rectification")
        if cfg.num_rectifications < 0:
            raise ValueError("rectification.num_rectifications must be nonnegative.")
        if cfg.steps_per_batch <= 0:
            raise ValueError("rectification.steps_per_batch must be positive.")
        if cfg.coupling_generation not in {"opposite_ema", "own_ema"}:
            raise ValueError("rectification.coupling_generation must be 'opposite_ema' or 'own_ema'.")
        if cfg.ema_target_refresh not in {"per_batch", "per_direction_pass"}:
            raise ValueError(
                "rectification.ema_target_refresh must be 'per_batch' or 'per_direction_pass'."
            )
        if any(direction not in {"forward", "backward"} for direction in cfg.direction_order):
            raise ValueError("rectification.direction_order values must be 'forward' or 'backward'.")
        if cfg.coupling_generation == "own_ema" and any(
            direction != "forward" for direction in cfg.direction_order
        ):
            raise ValueError("own_ema supports only forward direction_order entries.")
        return cfg


@dataclass(frozen=True)
class NodeSolverConfig:
    method: str = "euler"
    node_steps: int = 100

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "NodeSolverConfig":
        cfg = _strict_dataclass(cls, data, "node_solver")
        if cfg.method not in {"euler", "rk4"}:
            raise ValueError("node_solver.method must be 'euler' or 'rk4'.")
        if cfg.node_steps <= 0:
            raise ValueError("node_solver.node_steps must be positive.")
        return cfg


@dataclass(frozen=True)
class BridgeSolverConfig:
    kind: str = "scipy"
    sigma: Optional[float] = 1e-2
    sigma_source: Optional[float] = None
    sigma_target: Optional[float] = None
    bridge_steps: int = 30
    tol: float = 1e-2
    max_nodes: int = 1000
    quadrature_order: int = 4
    use_monte_carlo: bool = False
    monte_carlo_samples: int = 100
    n_density_samples: int = 1
    n_reference_grid: Optional[int] = None
    entropy_density_std_floor: Optional[float] = None
    num_workers: int = 1
    failure_policy: str = "skip_pair"

    def __post_init__(self):
        if (self.sigma_source is None) != (self.sigma_target is None):
            raise ValueError(
                "bridge_solver.sigma_source and bridge_solver.sigma_target must be provided together."
            )
        if self.sigma_source is None:
            if self.sigma is None:
                raise ValueError(
                    "bridge_solver requires either sigma or both sigma_source and sigma_target."
                )
            sigma = float(self.sigma)
            sigma_source = sigma
            sigma_target = sigma
            object.__setattr__(self, "sigma", sigma)
        else:
            sigma_source = float(self.sigma_source)
            sigma_target = float(self.sigma_target)
            if self.sigma is not None:
                object.__setattr__(self, "sigma", float(self.sigma))
        if sigma_source <= 0.0 or sigma_target <= 0.0:
            raise ValueError("bridge_solver sigma endpoints must be positive.")
        if self.entropy_density_std_floor is not None:
            entropy_density_std_floor = float(self.entropy_density_std_floor)
            if entropy_density_std_floor <= 0.0:
                raise ValueError("bridge_solver.entropy_density_std_floor must be positive when provided.")
            object.__setattr__(self, "entropy_density_std_floor", entropy_density_std_floor)
        object.__setattr__(self, "sigma_source", sigma_source)
        object.__setattr__(self, "sigma_target", sigma_target)


    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BridgeSolverConfig":
        data = dict(data)
        data.setdefault("sigma", None)
        data.setdefault("sigma_source", None)
        data.setdefault("sigma_target", None)
        data.setdefault("use_monte_carlo", False)
        data.setdefault("monte_carlo_samples", 100)
        data.setdefault("n_density_samples", 1)
        data.setdefault("n_reference_grid", None)
        data.setdefault("entropy_density_std_floor", None)
        data.setdefault("num_workers", 1)
        cfg = _strict_dataclass(cls, data, "bridge_solver")
        if cfg.kind != "scipy":
            raise ValueError("bridge_solver.kind currently supports only 'scipy'.")
        if cfg.bridge_steps <= 0 or cfg.max_nodes <= 0 or cfg.quadrature_order <= 0:
            raise ValueError("bridge_solver step, node, and quadrature sizes must be positive.")
        if cfg.monte_carlo_samples <= 0:
            raise ValueError("bridge_solver.monte_carlo_samples must be positive.")
        if cfg.n_density_samples <= 0:
            raise ValueError("bridge_solver.n_density_samples must be positive.")
        if cfg.n_reference_grid is not None and cfg.n_reference_grid < 2:
            raise ValueError("bridge_solver.n_reference_grid must be at least 2 when provided.")
        if cfg.entropy_density_std_floor is not None and cfg.entropy_density_std_floor <= 0.0:
            raise ValueError("bridge_solver.entropy_density_std_floor must be positive when provided.")
        if cfg.num_workers < 0:
            raise ValueError("bridge_solver.num_workers must be nonnegative.")
        if cfg.tol <= 0.0:
            raise ValueError("bridge_solver.tol must be positive.")
        if cfg.failure_policy not in {"skip_pair", "raise"}:
            raise ValueError("bridge_solver.failure_policy must be 'skip_pair' or 'raise'.")
        return cfg


@dataclass(frozen=True)
class SchedulerConfig:
    kind: str = "cosine"
    min_lr: float = 0.0

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SchedulerConfig":
        cfg = _strict_dataclass(cls, data, "optimization.scheduler")
        if cfg.kind not in {"cosine", "linear", "none"}:
            raise ValueError("optimization.scheduler.kind must be 'cosine', 'linear', or 'none'.")
        if cfg.min_lr < 0.0:
            raise ValueError("optimization.scheduler.min_lr must be nonnegative.")
        return cfg


@dataclass(frozen=True)
class OptimizationConfig:
    learning_rate: float = 5e-5
    weight_decay: float = 0.0
    scheduler: SchedulerConfig = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "OptimizationConfig":
        values = dict(_require_mapping(data, "optimization"))
        if values.get("scheduler") is None:
            values["scheduler"] = {"kind": "cosine", "min_lr": 0.0}
        values["scheduler"] = SchedulerConfig.from_dict(values["scheduler"])
        cfg = _strict_dataclass(cls, values, "optimization")
        if cfg.learning_rate <= 0.0 or cfg.weight_decay < 0.0:
            raise ValueError("optimization.learning_rate must be positive and weight_decay nonnegative.")
        if cfg.scheduler.min_lr > cfg.learning_rate:
            raise ValueError("optimization.scheduler.min_lr must not exceed learning_rate.")
        return cfg


@dataclass(frozen=True)
class EMAConfig:
    mode: str = "fixed"
    decay: float = 0.995
    gamma: float = 6.99

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EMAConfig":
        cfg = _strict_dataclass(cls, data, "ema")
        if cfg.mode not in {"fixed", "posthoc"}:
            raise ValueError("ema.mode must be 'fixed' or 'posthoc'.")
        if not 0.0 <= cfg.decay < 1.0:
            raise ValueError("ema.decay must be in [0, 1).")
        if cfg.gamma <= 0.0:
            raise ValueError("ema.gamma must be positive.")
        return cfg


@dataclass(frozen=True)
class OutputConfig:
    root: str = "results"
    run_name: Optional[str] = None
    checkpoint_every_direction_pass: bool = True
    save_figures: bool = False
    diagnostic_plot_count: int = 24

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "OutputConfig":
        cfg = _strict_dataclass(cls, data, "output")
        if cfg.diagnostic_plot_count < 0:
            raise ValueError("output.diagnostic_plot_count must be nonnegative.")
        return cfg


@dataclass(frozen=True)
class TrainConfig:
    seed: int
    device: str
    dtype: str
    data: DataConfig
    model: ModelConfig
    initial_fit: InitialFitConfig
    rectification: RectificationConfig
    node_solver: NodeSolverConfig
    bridge_solver: BridgeSolverConfig
    optimization: OptimizationConfig
    ema: EMAConfig
    output: OutputConfig
    cuda_device: Optional[int] = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TrainConfig":
        data = dict(_require_mapping(data, "train config"))
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
            raise ValueError(f"train config has invalid keys ({'; '.join(details)}).")
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
            rectification=RectificationConfig.from_dict(data["rectification"]),
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


@dataclass(frozen=True)
class BoundaryConfig:
    kind: str
    name: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    sample_noise: float = 0.0
    mean: Optional[List[float]] = None
    std: Optional[float] = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], name: str) -> "BoundaryConfig":
        data = _require_mapping(data, name)
        allowed = {field.name for field in fields(cls)}
        extra = set(data) - allowed
        if extra:
            raise ValueError(f"{name} has unknown keys: {', '.join(sorted(extra))}.")
        kind = str(data.get("kind"))
        if kind not in {"dataset", "gaussian"}:
            raise ValueError(f"{name}.kind must be 'dataset' or 'gaussian'.")
        parameters = dict(data.get("parameters") or {})
        sample_noise = float(data.get("sample_noise", 0.0))
        if sample_noise < 0.0:
            raise ValueError(f"{name}.sample_noise must be nonnegative.")
        if kind == "gaussian":
            if "mean" not in data or "std" not in data:
                raise ValueError(f"{name} gaussian boundaries require mean and std.")
            mean = list(data["mean"])
            std = float(data["std"])
            if std <= 0.0:
                raise ValueError(f"{name}.std must be positive.")
            parameters.update({"mean": mean, "std": std})
            return cls(
                kind="gaussian",
                name="2d_gaussian",
                parameters=parameters,
                sample_noise=sample_noise,
                mean=mean,
                std=std,
            )
        dataset_name = data.get("name")
        if not isinstance(dataset_name, str) or not dataset_name:
            raise ValueError(f"{name}.name must be a dataset name string.")
        return cls(
            kind="dataset",
            name=dataset_name,
            parameters=parameters,
            sample_noise=sample_noise,
            mean=None,
            std=None,
        )


@dataclass(frozen=True)
class BoundariesConfig:
    source: BoundaryConfig
    target: BoundaryConfig

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BoundariesConfig":
        data = _require_mapping(data, "boundaries")
        allowed = {"source", "target"}
        if set(data) != allowed:
            raise ValueError("boundaries must contain exactly source and target.")
        return cls(
            source=BoundaryConfig.from_dict(data["source"], "boundaries.source"),
            target=BoundaryConfig.from_dict(data["target"], "boundaries.target"),
        )


@dataclass(frozen=True)
class FunctionalConfig:
    linear: Optional[Any]
    internal: Optional[Any]
    interaction: Optional[Any]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FunctionalConfig":
        data = _require_mapping(data, "functional")
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
            raise ValueError(f"functional has invalid keys ({'; '.join(details)}).")
        return cls(
            linear=_parse_functional_component(data["linear"], "functional.linear"),
            internal=_parse_functional_component(data["internal"], "functional.internal"),
            interaction=_parse_functional_component(data["interaction"], "functional.interaction"),
        )

    def to_potential_cfg(self) -> Dict[str, Optional[tuple]]:
        return {
            "linear": self.linear,
            "internal": self.internal,
            "interaction": self.interaction,
        }


def _parse_functional_component(value: Any, name: str) -> Optional[tuple]:
    if value is None:
        return None
    if isinstance(value, list):
        if len(value) not in {2, 3}:
            raise ValueError(f"{name} list form must be [name, coefficient] or [name, coefficient, parameters].")
        component_name, coefficient = value[:2]
        parameters = value[2] if len(value) == 3 else {}
        if parameters is None:
            parameters = {}
        if not isinstance(parameters, Mapping):
            raise TypeError(f"{name}[2] parameters must be a mapping.")
        parameters = dict(parameters)
    elif isinstance(value, Mapping):
        allowed = {"name", "coefficient", "parameters"}
        keys = set(value)
        missing = {"name", "coefficient"} - keys
        extra = keys - allowed
        details = []
        if missing:
            details.append("missing: " + ", ".join(sorted(missing)))
        if extra:
            details.append("unknown: " + ", ".join(sorted(extra)))
        if details:
            raise ValueError(f"{name} has invalid keys ({'; '.join(details)}).")
        component_name = value["name"]
        coefficient = value["coefficient"]
        parameters = value.get("parameters", None)
        if parameters is None:
            parameters = {}
        if not isinstance(parameters, Mapping):
            raise TypeError(f"{name}.parameters must be a mapping.")
        parameters = dict(parameters)
    else:
        raise ValueError(f"{name} must be null, a two-item list, or a mapping.")
    if not isinstance(component_name, str) or not component_name:
        raise TypeError(f"{name}.name must be a nonempty string.")
    if isinstance(coefficient, bool):
        raise TypeError(f"{name}.coefficient must be numeric.")
    try:
        coefficient_value = float(coefficient)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name}.coefficient must be numeric.") from exc
    return (component_name, coefficient_value, parameters)


@dataclass(frozen=True)
class EvaluationConfig:
    num_samples: int = 3000
    max_metric_samples: int = 512
    num_sliced_projections: int = 128
    mmd_bandwidth: Optional[float] = None
    plot_trajectory_count: int = 64
    plot_dir1: int = 0
    plot_dir2: int = 1
    plot_xlim: List[float] = field(default_factory=lambda: [0.0, 0.0])
    plot_ylim: List[float] = field(default_factory=lambda: [0.0, 0.0])
    xaxis_hist: float = 0.4

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EvaluationConfig":
        values = dict(_require_mapping(data, "evaluation"))
        values.setdefault("plot_xlim", [0.0, 0.0])
        values.setdefault("plot_ylim", [0.0, 0.0])
        values["plot_xlim"] = _parse_axis_limit(values["plot_xlim"], "evaluation.plot_xlim")
        values["plot_ylim"] = _parse_axis_limit(values["plot_ylim"], "evaluation.plot_ylim")
        cfg = _strict_dataclass(cls, values, "evaluation")
        if cfg.num_samples <= 0 or cfg.max_metric_samples <= 0:
            raise ValueError("evaluation num_samples and max_metric_samples must be positive.")
        if cfg.num_sliced_projections <= 0:
            raise ValueError("evaluation.num_sliced_projections must be positive.")
        if cfg.mmd_bandwidth is not None and cfg.mmd_bandwidth <= 0.0:
            raise ValueError("evaluation.mmd_bandwidth must be null or positive.")
        if cfg.plot_trajectory_count < 0:
            raise ValueError("evaluation.plot_trajectory_count must be nonnegative.")
        if cfg.plot_dir1 < 0 or cfg.plot_dir2 < 0:
            raise ValueError("evaluation plot directions must be nonnegative indices.")
        return cfg


def _parse_axis_limit(value: Any, name: str) -> List[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must be a two-item list [min, max].")
    if isinstance(value[0], bool) or isinstance(value[1], bool):
        raise TypeError(f"{name} values must be numeric.")
    try:
        low = float(value[0])
        high = float(value[1])
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} values must be numeric.") from exc
    if (low, high) != (0.0, 0.0) and not low < high:
        raise ValueError(f"{name} must be [0.0, 0.0] for auto limits or satisfy min < max.")
    return [low, high]


@dataclass(frozen=True)
class ProblemConfig:
    name: str
    dimension: int
    boundaries: BoundariesConfig
    functional: FunctionalConfig
    evaluation: EvaluationConfig

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ProblemConfig":
        data = _require_mapping(data, "problem config")
        allowed = {field.name for field in fields(cls)}
        extra = set(data) - allowed
        missing = allowed - set(data)
        if extra or missing:
            details = []
            if missing:
                details.append("missing: " + ", ".join(sorted(missing)))
            if extra:
                details.append("unknown: " + ", ".join(sorted(extra)))
            raise ValueError(f"problem config has invalid keys ({'; '.join(details)}).")
        cfg = cls(
            name=str(data["name"]),
            dimension=int(data["dimension"]),
            boundaries=BoundariesConfig.from_dict(data["boundaries"]),
            functional=FunctionalConfig.from_dict(data["functional"]),
            evaluation=EvaluationConfig.from_dict(data["evaluation"]),
        )
        if cfg.dimension <= 0:
            raise ValueError("dimension must be positive.")
        if cfg.evaluation.plot_dir1 >= cfg.dimension or cfg.evaluation.plot_dir2 >= cfg.dimension:
            raise ValueError("evaluation plot directions must be within [0, dimension).")
        if cfg.evaluation.plot_dir1 == cfg.evaluation.plot_dir2 and cfg.dimension != 1:
            raise ValueError("evaluation plot directions must be distinct unless dimension is 1.")
        for label, boundary in (("source", cfg.boundaries.source), ("target", cfg.boundaries.target)):
            if boundary.mean is not None and len(boundary.mean) != cfg.dimension:
                raise ValueError(f"boundaries.{label}.mean length must match dimension.")
            param_dim = (boundary.parameters or {}).get("dim")
            if param_dim is not None and int(param_dim) != cfg.dimension:
                raise ValueError(f"boundaries.{label}.parameters.dim must match dimension.")
        return cfg


def load_train_config(path: str | Path) -> TrainConfig:
    return TrainConfig.from_dict(_load_yaml(path))


def load_problem_config(path: str | Path) -> ProblemConfig:
    return ProblemConfig.from_dict(_load_yaml(path))


def dump_resolved_yaml(path: str | Path, config: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(_plain(config), handle, sort_keys=False)
