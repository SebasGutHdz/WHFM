from __future__ import annotations

import importlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch


def _load_animate(monkeypatch):
    return importlib.import_module("whfm.evaluation.animate")


def test_resolve_checkpoint_path_defaults_to_final(monkeypatch, tmp_path):
    animate = _load_animate(monkeypatch)
    checkpoint = tmp_path / "run" / "checkpoints" / "final.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()

    assert animate.resolve_checkpoint_path(tmp_path / "run") == checkpoint


def test_resolve_checkpoint_path_accepts_stem(monkeypatch, tmp_path):
    animate = _load_animate(monkeypatch)
    checkpoint = tmp_path / "run" / "checkpoints" / "after_initial_fit.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()

    assert animate.resolve_checkpoint_path(tmp_path / "run", "after_initial_fit") == checkpoint


def test_available_model_specs_defaults_to_all_forward_first(monkeypatch):
    animate = _load_animate(monkeypatch)
    checkpoint = {
        "directions": {
            "backward": {"model": {"b": 1}, "ema": None},
            "forward": {"model": {"f": 1}, "ema": {"ema_model": {"e": 1}}},
        }
    }

    assert animate.available_model_specs(checkpoint) == [
        ("forward", "online"),
        ("forward", "ema"),
        ("backward", "online"),
    ]


def test_parser_defaults(monkeypatch):
    animate = _load_animate(monkeypatch)
    args = animate.build_parser().parse_args(["results/stunnel/run"])

    assert args.run_dir == "results/stunnel/run"
    assert args.checkpoint is None
    assert args.direction == "all"
    assert args.model_kind == "all"
    assert args.node_steps is None
    assert args.no_potential_background is False


def test_save_particle_evolution_gif_smoke(monkeypatch, tmp_path):
    pytest.importorskip("matplotlib")
    pytest.importorskip("PIL")
    animate = _load_animate(monkeypatch)

    traj = torch.tensor(
        [
            [[0.0, 0.0], [1.0, 0.0]],
            [[0.5, 0.2], [1.5, 0.2]],
            [[1.0, 0.5], [2.0, 0.5]],
        ],
        dtype=torch.float32,
    )
    eval_config = SimpleNamespace(plot_dir1=0, plot_dir2=1, plot_xlim=[0.0, 0.0], plot_ylim=[0.0, 0.0])
    potential = SimpleNamespace(has_linear=False)
    gif_path = tmp_path / "trajectory.gif"

    animate._save_particle_evolution_gif(
        gif_path=gif_path,
        tag="smoke",
        traj=traj,
        time_grid=torch.linspace(0.0, 1.0, traj.shape[0]),
        generated=traj[-1],
        reference=traj[-1],
        potential=potential,
        source_reference=traj[0],
        evaluation_config=eval_config,
        fps=4,
        dpi=60,
        potential_background=False,
    )

    assert gif_path.exists()
    assert gif_path.stat().st_size > 0


def test_dynamic_linear_contour_frame_values_use_current_reference(monkeypatch):
    animate = _load_animate(monkeypatch)

    class Potential:
        has_linear = True

        def linear_energy(self, x):
            return x[:, 0] + 2.0 * x[:, 1] + 10.0 * x[:, 2]

    traj = torch.tensor(
        [
            [[0.0, 0.0, 1.0], [2.0, 2.0, 3.0]],
            [[0.0, 0.0, 5.0], [2.0, 2.0, 7.0]],
        ],
        dtype=torch.float32,
    )
    eval_config = SimpleNamespace(plot_dir1=0, plot_dir2=1)
    grid_x = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    grid_y = torch.tensor([[3.0, 4.0]], dtype=torch.float32)

    result_0 = animate._dynamic_linear_contour_frame_values(
        Potential(),
        traj[0],
        eval_config,
        grid_x,
        grid_y,
        None,
    )
    result_1 = animate._dynamic_linear_contour_frame_values(
        Potential(),
        traj[1],
        eval_config,
        grid_x,
        grid_y,
        None,
    )

    assert result_0 is not None
    assert result_1 is not None
    values_0, clim_0 = result_0
    values_1, clim_1 = result_1
    expected_frame_0 = grid_x.numpy() + 2.0 * grid_y.numpy() + 10.0 * 2.0
    expected_frame_1 = grid_x.numpy() + 2.0 * grid_y.numpy() + 10.0 * 6.0
    assert np.array_equal(values_0, expected_frame_0)
    assert np.array_equal(values_1, expected_frame_1)
    assert not np.array_equal(values_0, values_1)
    assert np.all(np.diff(clim_0) > 0)
    assert np.all(np.diff(clim_1) > 0)


def test_save_particle_evolution_gif_with_dynamic_potential_background(monkeypatch, tmp_path):
    pytest.importorskip("matplotlib")
    pytest.importorskip("PIL")
    animate = _load_animate(monkeypatch)

    class Potential:
        has_linear = True

        def __init__(self):
            self.calls = 0

        def linear_energy(self, x):
            self.calls += 1
            return x[:, 0].pow(2) + x[:, 1].pow(2)

    from matplotlib.axes import Axes

    captured_contourf = []
    captured_contour = []
    original_contourf = Axes.contourf
    original_contour = Axes.contour

    def capture_contourf(self, *args, **kwargs):
        captured_contourf.append(
            {
                "shape": np.asarray(args[2]).shape,
                "alpha": kwargs.get("alpha"),
                "cmap": kwargs.get("cmap"),
                "zorder": kwargs.get("zorder"),
                "extend": kwargs.get("extend"),
                "levels": np.asarray(kwargs.get("levels")).shape,
            }
        )
        return original_contourf(self, *args, **kwargs)

    def capture_contour(self, *args, **kwargs):
        captured_contour.append(
            {
                "shape": np.asarray(args[2]).shape,
                "colors": kwargs.get("colors"),
                "linewidths": kwargs.get("linewidths"),
                "alpha": kwargs.get("alpha"),
                "zorder": kwargs.get("zorder"),
                "levels": np.asarray(kwargs.get("levels")).shape,
            }
        )
        return original_contour(self, *args, **kwargs)

    monkeypatch.setattr(Axes, "contourf", capture_contourf)
    monkeypatch.setattr(Axes, "contour", capture_contour)

    traj = torch.tensor(
        [
            [[0.0, 0.0], [1.0, 0.0]],
            [[0.5, 0.2], [1.5, 0.2]],
            [[1.0, 0.5], [2.0, 0.5]],
        ],
        dtype=torch.float32,
    )
    eval_config = SimpleNamespace(plot_dir1=0, plot_dir2=1, plot_xlim=[0.0, 0.0], plot_ylim=[0.0, 0.0])
    gif_path = tmp_path / "trajectory_with_potential.gif"
    potential = Potential()

    animate._save_particle_evolution_gif(
        gif_path=gif_path,
        tag="smoke",
        traj=traj,
        time_grid=torch.linspace(0.0, 1.0, traj.shape[0]),
        generated=traj[-1],
        reference=traj[-1],
        potential=potential,
        source_reference=traj[0],
        evaluation_config=eval_config,
        fps=4,
        dpi=60,
        potential_background=True,
    )

    assert gif_path.exists()
    assert gif_path.stat().st_size > 0
    assert captured_contourf
    assert captured_contour
    assert captured_contourf[-1] == {
        "shape": (80, 80),
        "alpha": 0.25,
        "cmap": animate.POTENTIAL_COLORMAP,
        "zorder": 0,
        "extend": "both",
        "levels": (35,),
    }
    assert captured_contour[-1] == {
        "shape": (80, 80),
        "colors": "0.20",
        "linewidths": 0.8,
        "alpha": 0.85,
        "zorder": 1,
        "levels": (35,),
    }
    assert potential.calls >= 2


def test_resolve_plot_domain_uses_configured_axis_limits(monkeypatch):
    _load_animate(monkeypatch)
    plotting = importlib.import_module("whfm.evaluation.plotting")
    eval_config = SimpleNamespace(
        plot_xlim=[-2.0, 3.0],
        plot_ylim=[-4.0, 5.0],
    )

    low, high = plotting._resolve_plot_domain(
        eval_config,
        torch.tensor([0.0, 1.0]),
        torch.tensor([1.0, 2.0]),
    )

    assert torch.equal(low, torch.tensor([-2.0, -4.0]))
    assert torch.equal(high, torch.tensor([3.0, 5.0]))


def test_linear_contour_values_preserve_non_particle_projection(monkeypatch):
    _load_animate(monkeypatch)
    plotting = importlib.import_module("whfm.evaluation.plotting")

    class Potential:
        def linear_energy(self, x):
            return 100.0 * x[:, 0] + 10.0 * x[:, 1] + x[:, 2]

    source_reference = torch.tensor(
        [[10.0, 20.0, 30.0], [14.0, 24.0, 34.0]],
        dtype=torch.float32,
    )
    eval_config = SimpleNamespace(plot_dir1=0, plot_dir2=2)
    grid_x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    grid_y = torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float32)

    values = plotting._linear_contour_values(
        Potential(),
        source_reference,
        eval_config,
        grid_x,
        grid_y,
        None,
    )

    expected = 100.0 * grid_x + 10.0 * 22.0 + grid_y
    assert torch.equal(values, expected)


def test_linear_contour_values_average_particle_slots(monkeypatch):
    _load_animate(monkeypatch)
    plotting = importlib.import_module("whfm.evaluation.plotting")

    class FixedCenterThreeBodyPotential:
        has_linear = True
        n_particles = 3
        particle_dim = 2

        def linear_energy(self, x):
            particles = x.reshape(x.shape[0], self.n_particles, self.particle_dim)
            weights = torch.tensor([1.0, 10.0, 100.0], device=x.device, dtype=x.dtype)
            return (particles[..., 0] * weights + particles[..., 1] * (weights + 0.5)).sum(dim=1)

    potential = FixedCenterThreeBodyPotential()
    source_reference = torch.tensor(
        [
            [0.0, 10.0, 1.0, 20.0, 2.0, 30.0],
            [4.0, 14.0, 5.0, 24.0, 6.0, 34.0],
        ],
        dtype=torch.float32,
    )
    eval_config = SimpleNamespace(plot_dir1=0, plot_dir2=1)
    grid_x = torch.tensor([[7.0, 8.0]], dtype=torch.float32)
    grid_y = torch.tensor([[9.0, 10.0]], dtype=torch.float32)
    descriptor = plotting._particle_plot_descriptor(potential, source_reference.shape[-1])

    values = plotting._linear_contour_values(
        potential,
        source_reference,
        eval_config,
        grid_x,
        grid_y,
        descriptor,
    )

    ref_particles = source_reference.mean(dim=0).reshape(3, 2)
    expected = []
    for x_coord, y_coord in zip(grid_x.reshape(-1), grid_y.reshape(-1)):
        per_particle = []
        for particle_idx in range(3):
            particles = ref_particles.clone()
            particles[particle_idx] = torch.stack([x_coord, y_coord])
            per_particle.append(potential.linear_energy(particles.reshape(1, -1))[0])
        expected.append(torch.stack(per_particle).mean())
    expected = torch.stack(expected).reshape(grid_x.shape)

    assert torch.equal(values, expected)
    particle_zero_values = []
    for x_coord, y_coord in zip(grid_x.reshape(-1), grid_y.reshape(-1)):
        particles = ref_particles.clone()
        particles[0] = torch.stack([x_coord, y_coord])
        particle_zero_values.append(potential.linear_energy(particles.reshape(1, -1))[0])
    particle_zero_values = torch.stack(particle_zero_values).reshape(grid_x.shape)

    assert not torch.equal(values, particle_zero_values)


def test_publication_plot_style_uses_larger_fonts_and_figures(monkeypatch):
    pytest.importorskip("matplotlib")
    _load_animate(monkeypatch)
    plotting = importlib.import_module("whfm.evaluation.plotting")

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, ax = plotting._publication_subplots(plt)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plotting._style_axis(ax)

    width, height = fig.get_size_inches()
    assert width > 6.4
    assert height > 4.8
    assert ax.xaxis.label.get_size() >= plotting.PUBLICATION_AXIS_LABEL_SIZE
    assert ax.get_xticklabels()[0].get_fontsize() >= plotting.PUBLICATION_TICK_SIZE
    plt.close(fig)
