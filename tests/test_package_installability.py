from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

import numpy as np
import pytest
import torch
from torch import nn

from whfm import gaussian_paths
from whfm.models import FourierTimeResidualMLP
from whfm.optimal_transport import OTPlanSampler


class LegacySiLUResBlock(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(w, w),
            nn.SiLU(),
            nn.Linear(w, w),
        )
        self.silu = nn.SiLU()

    def forward(self, x):
        return self.silu(self.net(x) + x)


class LegacyFourierTimeResidualMLP(nn.Module):
    def __init__(self, dim, out_dim=None, w=64, hidden=2, m=6, time_varying=True):
        super().__init__()
        self.dim = dim
        self.time_varying = time_varying
        self.m = m
        if out_dim is None:
            out_dim = dim

        in_dim = dim + (2 * m if time_varying else 0)
        self.input_linear = nn.Linear(in_dim, w)
        self.silu = nn.SiLU()
        self.blocks = nn.ModuleList([LegacySiLUResBlock(w) for _ in range(hidden)])
        self.output_linear = nn.Linear(w, out_dim)
        self.register_buffer("fourier_frequencies", torch.arange(1, m + 1).float())

    def time_embedding(self, t):
        frequencies = self.fourier_frequencies.to(device=t.device, dtype=t.dtype)
        angles = 2.0 * torch.pi * t * frequencies
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

    def forward(self, x):
        if self.time_varying:
            z = x[..., : self.dim]
            t = x[..., self.dim : self.dim + 1]
            x = torch.cat([z, self.time_embedding(t)], dim=-1)

        x = self.silu(self.input_linear(x))
        for block in self.blocks:
            x = block(x)
        return self.output_linear(x)


def test_fourier_model_shape_buffer_and_state_dict_compatibility():
    torch.manual_seed(12)
    legacy = LegacyFourierTimeResidualMLP(dim=3, out_dim=2, w=8, hidden=2, m=4)
    current = FourierTimeResidualMLP(dim=3, out_dim=2, w=8, hidden=2, m=4)

    assert list(current.state_dict()) == list(legacy.state_dict())
    assert "fourier_frequencies" in current.state_dict()
    assert current.fourier_frequencies.shape == (4,)

    current.load_state_dict(legacy.state_dict())
    inputs = torch.randn(5, 4)
    assert current(inputs).shape == (5, 2)
    torch.testing.assert_close(current(inputs), legacy(inputs))


def test_ot_sampler_preserves_device_and_label_alignment():
    sampler = OTPlanSampler(method="exact")
    x0 = torch.tensor([[0.0], [10.0], [20.0]])
    x1 = x0.clone()
    labels0 = x0 + 100.0
    labels1 = x1 + 200.0

    np.random.seed(9)
    paired_x0, paired_x1, paired_y0, paired_y1 = sampler.sample_plan_with_labels(
        x0, x1, labels0, labels1
    )

    assert paired_x0.shape == x0.shape
    assert paired_x1.shape == x1.shape
    assert paired_x0.device == x0.device
    assert paired_x1.device == x1.device
    torch.testing.assert_close(paired_y0, paired_x0 + 100.0)
    torch.testing.assert_close(paired_y1, paired_x1 + 200.0)


def test_ot_sampler_is_deterministic_for_seeded_sampling():
    sampler = OTPlanSampler(method="exact")
    x0 = torch.tensor([[0.0], [1.0], [2.0]])
    x1 = torch.tensor([[0.0], [1.0], [2.0]])

    np.random.seed(123)
    first = sampler.sample_plan(x0, x1)
    np.random.seed(123)
    second = sampler.sample_plan(x0, x1)

    torch.testing.assert_close(first[0], second[0])
    torch.testing.assert_close(first[1], second[1])



def test_worker_environment_uses_gnu_layer_and_restores_parent(monkeypatch):
    monkeypatch.setenv("MKL_THREADING_LAYER", "INTEL")

    with gaussian_paths._multiprocessing_worker_environment():
        assert gaussian_paths.os.environ["MKL_THREADING_LAYER"] == "GNU"

    assert gaussian_paths.os.environ["MKL_THREADING_LAYER"] == "INTEL"


@pytest.mark.parametrize("start_method", ["spawn", "forkserver"])
def test_gaussian_paths_imports_in_real_process_pool(start_method):
    if start_method not in mp.get_all_start_methods():
        pytest.skip(f"{start_method} is unavailable on this platform")

    context = mp.get_context(start_method)
    with gaussian_paths._multiprocessing_worker_environment():
        with ProcessPoolExecutor(max_workers=1, mp_context=context) as executor:
            result = executor.submit(
                gaussian_paths._mean_std_bvp_worker_init
            ).result(timeout=30)

    assert result is None
