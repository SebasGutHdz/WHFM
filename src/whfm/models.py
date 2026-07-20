"""Neural velocity models used by WHFM trainers and evaluation tools."""

from __future__ import annotations

import torch
import torch.nn as nn

class SiLUResBlock(nn.Module):
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


class FourierTimeResidualMLP(nn.Module):
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
        self.blocks = nn.ModuleList([SiLUResBlock(w) for _ in range(hidden)])
        self.output_linear = nn.Linear(w, out_dim)
        self.register_buffer("fourier_frequencies", torch.arange(1, m + 1).float())

    def time_embedding(self, t):
        frequencies = self.fourier_frequencies.to(device=t.device, dtype=t.dtype)
        angles = 2.0 * torch.pi * t * frequencies
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

    def forward(self, x):
        if self.time_varying:
            z = x[..., :self.dim]
            t = x[..., self.dim:self.dim + 1]
            x = torch.cat([z, self.time_embedding(t)], dim=-1)

        x = self.silu(self.input_linear(x))
        for block in self.blocks:
            x = block(x)
        return self.output_linear(x)



__all__ = ["FourierTimeResidualMLP", "SiLUResBlock"]
