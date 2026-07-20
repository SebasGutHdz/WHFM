"""Distribution matching metrics."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from geomloss import SamplesLoss
from ot.sliced import sliced_wasserstein_distance
from torch import Tensor



def sliced_wasserstein2(x: Tensor, y: Tensor, num_projections: int, *, generator = None)-> float:
    return sliced_wasserstein_distance(x,y).detach().cpu()

def sinkhorn(x: Tensor, y: Tensor)-> float:
    sinkhorn_cfg = {"p": 2, "blur": 0.05, "scaling": 0.95}
    sink = SamplesLoss('sinkhorn', ** sinkhorn_cfg)
    return sink(x,y).detach().cpu()

# From : https://github.com/facebookresearch/generalized-schrodinger-bridge-matching/blob/main/gsbm/evaluator.py#L132
class MMD_loss(nn.Module):
    def __init__(self, kernel_mul=2.0, kernel_num=5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        return

    def guassian_kernel(
        self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None
    ):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)

        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1))
        )
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1))
        )
        L2_distance = ((total0 - total1) ** 2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [
            torch.exp(-L2_distance / bandwidth_temp)
            for bandwidth_temp in bandwidth_list
        ]
        return sum(kernel_val)

    def forward(self, source, target):
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(
            source,
            target,
            kernel_mul=self.kernel_mul,
            kernel_num=self.kernel_num,
            fix_sigma=self.fix_sigma,
        )
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY - YX)
        return loss
