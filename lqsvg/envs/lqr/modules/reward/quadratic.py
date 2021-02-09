# pylint:disable=missing-module-docstring
import torch
import torch.nn as nn
from torch import Tensor

import lqsvg.torch.named as nt
from lqsvg.envs import lqr
from lqsvg.envs.lqr.utils import unpack_obs


class QuadraticCost(nn.Module):
    # pylint:disable=abstract-method,invalid-name,missing-docstring
    def __init__(self, cost: lqr.QuadCost):
        super().__init__()
        self.C = nn.Parameter(nt.unnamed(cost.C))
        self.c = nn.Parameter(nt.unnamed(cost.c))

    def copy(self, cost: lqr.QuadCost):
        def clone(tensor: Tensor) -> Tensor:
            return nt.unnamed(tensor.detach().clone())

        self.C = nn.Parameter(clone(nt.horizon(nt.matrix(cost.C))))
        self.c = nn.Parameter(clone(nt.horizon(nt.vector(cost.c))))

    def _named_quadcost(self) -> lqr.QuadCost:
        C = nt.horizon(nt.matrix(self.C))
        c = nt.horizon(nt.vector(self.c))
        return lqr.QuadCost(C, c)

    def forward(self, obs: Tensor, act: Tensor) -> Tensor:
        obs, act = (nt.vector(x) for x in (obs, act))
        state, time = unpack_obs(obs)
        tau = nt.vector_to_matrix(torch.cat([state, act], dim="R"))

        time = nt.vector_to_scalar(time)
        C, c = (nt.index_by(x, dim="H", index=time) for x in self._named_quadcost())
        c = nt.vector_to_matrix(c)

        cost = nt.transpose(tau) @ C @ tau / 2 + nt.transpose(c) @ tau
        reward = -cost
        return nt.matrix_to_scalar(reward)

    def standard_form(self) -> lqr.QuadCost:
        return self._named_quadcost()
