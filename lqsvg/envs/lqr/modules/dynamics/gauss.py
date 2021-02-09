"""Gaussian initial state dynamics as a PyTorch module."""
from typing import List

import raylab.torch.nn.distributions as ptd
import torch
import torch.nn as nn
from raylab.torch.nn.distributions.types import DistParams
from raylab.torch.nn.distributions.types import SampleLogp
from torch import Tensor

import lqsvg.torch.named as nt
from lqsvg.envs import lqr

from .common import assemble_scale_tril
from .common import disassemble_covariance
from .common import TVMultivariateNormal


class InitStateDynamics(ptd.Distribution):
    """Initial state distribution as a multivariate Normal.

    All outputs are named Tensors.

    Args:
        loc: location of the distribution
        covariance_matrix: covariance matrix of the distribution. May be
            non-diagonal.
    """

    # pylint:disable=missing-class-docstring
    def __init__(self, init: lqr.GaussInit):
        super().__init__()
        loc, covariance_matrix = init
        self.dist = TVMultivariateNormal()
        self.loc = nn.Parameter(nt.unnamed(loc))
        self.ltril, self.pre_diag = nt.unnamed(
            *disassemble_covariance(nt.matrix(covariance_matrix))
        )

    def scale_tril(self) -> Tensor:
        # pylint:disable=missing-function-docstring
        return nt.matrix(assemble_scale_tril(self.ltril, self.pre_diag))

    def forward(self) -> DistParams:
        # pylint:disable=missing-function-docstring
        return {
            "loc": nt.vector(self.loc),
            "scale_tril": self.scale_tril(),
            "time": nt.vector(torch.zeros_like(self.loc[..., -1:], dtype=torch.long)),
        }

    def sample(self, sample_shape: List[int] = ()) -> SampleLogp:
        params = self()
        return self.dist.sample(params, sample_shape)

    def rsample(self, sample_shape: List[int] = ()) -> SampleLogp:
        params = self()
        return self.dist.rsample(params, sample_shape)

    def log_prob(self, value: Tensor) -> Tensor:
        value = nt.vector(value)
        params = self()
        return self.dist.log_prob(value, params)

    def standard_form(self) -> lqr.GaussInit:
        # pylint:disable=missing-function-docstring
        loc = nt.vector(self.loc)
        scale_tril = self.scale_tril()
        covariance_matrix = scale_tril @ nt.transpose(scale_tril)
        return lqr.GaussInit(loc, covariance_matrix)
