"""Gaussian initial state dynamics as a PyTorch module."""
from typing import List

import raylab.torch.nn.distributions as ptd
import torch
import torch.nn as nn
from raylab.torch.nn.distributions.types import DistParams, SampleLogp
from torch import Tensor

import lqsvg.torch.named as nt
from lqsvg.envs import lqr

from .common import TVMultivariateNormal, assemble_scale_tril, disassemble_covariance


class InitStateDynamics(ptd.Distribution):
    """Initial state distribution as a multivariate Normal.

    All outputs are named Tensors.

    Args:
        init: pair of named tensors containing the location of
            the distribution and the (possibly non-diagonal)
            covariance matrix of the distribution.
    """

    # pylint:disable=missing-class-docstring
    def __init__(self, init: lqr.GaussInit):
        super().__init__()
        loc, sigma = init
        self.dist = TVMultivariateNormal()
        self.loc = nn.Parameter(nt.unnamed(loc))
        ltril, pre_diag = nt.unnamed(*disassemble_covariance(sigma))
        self.ltril, self.pre_diag = (nn.Parameter(x) for x in (ltril, pre_diag))
        unit_vector = next(iter(nt.split(loc, 1, "R")))
        self.register_buffer("time", -torch.ones_like(unit_vector, dtype=torch.int))

    def scale_tril(self) -> Tensor:
        # pylint:disable=missing-function-docstring
        return nt.matrix(assemble_scale_tril(self.ltril, self.pre_diag))

    def forward(self) -> DistParams:
        # pylint:disable=missing-function-docstring
        loc = nt.vector(self.loc)
        return {"loc": loc, "scale_tril": self.scale_tril(), "time": self.time}

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
