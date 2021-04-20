"""Gaussian initial state dynamics as a PyTorch module."""
from typing import List

import raylab.torch.nn.distributions as ptd
import torch
import torch.nn as nn
from raylab.torch.nn.distributions.types import DistParams, SampleLogp
from torch import Tensor

import lqsvg.torch.named as nt
from lqsvg.envs import lqr

from .common import (
    TVMultivariateNormal,
    assemble_scale_tril,
    disassemble_covariance,
    softplusinv,
)


class InitStateDynamics(ptd.Distribution):
    """Initial state distribution as a multivariate Normal.

    All outputs are named Tensors.

    Args:
        n_state: size of the state vector
    """

    n_state: int

    def __init__(self, n_state: int):
        super().__init__()
        self.n_state = n_state
        self.dist = TVMultivariateNormal()

        self.loc = nn.Parameter(Tensor(n_state))
        self.ltril = nn.Parameter(Tensor(n_state, n_state))
        self.pre_diag = nn.Parameter(Tensor(n_state))
        self.register_buffer("time", -nt.vector(torch.ones(1, dtype=torch.int)))
        self.reset_parameters()

    def reset_parameters(self):
        """Default parameter initialization.

        Initializes model as a standard Gaussian distribution.
        """
        nn.init.constant_(self.loc, 0)
        nn.init.constant_(self.ltril, 0)
        nn.init.constant_(self.pre_diag, softplusinv(torch.ones([])).item())

    @classmethod
    def from_existing(cls, init: lqr.GaussInit):
        """Create init state dynamics from existing Gaussian distribution."""
        loc, _ = init
        n_state = loc.size("R")
        return cls(n_state).copy_(init)

    def copy_(self, init: lqr.GaussInit) -> "InitStateDynamics":
        """Update parameters to existing Gaussian initial state distribution.

        Args:
            init: pair of named tensors containing the location of
                the distribution and the (possibly non-diagonal)
                covariance matrix of the distribution.

        Returns:
            self
        """
        loc, sigma = init
        ltril, pre_diag = nt.unnamed(*disassemble_covariance(sigma))
        self.loc.data.copy_(loc)
        self.ltril.data.copy_(ltril)
        self.pre_diag.data.copy_(pre_diag)
        return self

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
