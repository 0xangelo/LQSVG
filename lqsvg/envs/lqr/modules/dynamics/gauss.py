"""Gaussian initial state dynamics as a PyTorch module."""
from __future__ import annotations

from typing import Sequence

import nnrl.nn.distributions as ptd
import torch
from nnrl.nn.distributions.types import DistParams, SampleLogp
from torch import IntTensor, Tensor, nn

import lqsvg.torch.named as nt
from lqsvg.envs import lqr
from lqsvg.torch.nn.cholesky import CholeskyFactor

from .common import TVMultivariateNormal

SampleShape = Sequence[int]


class InitStateDynamics(ptd.Distribution):
    """Initial state distribution as a multivariate Normal.

    All outputs are named Tensors.

    Args:
        n_state: size of the state vector
    """

    n_state: int
    loc: nn.Parameter
    scale_tril: CholeskyFactor
    time: IntTensor
    dist: TVMultivariateNormal

    def __init__(self, n_state: int):
        super().__init__()
        self.n_state = n_state

        self.loc = nn.Parameter(Tensor(n_state))
        self.scale_tril = CholeskyFactor((n_state, n_state))
        self.register_buffer("time", -nt.vector(torch.ones(1, dtype=torch.int)))
        self.reset_parameters()

        self.dist = TVMultivariateNormal()

    def reset_parameters(self):
        """Default parameter initialization.

        Initializes model as a standard Gaussian distribution.
        """
        nn.init.constant_(self.loc, 0)
        self.scale_tril.reset_parameters()

    @classmethod
    def from_existing(cls, init: lqr.GaussInit):
        """Create init state dynamics from existing Gaussian distribution."""
        loc, _ = init
        n_state = loc.size("R")
        return cls(n_state).copy_(init)

    def copy_(self, init: lqr.GaussInit) -> InitStateDynamics:
        """Update parameters to existing Gaussian initial state distribution.

        Args:
            init: pair of named tensors containing the location of
                the distribution and the (possibly non-diagonal)
                covariance matrix of the distribution.

        Returns:
            self
        """
        loc, sigma = init
        self.loc.data.copy_(loc)
        self.scale_tril.factorize_(sigma)
        return self

    def forward(self) -> DistParams:
        # pylint:disable=missing-function-docstring
        loc = nt.vector(self.loc)
        return {"loc": loc, "scale_tril": self.scale_tril(), "time": self.time}

    def sample(self, sample_shape: SampleShape = ()) -> SampleLogp:
        params = self()
        return self.dist.sample(params, sample_shape)

    def rsample(self, sample_shape: SampleShape = ()) -> SampleLogp:
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
        sigma = scale_tril @ nt.transpose(scale_tril)
        return lqr.GaussInit(loc, sigma)
