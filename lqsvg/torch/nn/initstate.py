"""NN initial state models."""
from __future__ import annotations

from typing import Optional

import torch
from nnrl.nn import distributions as ptd
from nnrl.nn.distributions.types import DistParams, SampleLogp
from torch import IntTensor, Tensor, nn

from lqsvg.envs import lqr
from lqsvg.envs.lqr import make_gaussinit
from lqsvg.torch import named as nt
from lqsvg.torch.nn.cholesky import CholeskyFactor
from lqsvg.torch.nn.distributions import TVMultivariateNormal
from lqsvg.torch.types import SampleShape

__all__ = ["InitStateModule", "InitStateModel"]


class InitStateModule(ptd.Distribution):
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
        # noinspection PyArgumentList
        n_state = init.mu.size("R")
        return cls(n_state).copy_(init)

    def copy_(self, init: lqr.GaussInit) -> InitStateModule:
        """Update parameters to existing Gaussian initial state distribution.

        Args:
            init: pair of named tensors containing the location of
                the distribution and the (possibly non-diagonal)
                covariance matrix of the distribution.

        Returns:
            self
        """
        self.loc.data.copy_(init.mu)
        self.scale_tril.factorize_(init.sig)
        return self

    def forward(self) -> DistParams:
        # pylint:disable=missing-function-docstring
        loc = nt.vector(self.loc)
        return {"loc": loc, "scale_tril": self.scale_tril(), "time": self.time}

    def sample(self, sample_shape: SampleShape = ()) -> SampleLogp:
        return self.dist.sample(self(), sample_shape)

    def rsample(self, sample_shape: SampleShape = ()) -> SampleLogp:
        return self.dist.rsample(self(), sample_shape)

    def log_prob(self, value: Tensor) -> Tensor:
        return self.dist.log_prob(nt.vector(value), self())

    def standard_form(self) -> lqr.GaussInit:
        # pylint:disable=missing-function-docstring
        loc = nt.vector(self.loc)
        scale_tril = self.scale_tril()
        sigma = scale_tril @ nt.transpose(scale_tril)
        return lqr.GaussInit(loc, sigma)


class InitStateModel(InitStateModule):
    """Gaussian initial state distribution model."""

    def __init__(self, n_state: int, seed: Optional[int] = None):
        super().__init__(n_state)
        init = make_gaussinit(n_state, sample_covariance=True, rng=seed)
        self.copy_(init)
