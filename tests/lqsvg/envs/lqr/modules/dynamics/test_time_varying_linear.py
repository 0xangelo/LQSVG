import pytest
import torch
from torch import Tensor

import lqsvg.torch.named as nt
from lqsvg.envs.lqr.modules.dynamics.time_varying_linear import CovCholeskyFactor
from lqsvg.torch.utils import make_spd_matrix


@pytest.fixture
def sigma(n_tau: int, horizon: int):
    return nt.horizon(
        nt.matrix(
            make_spd_matrix(n_dim=n_tau, sample_shape=(horizon,), dtype=torch.float32)
        )
    )


def test_cov_cholesky_factor(sigma: Tensor):
    module = CovCholeskyFactor(sigma)
    untimed = module()

    scale_tril = nt.cholesky(sigma)
    assert nt.allclose(scale_tril, untimed)
    assert scale_tril.names == untimed.names
