import pytest
import torch

from lqsvg.np_util import make_spd_matrix


@pytest.fixture(params=(1, 2, 4), ids=lambda x: f"NDim:{x}")
def n_dim(request):
    return request.param


@pytest.fixture(params=((1,), (2,), (2, 2)), ids=lambda x: f"SampleShape:{x}")
def sample_shape(request):
    return request.param


@pytest.fixture
def rng():
    return 42


def test_spd_matrix(n_dim, sample_shape, rng):
    # pylint:disable=invalid-name
    A = make_spd_matrix(n_dim, sample_shape=sample_shape, rng=rng)

    assert A.shape == sample_shape + (n_dim, n_dim)
    B = torch.as_tensor(A)
    assert torch.allclose(B, B.transpose(-2, -1))
    eigval, _ = torch.symeig(B)
    assert eigval.ge(0).all()
