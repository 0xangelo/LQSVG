"""Neural network modules."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from lqsvg.torch import named as nt
from lqsvg.torch.utils import assemble_cholesky, disassemble_cholesky, softplusinv


class CholeskyFactor(nn.Module):
    """Matrix Cholesky factor stored as unconstrained parameters.

    Stores the Cholesky factor as two parameters:
    - lower triangular portion, excluding the diagonal
    - inverse softplus of the positive diagonal elements, as a vector
    """

    beta: float = 0.2
    ltril: nn.Parameter
    pre_diag: nn.Parameter

    def __init__(self, size: tuple[int, ...]):
        super().__init__()
        assert len(size) >= 2, "SPD matrix must have at least 2 dimensions"
        assert size[-2] == size[-1], "SPD matrix must be square"

        self.ltril = nn.Parameter(Tensor(*size))
        self.pre_diag = nn.Parameter(Tensor(*size[:-1]))
        self.reset_parameters()

    def reset_parameters(self):
        """Default parameter initialization.

        Factorizes an identity matrix by default.
        """
        nn.init.constant_(self.ltril, 0)
        softplusinv_one = softplusinv(torch.ones([]), beta=self.beta).item()
        nn.init.constant_(self.pre_diag, softplusinv_one)

    def factorize_(self, matrix: Tensor) -> CholeskyFactor:
        """Set parameters to factorize a symmetric positive definite matrix."""
        ltril, pre_diag = nt.unnamed(*disassemble_cholesky(matrix, beta=self.beta))
        self.ltril.data.copy_(ltril)
        self.pre_diag.data.copy_(pre_diag)
        return self

    def forward(self) -> Tensor:
        """Compute the Cholesky factor from parameters."""
        ltril = nt.matrix(self.ltril)
        pre_diag = nt.vector(self.pre_diag)
        return assemble_cholesky(ltril, pre_diag, beta=self.beta)


class SPDMatrix(CholeskyFactor):
    """Symmetric positive-definite matrix stored as unconstrained parameters.

    Stores the Cholesky factor of the matrix as two parameters:
    - lower triangular portion, excluding the diagonal
    - inverse softplus of the positive diagonal elements, as a vector
    """

    def forward(self) -> Tensor:
        """Compute the symmetric positive definite matrix from parameters."""
        ltril = nt.matrix(self.ltril)
        pre_diag = nt.vector(self.pre_diag)
        cholesky = assemble_cholesky(ltril, pre_diag, beta=self.beta)
        return cholesky @ nt.transpose(cholesky)
