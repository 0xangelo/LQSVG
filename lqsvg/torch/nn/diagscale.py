"""Learnable distribution scale modules."""
from typing import Tuple

import torch
from nnrl import nn as nnx
from nnrl.nn import init
from torch import Tensor, nn
from torch.nn.functional import softplus

__all__ = ["DiagScale"]


class DiagScale(nn.Module):
    """Input dependent/independent diagonal stddev.

    Utilizes bounded log_stddev as described in the 'Well behaved probabilistic
    networks' appendix of `PETS`_.

    .. _`PETS`: https://papers.nips.cc/paper/7725-deep-reinforcement-learning-in-a-handful-of-trials-using-probabilistic-dynamics-models

    Args:
        in_features: size of the input vector
        event_size: size of the corresponding random variable for which the
            diagonal stddev is predicted
        input_dependent_scale: Whether to parameterize the standard deviation
            as a function of the input. If False, uses the input only to infer
            the batch dimensions
        log_std_bounds: maximum and minimum values for the log standard
            deviation parameter
        bound_parameters: Whether to use buffers or learnable parameters for
            the log-scale bounds
    """  # pylint:disable=line-too-long

    def __init__(
        self,
        in_features: int,
        event_size: int,
        input_dependent_scale: bool,
        log_std_bounds: Tuple[float, float] = (2.0, -20),
        bound_parameters: bool = False,
    ):
        # pylint:disable=too-many-arguments
        super().__init__()
        if input_dependent_scale:
            self.log_scale_module = nn.Linear(in_features, event_size)
        else:
            self.log_scale_module = nnx.LeafParameter(event_size)

        max_logvar = torch.full((event_size,), log_std_bounds[0])
        min_logvar = torch.full((event_size,), log_std_bounds[1])
        if bound_parameters:
            self.max_logvar = nn.Parameter(max_logvar)
            self.min_logvar = nn.Parameter(min_logvar)
        else:
            self.register_buffer("max_logvar", max_logvar)
            self.register_buffer("min_logvar", min_logvar)

        self.apply(init.initialize_("orthogonal", gain=0.01))

    def forward(self, inputs: Tensor) -> Tensor:
        # pylint:disable=arguments-differ,missing-function-docstring
        log_scale = self.log_scale_module(inputs)
        max_logvar = self.max_logvar.expand_as(log_scale)
        min_logvar = self.min_logvar.expand_as(log_scale)
        log_scale = max_logvar - softplus(max_logvar - log_scale)
        log_scale = min_logvar + softplus(log_scale - min_logvar)
        scale = log_scale.exp()
        return torch.diag_embed(scale)
