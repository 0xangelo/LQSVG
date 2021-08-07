"""Recurring assertions for tests."""
import torch
from torch import Tensor, nn

from lqsvg.torch import named as nt

# pylint:disable=missing-function-docstring


def assert_grad_nonnull(tensor: Tensor, name: str = ""):
    assert tensor.grad is not None, (name, tensor)


def assert_grad_nonzero(tensor: Tensor, name: str = ""):
    assert not nt.allclose(tensor.grad, torch.zeros(())), (name, tensor)


def assert_all_grads_nonnull(module: nn.Module):
    for name, param in module.named_parameters():
        assert_grad_nonnull(param, name)


def assert_all_grads_nonzero(module: nn.Module):
    for name, param in module.named_parameters():
        assert_grad_nonzero(param, name)


def assert_any_grads_nonzero(module: nn.Module):
    null, zero = [], []
    for name, param in module.named_parameters():
        if param.grad is None:
            null += [name]
        elif torch.allclose(param.grad, torch.zeros(())):
            zero += [name]
        else:
            return
        assert False, f"All grads null or zero:\nNull: {null}\nZero: {zero}"
