from typing import Tuple

from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand, tensor

import numpy as np
import random

# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0
    new_height = height // kh
    new_width = width // kw
    ret = input.contiguous().view(batch, channel, new_height, kh, new_width, kw).permute(0, 1, 2, 4, 3, 5).contiguous().view(batch, channel, new_height, new_width, kh * kw)
    return (
        ret,
        new_height,
        new_width,
    )

def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Average pooling for 2D images

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width

    """
    new_input, new_height, new_width = tile(input, kernel)
    mean = new_input.mean(dim=4)
    ret = mean.contiguous().view(
        new_input.shape[0], new_input.shape[1], new_height, new_width
    )
    return ret

def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Max pooling for 2D images

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width

    """
    new_input, new_height, new_width = tile(input, kernel)
    max = new_input.max(dim=4)
    ret = max.contiguous().view(
        new_input.shape[0], new_input.shape[1], new_height, new_width
    )
    return ret

def dropout(input: Tensor, p: float, ignore: bool = False) -> Tensor:
    """Dropout for 2D images

    Args:
    ----
        input: batch x channel x height x width
        p: probability of dropping out a pixel
        ignore: if True, do not apply dropout

    Returns:
    -------
        Tensor of size batch x channel x height x width

    """
    if ignore or p == 0.0:
        return input
    mask = rand(input.shape, backend=input.backend)
    i = np.arange(mask.size)
    mask._tensor._storage[i] = mask._tensor._storage[i] if random.random() >= p else 0
    return input * mask

def softmax(input: Tensor, dim: int) -> Tensor:
    """Softmax for 2D images

    Args:
    ----
        input: batch x channel x height x width
        dim: dimension to apply softmax

    Returns:
    -------
        Tensor of size batch x channel x height x width

    """
    max = input.max(dim=dim)
    exp = (input - max).exp()
    sum = exp.sum(dim=dim)
    return exp / sum

def logsoftmax(input: Tensor, dim: int) -> Tensor:
    """Log softmax for 2D images

    Args:
    ----
        input: batch x channel x height x width
        dim: dimension to apply log softmax

    Returns:
    -------
        Tensor of size batch x channel x height x width

    """
    max = input.max(dim=dim)
    exp = (input - max).exp()
    sum = exp.sum(dim=dim)
    return (input - max) - sum.log()