"""
Implementation of the autodifferentiation Functions for Tensor.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import numpy as np

import minitorch

from . import operators
from .autodiff import Context
from .tensor_ops import SimpleBackend, TensorBackend, index_permutation, index_broadcast

if TYPE_CHECKING:
    from typing import Any, List, Tuple

    from .tensor import Tensor
    from .tensor_data import UserIndex, UserShape


def wrap_tuple(x):  # type: ignore
    "Turn a possible value into a tuple"
    if isinstance(x, tuple):
        return x
    return (x,)


# Constructors
class Function:
    @classmethod
    def _backward(cls, ctx: Context, grad_out: Tensor) -> Tuple[Tensor, ...]:
        return wrap_tuple(cls.backward(ctx, grad_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: Tensor) -> Tensor:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: Tensor) -> Tensor:
        raw_vals = []
        need_grad = False
        for v in vals:
            if v.requires_grad():
                need_grad = True
            raw_vals.append(v.detach())

        # Create the context.
        ctx = Context(not need_grad)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)

        # assert isinstance(c, Tensor), "Expected return type Tensor got %s" % (
        #     type(c)
        # )

        # Create a new variable from the result with a new history.
        back = None
        if need_grad:
            back = minitorch.History(cls, ctx, vals)
        return minitorch.Tensor(c._tensor, back, backend=c.backend)


class Neg(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        return t1.f.neg_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        return grad_output.f.neg_map(grad_output)


class Inv(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        ctx.save_for_backward(t1)
        return t1.f.inv_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        (t1,) = ctx.saved_values
        return grad_output.f.inv_back_zip(t1, grad_output)


class Add(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        return t1.f.add_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        return grad_output, grad_output


class Mul(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        ctx.save_for_backward(a, b)
        return a.f.mul_zip(a, b)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        a, b = ctx.saved_tensors
        # return grad_output.f.mul_zip(grad_output, b), grad_output.f.mul_zip(grad_output, a)

        a_grad_broad = grad_output.f.mul_zip(grad_output, b)
        b_grad_broad = grad_output.f.mul_zip(grad_output, a)
        if len(a_grad_broad.shape) != len(a.shape) or np.not_equal(a_grad_broad.shape, a.shape).any():
            a_grad_broad_indices, a_indices = index_broadcast(
                np.array(a_grad_broad.shape, dtype=np.int32),
                np.array(a_grad_broad._tensor.strides, dtype=np.int32),
                np.array(a.shape, dtype=np.int32),
                np.array(a._tensor.strides, dtype=np.int32),
            )
            
            a_grad = a.zeros(a.shape)
            for i in range(len(a_grad_broad_indices)):
                a_grad._tensor._storage[a_indices[i]] += a_grad_broad._tensor._storage[a_grad_broad_indices[i]]
        else:
            a_grad = a_grad_broad
        
        if len(b_grad_broad.shape) != len(b.shape) or np.not_equal(b_grad_broad.shape, b.shape).any():
            b_grad_broad_indices, b_indices = index_broadcast(
                np.array(b_grad_broad.shape, dtype=np.int32),
                np.array(b_grad_broad._tensor.strides, dtype=np.int32),
                np.array(b.shape, dtype=np.int32),
                np.array(b._tensor.strides, dtype=np.int32),
            )
            
            b_grad = b.zeros(b.shape)
            for i in range(len(b_grad_broad_indices)):
                b_grad._tensor._storage[b_indices[i]] += b_grad_broad._tensor._storage[b_grad_broad_indices[i]]
        else:
            b_grad = b_grad_broad

        assert len(a_grad.shape) == len(a.shape) and len(b_grad.shape) == len(b.shape) and np.equal(a_grad.shape, a.shape).all() and np.equal(b_grad.shape, b.shape).all(), f"Shape mismatch, a_grad.shape: {a_grad.shape}, a.shape: {a.shape}, b_grad.shape: {b_grad.shape}, b.shape: {b.shape}"
        return a_grad, b_grad

class Sigmoid(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        sigmoid = t1.f.sigmoid_map(t1)
        ctx.save_for_backward(sigmoid)
        return sigmoid

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        (sigmoid, ) = ctx.saved_tensors
        return grad_output * sigmoid * (1 - sigmoid)


class ReLU(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        ctx.save_for_backward(t1)
        return t1.f.relu_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        return grad_output.f.relu_back_zip(ctx.saved_values[0], grad_output)


class Log(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        ctx.save_for_backward(t1)
        return t1.f.log_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        (t, ) = ctx.saved_tensors
        return grad_output.f.log_back_zip(t, grad_output)

class Exp(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        exp = t1.f.exp_map(t1)
        ctx.save_for_backward(exp)
        return exp

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        (exp, ) = ctx.saved_values
        return grad_output.f.mul_zip(grad_output, exp)


class Sum(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Tensor) -> Tensor:
        reduce_dim = int(dim.item())
        ctx.save_for_backward(a.shape, reduce_dim, a.backend)
        return a.f.add_reduce(a, reduce_dim)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        a_shape, reduce_dim, backend = ctx.saved_values
        ret = minitorch.zeros(shape=a_shape, backend=backend)
        if len(a_shape) == 1:
            ret._tensor._storage[np.arange(a_shape[0])] = grad_output.item()
            return ret, 0.0

        dim_stride = ret._tensor.strides[reduce_dim]
        grad_shape = np.array(grad_output.shape, dtype=np.int32)
        grad_strides = np.array(grad_output._tensor._strides, dtype=np.int32)
        grad_indices  = index_permutation(grad_shape, grad_strides)
        
        input_strides = np.array(ret._tensor.strides, dtype=np.int32)
        input_indices = index_permutation(grad_shape, input_strides)

        assert len(input_indices) == len(grad_indices), f"Length mismatch {len(input_indices)} != {len(grad_indices)}"
        
        for i in range(a_shape[reduce_dim]):
            ret_indices = input_indices + np.ones_like(input_indices, dtype=np.int32) * i * dim_stride
            if grad_output.backend.cuda:
                for i in range(len(grad_indices)):
                    ret._tensor._storage[ret_indices[i]] = grad_output._tensor._storage[grad_indices[i]]
            else:
                ret._tensor._storage[ret_indices] = grad_output._tensor._storage[grad_indices]

        assert np.equal(ret.shape, a_shape).all(), f"Shape mismatch {ret.shape} != {a_shape}"
        return ret, 0.0

class Max(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Tensor) -> Tensor:
        reduce_dim = int(dim.item())
        ctx.save_for_backward(a, reduce_dim)
        return a.f.max_reduce(a, reduce_dim)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        a: Tensor = ctx.saved_values[0]
        reduce_dim: int = ctx.saved_values[1]
        assert len(a.shape) == len(grad_output.shape), f"Shape mismatch {len(a.shape)} != {len(grad_output.shape)}"
        
        ret = minitorch.zeros(shape=a.shape, backend=a.backend)
        input_strides = np.array(a._tensor.strides, dtype=np.int32)

        grad_shape = np.array(grad_output.shape, dtype=np.int32)
        grad_strides = np.array(grad_output._tensor.strides, dtype=np.int32)
        inner_shape = np.ones_like(a.shape, dtype=np.int32)
        inner_shape[reduce_dim] = a.shape[reduce_dim]
        
        outer_indices = index_permutation(grad_shape, input_strides)
        arg_indices = np.zeros_like(outer_indices, dtype=np.int32)
        grad_indices = index_permutation(grad_shape, grad_strides)
        inner_indices = index_permutation(inner_shape, input_strides)
        for i in range(len(outer_indices)):
            j = np.argmax(a._tensor._storage[outer_indices[i] + inner_indices], axis=0)
            arg_indices[i] = outer_indices[i] + inner_indices[j]
        ret._tensor._storage[arg_indices] = grad_output._tensor._storage[grad_indices]
        return (ret, 0.0)

class Dropout(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, p: Tensor) -> Tensor:
        ctx.save_for_backward(a, p)
        return a.f.dropout_zip(a, p)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        return grad_output, 0.0
    
class All(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Tensor) -> Tensor:
        if dim is not None:
            return a.f.mul_reduce(a, int(dim.item()))
        else:
            return a.f.mul_reduce(a.contiguous().view(int(operators.prod(a.shape))), 0)


class LT(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        return a.f.lt_zip(a, b)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        return 0.0, 0.0


class EQ(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        return a.f.eq_zip(a, b)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        return 0.0, 0.0


class IsClose(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        return a.f.is_close_zip(a, b)


class Permute(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, order: Tensor) -> Tensor:
        order = np.array(order.to_numpy(), dtype = np.int32)
        b = a._new(a._tensor.permute(*order))
        ctx.save_for_backward(a.shape)
        return b

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        (original,) = ctx.saved_values
        return (
            minitorch.Tensor.make(
                grad_output._tensor._storage, original, backend=grad_output.backend
            ),
            0.0,
        )

class View(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, shape: Tensor) -> Tensor:
        ctx.save_for_backward(a.shape)

        assert a._tensor.is_contiguous(), "Must be contiguous to view"
        shape2 = [int(shape[i]) for i in range(shape.size)]
        return minitorch.Tensor.make(
            a._tensor._storage, tuple(shape2), backend=a.backend
        )

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        (original,) = ctx.saved_values
        ret_tensor = minitorch.Tensor.make(
            grad_output._tensor._storage, original, backend=grad_output.backend
        )

        assert np.equal(ret_tensor.shape, original).all(), f"Shape mismatch {ret_tensor.shape} != {original}"
        return (
            ret_tensor,
            0.0,
        )


class Copy(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor) -> Tensor:
        return a.f.id_map(a)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        return grad_output


class MatMul(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        ctx.save_for_backward(t1, t2)
        return t1.f.matrix_multiply(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        t1, t2 = ctx.saved_values

        def transpose(a: Tensor) -> Tensor:
            order = list(range(a.dims))
            order[-2], order[-1] = order[-1], order[-2]
            return a._new(a._tensor.permute(*order))

        return (
            grad_output.f.matrix_multiply(grad_output, transpose(t2)),
            grad_output.f.matrix_multiply(transpose(t1), grad_output),
        )


# Helpers for Constructing tensors
def zeros(shape: UserShape, backend: TensorBackend = SimpleBackend) -> Tensor:
    """
    Produce a zero tensor of size `shape`.

    Args:
        shape : shape of tensor
        backend : tensor backend

    Returns:
        new tensor
    """
    return minitorch.Tensor.make(
        [0.0] * int(np.prod(shape)), shape, backend=backend
    )


def rand(
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """
    Produce a random tensor of size `shape`.

    Args:
        shape : shape of tensor
        backend : tensor backend
        requires_grad : turn on autodifferentiation

    Returns:
        :class:`Tensor` : new tensor
    """
    vals = [random.random() for _ in range(int(operators.prod(shape)))]
    tensor = minitorch.Tensor.make(vals, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def _tensor(
    ls: Any,
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """
    Produce a tensor with data ls and shape `shape`.

    Args:
        ls: data for tensor
        shape: shape of tensor
        backend: tensor backend
        requires_grad: turn on autodifferentiation

    Returns:
        new tensor
    """
    tensor = minitorch.Tensor.make(ls, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def tensor(
    ls: Any, backend: TensorBackend = SimpleBackend, requires_grad: bool = False
) -> Tensor:
    """
    Produce a tensor with data and shape from ls

    Args:
        ls: data for tensor
        backend : tensor backend
        requires_grad : turn on autodifferentiation

    Returns:
        :class:`Tensor` : new tensor
    """

    def shape(ls: Any) -> List[int]:
        if isinstance(ls, (list, tuple)):
            return [len(ls)] + shape(ls[0])
        else:
            return []

    def flatten(ls: Any) -> List[float]:
        if isinstance(ls, (list, tuple)):
            return [y for x in ls for y in flatten(x)]
        else:
            return [ls]

    cur = flatten(ls)
    shape2 = shape(ls)
    return _tensor(cur, tuple(shape2), backend=backend, requires_grad=requires_grad)


# Gradient check for tensors


def grad_central_difference(
    f: Any, *vals: Tensor, arg: int = 0, epsilon: float = 1e-6, ind: UserIndex
) -> float:
    x = vals[arg]
    up = zeros(x.shape)
    up[ind] = epsilon
    vals1 = [x if j != arg else x + up for j, x in enumerate(vals)]
    vals2 = [x if j != arg else x - up for j, x in enumerate(vals)]
    delta: Tensor = f(*vals1).sum() - f(*vals2).sum()

    return delta[0] / (2.0 * epsilon)


def grad_check(f: Any, *vals: Tensor) -> None:
    for x in vals:
        x.requires_grad_(True)
        x.zero_grad_()
    random.seed(10)
    out = f(*vals)
    out.sum().backward()
    err_msg = """

Gradient check error for function %s.

Input %s

Received derivative %f for argument %d and index %s,
but was expecting derivative %f from central difference.

"""

    for i, x in enumerate(vals):
        ind = x._tensor.sample()
        check = grad_central_difference(f, *vals, arg=i, ind=ind)

        assert x.grad is not None
        np.testing.assert_allclose(
            x.grad[ind],
            check,
            1e-2,
            1e-2,
            err_msg=err_msg % (f, vals, x.grad[ind], i, ind, check),
        )
