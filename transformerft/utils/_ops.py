from __future__ import annotations

import builtins
import typing

import torch

T_co = typing.TypeVar(
    "T_co",
    dict,
    tuple,
    typing.Sequence,
    torch.Tensor,
    covariant=True,
)
M_co = typing.TypeVar(
    "M_co",
    dict[typing.Hashable, torch.Tensor],
    tuple[torch.Tensor, ...],
    typing.Sequence[torch.Tensor],
    torch.Tensor,
    covariant=True,
)


# pyright: reportGeneralTypeIssues=false
def _op_T(
    data: T_co, op: typing.Callable[[torch.Tensor], torch.Tensor]
) -> T_co:
    if isinstance(data, dict):
        return type(data)({k: _op_T(v, op) for k, v in data.items()})
    elif isnamedtupleinstance(data):
        return type(data)(*[_op_T(v, op) for v in data])
    elif isinstance(data, tuple):
        return type(data)([_op_T(v, op) for v in data])
    elif isinstance(data, list):
        return type(data)([_op_T(v, op) for v in data])
    elif isinstance(data, typing.Sequence):
        return type(data)([_op_T(v, op) for v in data])  # type: ignore[call-arg]
    elif isinstance(data, torch.Tensor):
        return op(data)
    else:
        raise TypeError(f"Unknown type {type(data)}.")


def detach(data: T_co) -> T_co:
    """
    Detach data from the computational graph.
    """
    return _op_T(data, lambda x: x.detach())


def to(data: T_co, device: torch.device) -> T_co:
    """
    Copy the data from GPU to CPU
    """
    return _op_T(data, lambda x: x.to(device))


def to_cpu(data: T_co) -> T_co:
    """
    Copy the data from GPU to CPU
    """
    return _op_T(data, lambda x: x.to(torch.device("cpu")))


def truncate(data: M_co, length: int) -> M_co:
    return _op_T(data, lambda x: x[:length])


def squeeze(data: T_co) -> T_co:
    return _op_T(data, lambda x: x.squeeze())


def concatenate(*args: M_co) -> M_co:
    type_ = type(args[0])
    if not all(isinstance(x, type_) for x in args):
        raise TypeError("All arguments must be of the same type.")

    if isinstance(args[0], dict):
        return type(args[0])(
            {k: concatenate(*[x[k] for x in args]) for k in args[0]}  # type: ignore
        )
    elif isnamedtupleinstance(args[0]):
        return type_(
            *[concatenate(*[x[i] for x in args]) for i in range(len(args[0]))]
        )
    elif isinstance(args[0], tuple):
        return type(args[0])(
            [concatenate(*[x[i] for x in args]) for i in range(len(args[0]))]
        )
    elif isinstance(args[0], list):
        return type(args[0])(
            [concatenate(*[x[i] for x in args]) for i in range(len(args[0]))]
        )
    elif isinstance(args[0], typing.Sequence):
        return type(args[0])(
            *[concatenate(*[x[i] for x in args]) for i in range(len(args[0]))]
        )
    elif isinstance(args[0], torch.Tensor):
        return torch.cat(tuple(args))  # type: ignore
    else:
        raise TypeError(f"Unknown type {type(args[0])}.")


def slice(data: M_co, s: builtins.slice) -> M_co:
    return _op_T(data, lambda x: x.__getitem__(s))


def isnamedtupleinstance(x: typing.Any) -> bool:
    t = type(x)
    b = t.__bases__
    if len(b) != 1 or b[0] != tuple:
        return False
    f = getattr(t, "_fields", None)
    if not isinstance(f, tuple):
        return False
    return all(type(n) == str for n in f)
