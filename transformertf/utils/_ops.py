from __future__ import annotations

import builtins
import typing

import numpy as np
import torch

arr_t = typing.TypeVar("arr_t", torch.Tensor, np.ndarray)
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
    torch.Tensor,
    tuple[np.ndarray, ...],
    tuple[torch.Tensor, ...],
    list[np.ndarray],
    list[torch.Tensor],
    typing.Sequence[np.ndarray],
    typing.Sequence[torch.Tensor],
    covariant=True,
)
D_co = typing.TypeVar(
    "D_co",
    dict[typing.Hashable, np.ndarray],
    dict[typing.Hashable, torch.Tensor],
    covariant=True,
)


# pyright: reportGeneralTypeIssues=false
def _op_T(  # noqa: N802
    data: T_co, op: typing.Callable[[torch.Tensor], torch.Tensor]
) -> T_co:
    if isinstance(data, dict):
        return type(data)({k: _op_T(v, op) for k, v in data.items()})
    if isnamedtupleinstance(data):
        return type(data)(*[_op_T(v, op) for v in data])
    if isinstance(data, list | tuple):
        return type(data)([_op_T(v, op) for v in data])
    if isinstance(data, typing.Sequence):
        return type(data)([_op_T(v, op) for v in data])  # type: ignore[call-arg]
    if isinstance(data, torch.Tensor):
        return op(data)
    msg = f"Unknown type {type(data)}."
    raise TypeError(msg)


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


@typing.overload
def concatenate(
    value: tuple[arr_t, ...] | list[arr_t] | typing.Sequence[arr_t],
) -> arr_t: ...


@typing.overload
def concatenate(value: typing.Sequence[tuple[arr_t, ...]]) -> tuple[arr_t, ...]: ...


@typing.overload  # type: ignore[misc]
def concatenate(value: typing.Sequence[dict[str, arr_t]]) -> dict[str, arr_t]: ...


@typing.overload
def concatenate(value: typing.Sequence[list[arr_t]]) -> list[arr_t]: ...


@typing.overload
def concatenate(
    value: typing.Sequence[typing.Sequence[arr_t]],
) -> typing.Sequence[arr_t]: ...


def concatenate(  # type: ignore
    value: (
        list[tuple[arr_t, ...]]
        | list[arr_t]
        | typing.Sequence[dict[str, arr_t]]
        | typing.Sequence[arr_t]
        | typing.Sequence[tuple[arr_t, ...]]
        | typing.Sequence[list[arr_t]]
        | typing.Sequence[typing.Sequence[arr_t]]
    ),
) -> (
    arr_t | dict[str, arr_t] | list[arr_t] | tuple[arr_t, ...] | typing.Sequence[arr_t]
):
    if isinstance(value[0], torch.Tensor):
        value = typing.cast(
            typing.Sequence[torch.Tensor],
            value,
        )
        return torch.cat(list(value))
    if isinstance(value[0], np.ndarray):
        return np.concatenate(list(value))

    if isinstance(value[0], dict):
        if not all(isinstance(x, dict) for x in value):
            msg = "All arguments must be of the same type."
            raise TypeError(msg)

        value = typing.cast(
            typing.Sequence[dict[str, arr_t]],
            value,
        )
        arr_type = type(next(iter(value[0].values())))

        if not all(all(isinstance(x, arr_type) for x in d.values()) for d in value):
            msg = "All arguments must be of the same type."
            raise TypeError(msg)

        return type(value[0])({k: concatenate([x[k] for x in value]) for k in value[0]})

    type_ = type(value[0])
    if not all(isinstance(x, type_) for x in value):
        msg = "All arguments must be of the same type."
        raise TypeError(msg)

    if isinstance(value[0], tuple) and all(isinstance(x, tuple) for x in value):
        value = typing.cast(
            typing.Sequence[tuple[arr_t, ...]],
            value,
        )
        return type_(  # type: ignore[call-arg]
            [concatenate([x[i] for x in value]) for i in range(len(value[0]))]
        )
    if isinstance(value[0], list) and all(isinstance(x, list) for x in value):
        if all(isinstance(x, tuple) for x in value[0]):
            value = typing.cast(
                typing.Sequence[tuple[arr_t, ...]],
                value,
            )
            return type_(  # type: ignore[call-arg]
                [concatenate([x[i] for x in value]) for i in range(len(value[0]))]
            )
        value = typing.cast(
            typing.Sequence[list[arr_t]],
            value,
        )
        return type_(  # type: ignore[call-arg]
            [concatenate([x[i] for x in value]) for i in range(len(value[0]))]
        )
    if isinstance(value[0], typing.Sequence):
        value = typing.cast(
            typing.Sequence[typing.Sequence[arr_t]],
            value,
        )
        return type(value[0])(*[
            concatenate([x[i] for x in value]) for i in range(len(value[0]))
        ])

    msg = f"Unknown type {type(value[0])}."
    raise TypeError(msg)


def slice(data: M_co, s: builtins.slice) -> M_co:  # noqa: A001
    return _op_T(data, lambda x: x[s])  # type: ignore[return-value]


def isnamedtupleinstance(x: typing.Any) -> bool:
    """
    Determine if x is a namedtuple instance.

    Find the superclasses of x and check if any of them is tuple,
    and if so, check if x has a _fields attribute that is a tuple.
    """
    t = type(x)
    superclasses: set[type] = set()

    def find_superclasses(t: type) -> None:
        nonlocal superclasses
        bases = list(t.__bases__)
        for b in bases:
            find_superclasses(b)

        superclasses |= set(bases)

    find_superclasses(t)
    if tuple not in superclasses:
        return False
    f = getattr(t, "_fields", None)
    if not isinstance(f, tuple):
        return False
    return all(isinstance(n, str) for n in f)
