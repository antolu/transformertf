from __future__ import annotations

import contextlib
import functools
import typing

import torch

__all__ = ["compile", "maybe_compile", "set_compile"]


_should_compile = False
_compile_kwargs = {}


def set_compile(value: bool, **kwargs: typing.Any) -> None:  # noqa: FBT001
    global _should_compile  # noqa: PLW0603
    global _compile_kwargs  # noqa: PLW0603

    _should_compile = value
    _compile_kwargs = kwargs


@contextlib.contextmanager
def compile(**kwargs: typing.Any) -> typing.Iterator[None]:  # noqa: A001
    global _should_compile  # noqa: PLW0603
    global _compile_kwargs  # noqa: PLW0603

    _should_compile = True
    _compile_kwargs = kwargs

    yield

    _should_compile = False
    _compile_kwargs = {}


def maybe_compile(func: typing.Callable) -> typing.Callable:
    @functools.wraps(func)
    def wrapper(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
        if _should_compile:
            return torch.compile(func, _compile_kwargs)(*args, **kwargs)

        return func(*args, **kwargs)

    return wrapper
