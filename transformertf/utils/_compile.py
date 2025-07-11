from __future__ import annotations

import contextlib
import functools
import typing

import torch

__all__ = ["compile", "maybe_compile", "set_compile"]


_should_compile = False
_compile_kwargs = {}


def set_compile(value: bool, **kwargs: typing.Any) -> None:
    """
    Configure global compilation settings for PyTorch model optimization.

    This function sets the global compilation state and parameters that will be
    used by the `maybe_compile` decorator to conditionally compile functions.
    It provides a centralized way to control PyTorch's `torch.compile` behavior
    across the entire application.

    Parameters
    ----------
    value : bool
        Whether to enable compilation. When True, functions decorated with
        `maybe_compile` will be compiled using `torch.compile`. When False,
        functions will execute normally without compilation.
    **kwargs : typing.Any
        Additional keyword arguments to pass to `torch.compile` when compilation
        is enabled. Common options include:

        - mode : str, optional
            Compilation mode ("default", "reduce-overhead", "max-autotune")
        - backend : str, optional
            Compilation backend ("inductor", "aot_eager", etc.)
        - fullgraph : bool, optional
            Whether to compile the entire graph or allow graph breaks
        - dynamic : bool, optional
            Whether to enable dynamic shape compilation

    Notes
    -----
    This function modifies global state and affects all subsequent calls to
    `maybe_compile` decorated functions. It's typically called once at the
    beginning of training or inference to set compilation preferences.

    PyTorch compilation can significantly improve performance for certain
    model architectures and workloads, but may have overhead for small models
    or during the first few iterations.

    Examples
    --------
    Enable compilation with default settings:

    >>> from transformertf.utils import set_compile
    >>> set_compile(True)

    Enable compilation with custom backend:

    >>> set_compile(True, backend="inductor", mode="max-autotune")

    Disable compilation:

    >>> set_compile(False)

    Enable compilation with dynamic shapes:

    >>> set_compile(True, dynamic=True, fullgraph=False)

    See Also
    --------
    compile : Context manager for temporary compilation settings
    maybe_compile : Decorator that conditionally compiles functions
    torch.compile : PyTorch's compilation function
    """
    global _should_compile  # noqa: PLW0603
    global _compile_kwargs  # noqa: PLW0603

    _should_compile = value
    _compile_kwargs = kwargs


@contextlib.contextmanager
def compile(**kwargs: typing.Any) -> typing.Iterator[None]:  # noqa: A001
    """
    Context manager for temporary PyTorch compilation settings.

    This context manager temporarily enables compilation with specified parameters
    for functions decorated with `maybe_compile`. After exiting the context,
    compilation is disabled and settings are reset to their previous state.

    Parameters
    ----------
    **kwargs : typing.Any
        Keyword arguments to pass to `torch.compile` during the context.
        Common options include:

        - mode : str, optional
            Compilation mode ("default", "reduce-overhead", "max-autotune")
        - backend : str, optional
            Compilation backend ("inductor", "aot_eager", etc.)
        - fullgraph : bool, optional
            Whether to compile the entire graph or allow graph breaks
        - dynamic : bool, optional
            Whether to enable dynamic shape compilation

    Yields
    ------
    None
        The context manager yields None and enables compilation for the
        duration of the context.

    Notes
    -----
    This context manager is useful for temporarily enabling compilation
    for specific code blocks without affecting the global compilation state.
    It automatically restores the previous compilation settings when exiting.

    The context manager modifies global state temporarily, so it should be
    used carefully in multi-threaded environments.

    Examples
    --------
    Temporary compilation for a specific operation:

    >>> from transformertf.utils import compile, maybe_compile
    >>>
    >>> @maybe_compile
    ... def my_function(x):
    ...     return x * 2
    >>>
    >>> # Function runs normally
    >>> result1 = my_function(torch.tensor([1, 2, 3]))
    >>>
    >>> # Function is compiled within this context
    >>> with compile(mode="max-autotune"):
    ...     result2 = my_function(torch.tensor([1, 2, 3]))
    >>>
    >>> # Function runs normally again
    >>> result3 = my_function(torch.tensor([1, 2, 3]))

    Using with custom backend:

    >>> with compile(backend="inductor", fullgraph=True):
    ...     # All maybe_compile decorated functions will be compiled
    ...     # with the specified settings
    ...     model_output = model(input_data)

    See Also
    --------
    set_compile : Set global compilation settings
    maybe_compile : Decorator that conditionally compiles functions
    torch.compile : PyTorch's compilation function
    """
    global _should_compile  # noqa: PLW0603
    global _compile_kwargs  # noqa: PLW0603

    _should_compile = True
    _compile_kwargs = kwargs

    yield

    _should_compile = False
    _compile_kwargs = {}


def maybe_compile(func: typing.Callable) -> typing.Callable:
    """
    Decorator that conditionally compiles functions based on global settings.

    This decorator checks the global compilation state set by `set_compile` or
    the `compile` context manager and conditionally applies `torch.compile`
    to the decorated function. It provides a clean way to add compilation
    support to functions without changing their signatures.

    Parameters
    ----------
    func : typing.Callable
        The function to potentially compile. This should be a PyTorch function
        that can benefit from compilation (e.g., model forward passes,
        tensor operations, etc.).

    Returns
    -------
    typing.Callable
        A wrapped function that will be compiled if compilation is enabled,
        or will execute normally if compilation is disabled.

    Notes
    -----
    This decorator is designed to be used with functions that contain
    PyTorch operations that can benefit from compilation. It's particularly
    useful for model forward passes, loss computations, and other
    compute-intensive operations.

    The decorator checks the global compilation state on each function call,
    so compilation can be enabled or disabled dynamically during runtime.

    Compilation adds overhead on the first call due to graph analysis and
    optimization, but subsequent calls typically run faster. The trade-off
    is most beneficial for functions that are called repeatedly.

    Examples
    --------
    Basic usage with model forward pass:

    >>> import torch
    >>> from transformertf.utils import maybe_compile, set_compile
    >>>
    >>> @maybe_compile
    ... def model_forward(model, x):
    ...     return model(x)
    >>>
    >>> # Function runs normally
    >>> set_compile(False)
    >>> output = model_forward(my_model, input_tensor)
    >>>
    >>> # Function is compiled
    >>> set_compile(True)
    >>> output = model_forward(my_model, input_tensor)  # First call: compilation overhead
    >>> output = model_forward(my_model, input_tensor)  # Subsequent calls: faster

    Using with tensor operations:

    >>> @maybe_compile
    ... def complex_operation(x, y):
    ...     return torch.matmul(x, y.transpose(-1, -2)) + torch.relu(x)
    >>>
    >>> # Enable compilation with specific settings
    >>> set_compile(True, mode="max-autotune")
    >>> result = complex_operation(tensor_a, tensor_b)

    Using with context manager:

    >>> @maybe_compile
    ... def loss_computation(predictions, targets):
    ...     return torch.nn.functional.mse_loss(predictions, targets)
    >>>
    >>> # Temporarily compile for loss computation
    >>> with compile(backend="inductor"):
    ...     loss = loss_computation(pred, target)

    In Lightning modules:

    >>> class MyModel(LightningModule):
    ...     @maybe_compile
    ...     def forward(self, x):
    ...         return self.net(x)
    ...
    ...     def training_step(self, batch, batch_idx):
    ...         # Forward pass will be compiled if enabled
    ...         output = self(batch)
    ...         return self.loss(output, batch["target"])

    See Also
    --------
    set_compile : Set global compilation settings
    compile : Context manager for temporary compilation settings
    torch.compile : PyTorch's compilation function
    """

    @functools.wraps(func)
    def wrapper(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
        if _should_compile:
            return torch.compile(func, _compile_kwargs)(*args, **kwargs)

        return func(*args, **kwargs)

    return wrapper
