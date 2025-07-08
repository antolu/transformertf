"""
Activation Function Factory for Neural Network Components.

This module provides a centralized factory function for creating activation
functions used throughout the transformertf neural network components. It
supports commonly used activation functions with a unified interface.

Functions
---------
get_activation : Function factory for activation functions

Constants
---------
VALID_ACTIVATIONS : Type alias for supported activation function names

Notes
-----
The activation function factory pattern allows for consistent activation
function creation across different components while maintaining type safety
and providing a single point of configuration for supported activations.

All activation functions returned are PyTorch nn.Module instances that can
be used in neural network architectures with proper gradient computation
and device placement.

References
----------
.. [1] Glorot, Xavier, and Yoshua Bengio. "Understanding the difficulty of
   training deep feedforward neural networks." AISTATS 2010.
.. [2] Hendrycks, Dan, and Kevin Gimpel. "Gaussian error linear units (gelus)."
   arXiv preprint arXiv:1606.08415 (2016).
"""

from __future__ import annotations

import typing

import torch

VALID_ACTIVATIONS = typing.Literal["elu", "relu", "lrelu", "gelu", "tanh", "sigmoid"]
_ACTIVATION_MAP: dict[VALID_ACTIVATIONS, type[torch.nn.Module]] = {
    "elu": torch.nn.ELU,  # type: ignore[attr-defined]
    "relu": torch.nn.ReLU,
    "lrelu": torch.nn.LeakyReLU,
    "gelu": torch.nn.GELU,
    "tanh": torch.nn.Tanh,
    "sigmoid": torch.nn.Sigmoid,
}


def get_activation(
    activation: VALID_ACTIVATIONS,
    **activation_kwargs: typing.Any,
) -> (
    torch.nn.ELU
    | torch.nn.ReLU
    | torch.nn.LeakyReLU
    | torch.nn.GELU
    | torch.nn.Tanh
    | torch.nn.Sigmoid
):
    """
    Factory function for creating activation function instances.

    Creates and returns an activation function instance based on the specified
    activation type. This function provides a centralized way to instantiate
    activation functions with consistent naming and optional parameters.

    Parameters
    ----------
    activation : VALID_ACTIVATIONS
        Name of the activation function to create. Must be one of:
        - "elu": Exponential Linear Unit (α=1.0)
        - "relu": Rectified Linear Unit
        - "lrelu": Leaky ReLU (negative_slope=0.01)
        - "gelu": Gaussian Error Linear Unit
        - "tanh": Hyperbolic Tangent
        - "sigmoid": Sigmoid activation
    **activation_kwargs : Any
        Additional keyword arguments passed to the activation function constructor.
        The accepted arguments depend on the specific activation function:
        - ELU: alpha (default 1.0)
        - LeakyReLU: negative_slope (default 0.01)
        - Other activations: typically no additional arguments

    Returns
    -------
    torch.nn.Module
        An instance of the requested activation function. The exact type depends
        on the activation parameter:
        - torch.nn.ELU for "elu"
        - torch.nn.ReLU for "relu"
        - torch.nn.LeakyReLU for "lrelu"
        - torch.nn.GELU for "gelu"
        - torch.nn.Tanh for "tanh"
        - torch.nn.Sigmoid for "sigmoid"

    Raises
    ------
    ValueError
        If the activation name is not in the list of supported activations.
    TypeError
        If invalid keyword arguments are provided for the specific activation function.

    Notes
    -----
    The function uses a mapping dictionary to maintain the association between
    string names and PyTorch activation classes. This design allows for:
    1. Consistent activation function creation across components
    2. Easy addition of new activation functions
    3. Type safety through literal types
    4. Clear error messages for invalid activations

    Each activation function has different characteristics:
    - **ELU**: Smooth negative values, helps with vanishing gradients
    - **ReLU**: Simple, efficient, prone to dying ReLU problem
    - **LeakyReLU**: Addresses dying ReLU with small negative slope
    - **GELU**: Smooth approximation to ReLU, good for transformers
    - **Tanh**: Symmetric, bounded output [-1, 1]
    - **Sigmoid**: Bounded output [0, 1], prone to saturation

    Examples
    --------
    >>> from transformertf.nn import get_activation
    >>>
    >>> # Basic activation functions
    >>> relu = get_activation("relu")
    >>> print(type(relu))  # <class 'torch.nn.modules.activation.ReLU'>
    >>>
    >>> # With custom parameters
    >>> elu = get_activation("elu", alpha=0.5)
    >>> leaky_relu = get_activation("lrelu", negative_slope=0.2)
    >>>
    >>> # Use in neural network layers
    >>> import torch
    >>> x = torch.randn(32, 256)
    >>> gelu = get_activation("gelu")
    >>> activated = gelu(x)
    >>> print(activated.shape)  # torch.Size([32, 256])
    >>>
    >>> # Common usage in transformer components
    >>> activation_name = "elu"  # From configuration
    >>> activation_fn = get_activation(activation_name)
    >>> # Use activation_fn in GRN or other components
    >>>
    >>> # Error handling
    >>> try:
    ...     invalid_activation = get_activation("invalid")
    ... except ValueError as e:
    ...     print(f"Error: {e}")

    See Also
    --------
    transformertf.nn.GatedResidualNetwork : Uses activation functions
    torch.nn.functional : Functional versions of activation functions

    References
    ----------
    .. [1] Nair, Vinod, and Geoffrey E. Hinton. "Rectified linear units improve
       restricted boltzmann machines." ICML 2010.
    .. [2] Maas, Andrew L., Awni Y. Hannun, and Andrew Y. Ng. "Rectifier
       nonlinearities improve neural network acoustic models." ICML 2013.
    .. [3] Hendrycks, Dan, and Kevin Gimpel. "Gaussian error linear units (gelus)."
       arXiv preprint arXiv:1606.08415 (2016).
    """
    if activation not in _ACTIVATION_MAP:
        msg = f"activation must be one of {list(_ACTIVATION_MAP)}, not {activation}"
        raise ValueError(msg)

    return _ACTIVATION_MAP[activation](**activation_kwargs)  # type: ignore[call-arg]
