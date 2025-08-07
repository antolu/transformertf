"""
Loss Function Factory for Neural Network Training.

This module provides a centralized factory function for creating loss functions
used in transformertf models. It supports various loss functions suitable for
different types of regression and forecasting tasks.

Functions
---------
get_loss : Factory function for loss function creation

Constants
---------
VALID_LOSS : Type alias for supported loss function names

Notes
-----
The loss function factory pattern enables consistent loss function creation
across different models and training scenarios. All loss functions support
weighted training and are designed for time series forecasting applications.

The supported loss functions cover different use cases:
- Point estimation (MSE, MAE, Huber)
- Uncertainty quantification (Quantile)
- Robust regression (Huber, MAE)

References
----------
.. [1] Huber, Peter J. "Robust estimation of a location parameter."
   The annals of mathematical statistics 35.1 (1964): 73-101.
.. [2] Koenker, Roger, and Gilbert Bassett Jr. "Regression quantiles."
   Econometrica (1978): 33-50.
"""

from __future__ import annotations

import typing

import torch

from ._quantile_loss import QuantileLoss
from ._weighted_loss import HuberLoss, MAELoss, MSELoss

VALID_LOSS = typing.Literal["mse", "mae", "quantile", "huber"]


_LOSS_MAP: dict[VALID_LOSS, type[torch.nn.Module]] = {
    "huber": HuberLoss,
    "mse": MSELoss,
    "mae": MAELoss,
    "quantile": QuantileLoss,
}


def get_loss(
    loss: VALID_LOSS,
    **loss_kwargs: typing.Any,
) -> torch.nn.MSELoss | QuantileLoss | torch.nn.L1Loss | torch.nn.HuberLoss:
    """
    Factory function for creating loss function instances.

    Creates and returns a loss function instance based on the specified loss type.
    This function provides a centralized way to instantiate loss functions with
    consistent naming and support for weighted training scenarios common in
    time series forecasting.

    Parameters
    ----------
    loss : VALID_LOSS
        Name of the loss function to create. Must be one of:
        - "mse": Mean Squared Error loss with masking and weighting support
        - "mae": Mean Absolute Error loss with masking and weighting support
        - "huber": Huber loss with masking and weighting support (robust to outliers)
        - "quantile": Quantile loss for uncertainty quantification
    **loss_kwargs : Any
        Additional keyword arguments passed to the loss function constructor.
        The accepted arguments depend on the specific loss function:
        - MSELoss: reduction, regularization, regularization_order, regularization_dim
        - MAELoss: reduction
        - HuberLoss: delta, reduction
        - QuantileLoss: quantiles

    Returns
    -------
    torch.nn.Module
        An instance of the requested loss function. The exact type depends
        on the loss parameter:
        - MSELoss for "mse"
        - MAELoss for "mae"
        - HuberLoss for "huber"
        - QuantileLoss for "quantile"

    Raises
    ------
    ValueError
        If the loss name is not in the list of supported loss functions.
    TypeError
        If invalid keyword arguments are provided for the specific loss function.

    Notes
    -----
    All loss functions support masking and sample weighting, which is essential for:
    1. Handling missing data and padding in variable-length sequences
    2. Masking invalid positions in packed sequences (RNN applications)
    3. Emphasizing certain time periods or samples
    4. Dealing with imbalanced datasets
    5. Implementing curriculum learning strategies

    Loss function characteristics:
    - **MSE**: Penalizes large errors quadratically, sensitive to outliers
    - **MAE**: Robust to outliers, provides median-based estimates
    - **Huber**: Combines MSE and MAE benefits, robust yet differentiable
    - **Quantile**: Enables uncertainty quantification and prediction intervals

    These implementations extend standard PyTorch loss functions with masking
    and weighting support, making them suitable for masked sequence modeling,
    variable-length sequences, and time series applications with irregular sampling.

    Examples
    --------
    >>> from transformertf.nn import get_loss
    >>> import torch
    >>>
    >>> # Basic loss functions
    >>> mse_loss = get_loss("mse")
    >>> mae_loss = get_loss("mae")
    >>>
    >>> # With custom parameters
    >>> huber_loss = get_loss("huber", delta=0.5, reduction="mean")
    >>> quantile_loss = get_loss("quantile", quantiles=[0.1, 0.5, 0.9])
    >>>
    >>> # Usage in training
    >>> y_pred = torch.randn(32, 128, 1)  # Batch predictions
    >>> y_true = torch.randn(32, 128, 1)  # Ground truth
    >>> weights = torch.ones(32, 128, 1)  # Sample weights
    >>>
    >>> # Weighted MSE loss
    >>> loss_value = mse_loss(y_pred, y_true, weights=weights)
    >>> print(f"MSE Loss: {loss_value.item():.4f}")
    >>>
    >>> # With masking for variable-length sequences
    >>> mask = torch.ones(32, 128, 1)
    >>> mask[:, 100:] = 0  # Mask padding positions
    >>> masked_loss = mse_loss(y_pred, y_true, mask=mask)
    >>>
    >>> # Quantile predictions for uncertainty
    >>> y_pred_quantiles = torch.randn(32, 128, 3)  # 3 quantiles
    >>> quantile_loss_value = quantile_loss(y_pred_quantiles, y_true.squeeze(-1))
    >>> print(f"Quantile Loss: {quantile_loss_value.item():.4f}")
    >>>
    >>> # Huber loss for robust training
    >>> robust_loss = get_loss("huber", delta=1.0)
    >>> robust_loss_value = robust_loss(y_pred, y_true, weights=weights)
    >>>
    >>> # Time series forecasting example
    >>> def create_forecasting_loss(loss_type, **kwargs):
    ...     return get_loss(loss_type, **kwargs)
    >>>
    >>> # For point forecasting
    >>> point_loss = create_forecasting_loss("mse", reduction="mean")
    >>>
    >>> # For probabilistic forecasting
    >>> prob_loss = create_forecasting_loss(
    ...     "quantile",
    ...     quantiles=[0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
    ... )
    >>>
    >>> # Error handling
    >>> try:
    ...     invalid_loss = get_loss("invalid_loss")
    ... except ValueError as e:
    ...     print(f"Error: {e}")

    See Also
    --------
    transformertf.nn.QuantileLoss : Detailed quantile loss documentation
    transformertf.nn.MSELoss : MSE loss implementation with masking support
    transformertf.nn.MAELoss : MAE loss implementation with masking support
    transformertf.nn.HuberLoss : Huber loss implementation with masking support

    References
    ----------
    .. [1] Huber, Peter J. "Robust estimation of a location parameter."
       The annals of mathematical statistics 35.1 (1964): 73-101.
    .. [2] Koenker, Roger, and Gilbert Bassett Jr. "Regression quantiles."
       Econometrica: journal of the Econometric Society (1978): 33-50.
    .. [3] Lim, Bryan, et al. "Temporal fusion transformers for interpretable
       multi-horizon time series forecasting." ICML 2021.
    """
    if loss not in _LOSS_MAP:
        valid_losses = list(_LOSS_MAP)
        error_message = f"loss must be one of {valid_losses}, not {loss}"
        raise ValueError(error_message)

    return _LOSS_MAP[loss](**loss_kwargs)  # type: ignore[call-arg,return-value]
