"""
Implementations of loss functions with masking and weighting support.

Classes:
    MSELoss: Mean squared error loss with optional masking and weighting.
    MAELoss: Mean absolute error loss with optional masking and weighting.
    HuberLoss: Huber loss with optional masking and weighting.
    MAPELoss: Mean absolute percentage error loss with optional masking and weighting.
    SMAPELoss: Symmetric mean absolute percentage error loss with optional masking and weighting.
    L1Loss: Alias for MAELoss for standard naming compatibility.

The quantile loss function is implemented in the QuantileLoss class in the _quantile_loss module.
"""

from __future__ import annotations

import typing
import warnings

import torch

from .functional import mse_loss
from .functional._functional import mape_loss, smape_loss


class MSELoss(torch.nn.Module):
    """
    Mean squared error loss with optional masking and weighting.

    This loss function computes the mean squared error between predictions and targets,
    with support for masking out specific positions (e.g., padding in variable-length
    sequences) and applying sample-specific weights.

    Examples
    --------
    >>> import torch
    >>> from transformertf.nn import MSELoss
    >>>
    >>> # Basic usage
    >>> loss_fn = MSELoss()
    >>> pred = torch.randn(2, 10, 1)
    >>> target = torch.randn(2, 10, 1)
    >>> loss = loss_fn(pred, target)
    >>>
    >>> # With masking for variable-length sequences
    >>> mask = torch.ones(2, 10, 1)
    >>> mask[0, 7:] = 0  # Mask padding positions
    >>> mask[1, 5:] = 0
    >>> masked_loss = loss_fn(pred, target, mask=mask)
    >>>
    >>> # With both weights and masking
    >>> weights = torch.ones(2, 10, 1)
    >>> combined_loss = loss_fn(pred, target, weights=weights, mask=mask)
    """

    def __init__(
        self,
        reduction: typing.Literal["mean", "sum", "none"] | None = "mean",
        regularization: float | None = None,
        regularization_order: typing.Literal[1, 2, 3] = 1,
        regularization_dim: int = 0,
    ):
        """
        Mean squared error loss with optional masking and weighting.

        Parameters
        ----------
        reduction : str, optional
            Reduction method for the loss, by default "mean"
        """
        super().__init__()
        self.reduction_ = reduction
        self.regularization = regularization
        self.regularization_order = regularization_order
        self.regularization_dim = regularization_dim

    def forward(
        self,
        y_pred: torch.Tensor,
        target: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Calculate the mean squared error loss with optional masking and weighting.

        Parameters
        ----------
        y_pred: torch.Tensor
            Predicted values, shape (batch_size, seq_len, features).
        target: torch.Tensor
            Target values, same shape as y_pred.
        mask: torch.Tensor, optional
            Numeric mask to exclude certain positions from loss calculation.
            Values of 1.0 include the position, 0.0 excludes it. Intermediate
            values provide partial weighting. Shape must match y_pred and target.
        weights: torch.Tensor, optional
            Sample-specific weights to apply to the loss. If None, all samples
            are weighted equally. Shape must match y_pred and target.

        Returns
        -------
        torch.Tensor
            Mean squared error loss, scalar if reduction='mean' or 'sum',
            otherwise same shape as input.
        """
        return mse_loss(
            y_pred,
            target,
            mask=mask,
            weight=weights,
            reduction=self.reduction_,
            regularization=self.regularization,
            regularization_order=self.regularization_order,
            regularization_dim=self.regularization_dim,
        )


class MAELoss(torch.nn.L1Loss):
    """
    Mean absolute error loss with optional masking and weighting.

    This loss function computes the mean absolute error (L1 loss) between predictions
    and targets, with support for masking out specific positions and applying
    sample-specific weights. Also available as L1Loss alias.

    Examples
    --------
    >>> import torch
    >>> from transformertf.nn import MAELoss, L1Loss
    >>>
    >>> # Basic usage
    >>> loss_fn = MAELoss()
    >>> pred = torch.randn(2, 10, 1)
    >>> target = torch.randn(2, 10, 1)
    >>> loss = loss_fn(pred, target)
    >>>
    >>> # L1Loss alias
    >>> l1_loss = L1Loss()  # Same as MAELoss()
    >>> assert type(l1_loss) is MAELoss
    >>>
    >>> # With masking for RNN packed sequences
    >>> mask = torch.ones(2, 10, 1)
    >>> mask[0, 8:] = 0  # First sequence has length 8
    >>> mask[1, 6:] = 0  # Second sequence has length 6
    >>> masked_loss = loss_fn(pred, target, mask=mask)
    """

    def __init__(
        self, reduction: typing.Literal["mean", "sum", "none"] | None = "mean"
    ):
        """
        Mean absolute error loss with optional masking and weighting.

        Parameters
        ----------
        reduction : str, optional
            Reduction method for the loss, by default "mean"
        """
        super().__init__(reduction="none")
        self.reduction_ = reduction

    def forward(
        self,
        y_pred: torch.Tensor,
        target: torch.Tensor,
        *,
        weights: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Calculate the mean absolute error loss with optional masking and weighting.

        Parameters
        ----------
        y_pred: torch.Tensor
            Predicted values.
        target: torch.Tensor
            Target values.
        weights: torch.Tensor, optional
            Weights for the loss. If None, no weights are applied.
        mask: torch.Tensor, optional
            Numeric mask to exclude certain positions from loss calculation.
            Values of 1.0 include the position, 0.0 excludes it. Intermediate
            values provide partial weighting.

        Returns
        -------
        torch.Tensor
            Mean absolute error loss.
        """
        ae = super().forward(y_pred, target)

        # Combine mask and weights
        if mask is not None and weights is not None:
            mae = ae * weights * mask.float()
        elif mask is not None:
            mae = ae * mask.float()
        elif weights is not None:
            mae = ae * weights
        else:
            mae = ae

        if self.reduction_ == "mean":
            if mask is not None:
                # Masked mean: sum over valid positions divided by number of valid positions
                return (
                    mae.sum() / mask.float().sum()
                    if mask.any()
                    else torch.tensor(0.0, device=mae.device, dtype=mae.dtype)
                )
            return torch.mean(mae)
        if self.reduction_ == "sum":
            return torch.sum(mae)

        return mae


class HuberLoss(torch.nn.HuberLoss):
    """
    Huber loss with optional masking and weighting.

    The Huber loss combines the best properties of MSE and MAE losses: it is quadratic
    for small errors (like MSE) and linear for large errors (like MAE), making it
    robust to outliers while remaining differentiable everywhere.

    Examples
    --------
    >>> import torch
    >>> from transformertf.nn import HuberLoss
    >>>
    >>> # Basic usage with default delta=1.0
    >>> loss_fn = HuberLoss()
    >>> pred = torch.randn(2, 10, 1)
    >>> target = torch.randn(2, 10, 1)
    >>> loss = loss_fn(pred, target)
    >>>
    >>> # Custom delta for different sensitivity to outliers
    >>> robust_loss = HuberLoss(delta=0.5)
    >>>
    >>> # With masking for variable sequences
    >>> mask = torch.ones(2, 10, 1)
    >>> mask[0, -3:] = 0  # Mask last 3 positions
    >>> masked_loss = loss_fn(pred, target, mask=mask)
    """

    def __init__(
        self,
        delta: float = 1.0,
        reduction: typing.Literal["mean", "sum", "none"] | None = "mean",
    ):
        """
        Huber loss with optional masking and weighting.

        Parameters
        ----------
        delta : float, optional
            Threshold for the Huber loss, by default 1.0
        reduction : str, optional
            Reduction method for the loss, by default "mean"
        """
        super().__init__(delta=delta, reduction="none")
        self.reduction_ = reduction

    def forward(
        self,
        y_pred: torch.Tensor,
        target: torch.Tensor,
        *,
        weights: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Calculate the Huber loss with optional masking and weighting.

        Parameters
        ----------
        y_pred: torch.Tensor
            Predicted values.
        target: torch.Tensor
            Target values.
        weights: torch.Tensor, optional
            Weights for the loss. If None, no weights are applied.
        mask: torch.Tensor, optional
            Numeric mask to exclude certain positions from loss calculation.
            Values of 1.0 include the position, 0.0 excludes it. Intermediate
            values provide partial weighting.

        Returns
        -------
        torch.Tensor
            Huber loss.
        """
        loss = super().forward(y_pred, target)

        # Combine mask and weights
        if mask is not None and weights is not None:
            huber_loss = loss * weights * mask.float()
        elif mask is not None:
            huber_loss = loss * mask.float()
        elif weights is not None:
            huber_loss = loss * weights
        else:
            huber_loss = loss

        if self.reduction_ == "mean":
            if mask is not None:
                # Masked mean: sum over valid positions divided by number of valid positions
                return (
                    huber_loss.sum() / mask.float().sum()
                    if mask.any()
                    else torch.tensor(
                        0.0, device=huber_loss.device, dtype=huber_loss.dtype
                    )
                )
            return torch.mean(huber_loss)
        if self.reduction_ == "sum":
            return torch.sum(huber_loss)

        return huber_loss


class MAPELoss(torch.nn.Module):
    """
    Mean Absolute Percentage Error loss with optional masking and weighting.

    MAPE computes the mean of the absolute percentage differences between predictions
    and targets. It provides interpretable error metrics as percentages, useful when
    the scale of targets varies significantly.

    Examples
    --------
    >>> import torch
    >>> from transformertf.nn import MAPELoss
    >>>
    >>> # Basic usage
    >>> loss_fn = MAPELoss()
    >>> pred = torch.randn(2, 10, 1)
    >>> target = torch.abs(torch.randn(2, 10, 1)) + 0.1  # Avoid zeros
    >>> loss = loss_fn(pred, target)
    >>>
    >>> # With masking for padded sequences
    >>> mask = torch.ones(2, 10, 1)
    >>> mask[0, 7:] = 0  # Mask padding
    >>> masked_loss = loss_fn(pred, target, mask=mask)
    """

    def __init__(
        self,
        reduction: typing.Literal["mean", "sum", "none"] | None = "mean",
    ):
        """
        Mean Absolute Percentage Error loss with optional masking and weighting.

        Parameters
        ----------
        reduction : str, optional
            Reduction method for the loss, by default "mean"
        """
        super().__init__()
        self.reduction_ = reduction

    def forward(
        self,
        y_pred: torch.Tensor,
        target: torch.Tensor,
        *,
        weights: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Calculate the Mean Absolute Percentage Error loss.

        Parameters
        ----------
        y_pred: torch.Tensor
            Predicted values.
        target: torch.Tensor
            Target values.
        weights: torch.Tensor, optional
            Weights for the loss. If None, no weights are applied.
        mask: torch.Tensor, optional
            Numeric mask to exclude certain positions from loss calculation.
            Values of 1.0 include the position, 0.0 excludes it. Intermediate
            values provide partial weighting.

        Returns
        -------
        torch.Tensor
            MAPE loss.
        """
        return mape_loss(
            y_pred=y_pred,
            y_true=target,
            weights=weights,
            mask=mask,
            reduction=self.reduction_,
        )


class SMAPELoss(torch.nn.Module):
    """
    Symmetric Mean Absolute Percentage Error loss with optional masking and weighting.

    SMAPE provides a symmetric version of MAPE that treats over-prediction and
    under-prediction equally. It's bounded between 0 and 2, making it more stable
    than MAPE when targets are near zero. The symmetric nature makes it particularly
    useful for time series forecasting where prediction direction matters equally.

    Examples
    --------
    >>> import torch
    >>> from transformertf.nn import SMAPELoss
    >>>
    >>> # Basic usage
    >>> loss_fn = SMAPELoss()
    >>> pred = torch.randn(2, 10, 1)
    >>> target = torch.randn(2, 10, 1)
    >>> loss = loss_fn(pred, target)
    >>>
    >>> # With masking for time series with different lengths
    >>> mask = torch.ones(2, 10, 1)
    >>> mask[0, 8:] = 0  # First series has 8 valid points
    >>> mask[1, 6:] = 0  # Second series has 6 valid points
    >>> masked_loss = loss_fn(pred, target, mask=mask)
    >>>
    >>> # With sample importance weighting
    >>> weights = torch.ones(2, 10, 1)
    >>> weights[:, -3:] = 2.0  # Weight recent observations more heavily
    >>> weighted_loss = loss_fn(pred, target, weights=weights, mask=mask)
    """

    def __init__(
        self,
        reduction: typing.Literal["mean", "sum", "none"] | None = "mean",
    ):
        """
        Symmetric Mean Absolute Percentage Error loss with optional masking and weighting.

        Parameters
        ----------
        reduction : str, optional
            Reduction method for the loss, by default "mean"
        """
        super().__init__()
        self.reduction_ = reduction

    def forward(
        self,
        y_pred: torch.Tensor,
        target: torch.Tensor,
        weights: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Calculate the Symmetric Mean Absolute Percentage Error loss.

        Parameters
        ----------
        y_pred: torch.Tensor
            Predicted values.
        target: torch.Tensor
            Target values.
        weights: torch.Tensor, optional
            Weights for the loss. If None, no weights are applied.
        mask: torch.Tensor, optional
            Numeric mask to exclude certain positions from loss calculation.
            Values of 1.0 include the position, 0.0 excludes it. Intermediate
            values provide partial weighting.

        Returns
        -------
        torch.Tensor
            SMAPE loss.
        """
        return smape_loss(
            y_pred=y_pred,
            y_true=target,
            weights=weights,
            mask=mask,
            reduction=self.reduction_,
        )


# Alias for compatibility with standard naming
L1Loss = MAELoss


# Deprecated aliases for backward compatibility
def _deprecated_class_factory(new_class, old_name):
    """Create a deprecated alias class that issues warnings when instantiated."""

    class DeprecatedClass(new_class):
        def __init__(self, *args, **kwargs):
            warnings.warn(
                f"{old_name} is deprecated and will be removed in a future version. "
                f"Use {new_class.__name__} instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            super().__init__(*args, **kwargs)

    DeprecatedClass.__name__ = old_name
    DeprecatedClass.__qualname__ = old_name
    return DeprecatedClass


# Create deprecated aliases
WeightedMSELoss = _deprecated_class_factory(MSELoss, "WeightedMSELoss")
WeightedMAELoss = _deprecated_class_factory(MAELoss, "WeightedMAELoss")
WeightedHuberLoss = _deprecated_class_factory(HuberLoss, "WeightedHuberLoss")
