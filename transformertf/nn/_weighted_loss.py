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

import torch

from .functional import mse_loss
from .functional._functional import mape_loss, smape_loss


class MSELoss(torch.nn.Module):
    """
    Mean squared error loss with optional masking and weighting.
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
        mask: torch.Tensor | None = None,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Calculate the weighted mean squared error loss.

        Parameters
        ----------
        y_pred: torch.Tensor
            Predicted values.
        target: torch.Tensor

        Returns
        -------
        torch.Tensor
            Weighted mean squared error loss.
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
            Boolean mask to exclude certain positions from loss calculation.
            If provided, only positions where mask=True contribute to the loss.

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
            Boolean mask to exclude certain positions from loss calculation.
            If provided, only positions where mask=True contribute to the loss.

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
            Boolean mask to exclude certain positions from loss calculation.
            If provided, only positions where mask=True contribute to the loss.

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
            Boolean mask to exclude certain positions from loss calculation.
            If provided, only positions where mask=True contribute to the loss.

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
