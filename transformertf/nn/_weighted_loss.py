"""
Implementations of weighted loss functions.

Classes:
    WeightedMSELoss: Weighted mean squared error loss.
    WeightedMAELoss: Weighted mean absolute error loss.
    WeightedHuberLoss: Weighted Huber loss.

The quantile loss function is implemented in the QuantileLoss class in the _quantile_loss module.
"""

from __future__ import annotations

import typing

import torch


class WeightedMSELoss(torch.nn.MSELoss):
    """
    Weighted mean squared error loss.
    """

    def __init__(
        self, reduction: typing.Literal["mean", "sum", "none"] | None = "mean"
    ):
        """
        Weighted mean squared error loss.

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
        se = super().forward(y_pred, target)
        mse = weights * se if weights is not None else se

        if self.reduction_ == "mean":
            return torch.mean(mse)
        if self.reduction_ == "sum":
            return torch.sum(mse)

        return mse


class WeightedMAELoss(torch.nn.L1Loss):
    """
    Weighted mean absolute error loss.
    """

    def __init__(
        self, reduction: typing.Literal["mean", "sum", "none"] | None = "mean"
    ):
        """
        Weighted mean absolute error loss.

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
    ) -> torch.Tensor:
        """
        Calculate the weighted mean absolute error loss.

        Parameters
        ----------
        y_pred: torch.Tensor
            Predicted values.
        target: torch.Tensor
            Target values.
        weights: torch.Tensor, optional
            Weights for the loss. If None, no weights are applied.

        Returns
        -------
        torch.Tensor
            Weighted mean absolute error loss.
        """
        ae = super().forward(y_pred, target)
        mae = weights * ae if weights is not None else ae

        if self.reduction_ == "mean":
            return torch.mean(mae)
        if self.reduction_ == "sum":
            return torch.sum(mae)

        return mae


class WeightedHuberLoss(torch.nn.HuberLoss):
    """
    Weighted Huber loss.
    """

    def __init__(
        self,
        delta: float = 1.0,
        reduction: typing.Literal["mean", "sum", "none"] | None = "mean",
    ):
        """
        Weighted Huber loss.

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
    ) -> torch.Tensor:
        """
        Calculate the weighted Huber loss.

        Parameters
        ----------
        y_pred: torch.Tensor
            Predicted values.
        target: torch.Tensor
            Target values.
        weights: torch.Tensor, optional
            Weights for the loss. If None, no weights are applied.

        Returns
        -------
        torch.Tensor
            Weighted Huber loss.
        """
        loss = super().forward(y_pred, target)
        huber_loss = weights * loss if weights is not None else loss

        if self.reduction_ == "mean":
            return torch.mean(huber_loss)
        if self.reduction_ == "sum":
            return torch.sum(huber_loss)

        return huber_loss
