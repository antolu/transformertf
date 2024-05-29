from __future__ import annotations

import dataclasses
import typing

import torch
from torch import nn
from torch.nn import functional as F

from ._types import BoucWenOutput1, BoucWenOutput2, BoucWenOutput3


class BoucWenOutput12(BoucWenOutput1, BoucWenOutput2):  # type: ignore[misc]
    pass


class BoucWenOutput123(BoucWenOutput12, BoucWenOutput3):  # type: ignore[misc]
    pass


class BWLoss1(typing.TypedDict):
    loss1: torch.Tensor
    loss2: torch.Tensor


class BWLoss2(BWLoss1):  # type: ignore[misc]
    loss3: torch.Tensor
    loss4: torch.Tensor


class BWLoss3(BWLoss2):  # type: ignore[misc]
    loss5: torch.Tensor


class BoucWenLoss(nn.Module):
    """
    This module implements the PINN loss function for the PhyLSTM model.

    The loss function is computed the following way:
    .. math::
        \\mathcal{L}_1 = ||z_1 - y_1||^2 \\newline
        \\mathcal{L}_2 = ||z_2 - y_2||^2  \\newline
        \\mathcal{L}_3 = ||\\dot{z}_1 - z_2||^2 \\newline
        \\mathcal{L}_4 = ||\\dot{z}_2 + \\text{MLP}(g, i)||^2 \\newline
        \\mathcal{L}_5 = ||\\dot{r} - \\dot{z}_3||^2
        \\mathcal{L}_{tot} = \\alpha \\mathcal{L}_1 + \\beta \\mathcal{L}_2 + \\gamma \\mathcal{L}_3 + \\eta \\mathcal{L}_4 + \\kappa \\mathcal{L}_5

    where

    .. math::
        \\dot{r} = f(\\phi) \\,\\text{and} \\phi = \\{\\Delta z_2, r\\}

    The components the loss computes depend on the output of the model. The
    following table shows which components are computed for which model output.

    +------------------------+-------------------------+-------------------------+-------------------------+
    |                        | :class:`PhyLSTM1Output` | :class:`PhyLSTM2Output` | :class:`PhyLSTM3Output` |
    +========================+=========================+=========================+=========================+
    | :math:`\\mathcal{L}_1` |   :white_check_mark:    |   :white_check_mark:    |   :white_check_mark:    |
    | :math:`\\mathcal{L}_2` |   :white_check_mark:    |   :white_check_mark:    |   :white_check_mark:    |
    | :math:`\\mathcal{L}_3` |                         |   :white_check_mark:    |   :white_check_mark:    |
    | :math:`\\mathcal{L}_4` |                         |   :white_check_mark:    |   :white_check_mark:    |
    | :math:`\\mathcal{L}_5` |                         |                         |   :white_check_mark:    |
    +========================+=========================+=========================+=========================+

    """

    @dataclasses.dataclass
    class LossWeights:
        """
        This class contains the weights for the loss function.

        Attributes:
            alpha: Weight for the first loss term (PhyLSTM1).
            beta: Weight for the second loss term (PhyLSTM1).
            gamma: Weight for the third loss term (PhyLSTM2).
            eta: Weight for the fourth loss term (PhyLSTM2).
            kappa: weight for the fifth loss term (PhyLSTM3).
        """

        alpha: float = 1.0
        beta: float = 1.0
        gamma: float = 1.0
        eta: float = 1.0
        kappa: float = 1.0

    def __init__(self, loss_weights: BoucWenLoss.LossWeights | None = None):
        super().__init__()
        loss_weights = loss_weights or BoucWenLoss.LossWeights()

        def to_nn_parameter(x: float) -> nn.Parameter:
            return nn.Parameter(torch.tensor(x), requires_grad=False)

        self.alpha = to_nn_parameter(loss_weights.alpha)
        self.beta = to_nn_parameter(loss_weights.beta)
        self.gamma = to_nn_parameter(loss_weights.gamma)
        self.eta = to_nn_parameter(loss_weights.eta)
        self.kappa = to_nn_parameter(loss_weights.kappa)

    @property
    def weights(self) -> BoucWenLoss.LossWeights:
        return BoucWenLoss.LossWeights(
            alpha=self.alpha.item(),
            beta=self.beta.item(),
            gamma=self.gamma.item(),
            eta=self.eta.item(),
            kappa=self.kappa.item(),
        )

    @staticmethod
    def point_prediction(
        y_hat: BoucWenOutput1 | BoucWenOutput2 | BoucWenOutput3,
    ) -> torch.Tensor:
        """
        Returns the magnetic field prediction, with eddy currents if they are present.

        Parameters
        ----------
        y_hat : BoucWenOutput1 | BoucWenOutput2 | BoucWenOutput3
            The output of the PhyLSTM model. Must contain the "z" key, and
            optionally the "b" key.
            z should have shape (batch_size, seq_len, 3)
            b should have shape (batch_size, seq_len, 2)

        Returns
        -------
        torch.Tensor
            The predicted magnetic field. If eddy currents are present, the
            prediction will be the sum of the hysteresis and eddy current
            predictions. The shape is (batch_size, seq_len, 1).
        """
        if "b" in y_hat:
            return y_hat["z"][..., 0, None] + y_hat["b"][..., 0, None]

        return y_hat["z"][..., 0, None]

    @typing.overload
    def forward(
        self,
        y_hat: BoucWenOutput1 | BoucWenOutput12 | BoucWenOutput123,
        targets: torch.torch.Tensor,
        weights: BoucWenLoss.LossWeights | None = None,
        *,
        return_all: typing.Literal[False],
    ) -> torch.Tensor: ...

    @typing.overload
    def forward(
        self,
        y_hat: BoucWenOutput1 | BoucWenOutput12 | BoucWenOutput123,
        targets: torch.torch.Tensor,
        weights: BoucWenLoss.LossWeights | None = None,
        *,
        return_all: typing.Literal[True],
    ) -> tuple[torch.Tensor, BWLoss1 | BWLoss2 | BWLoss3]: ...

    @typing.overload
    def forward(
        self,
        y_hat: BoucWenOutput1 | BoucWenOutput12 | BoucWenOutput123,
        targets: torch.torch.Tensor,
        weights: BoucWenLoss.LossWeights | None = None,
        *,
        return_all: typing.Literal[False] = False,
    ) -> torch.Tensor: ...

    def forward(
        self,
        y_hat: BoucWenOutput1 | BoucWenOutput12 | BoucWenOutput123,
        targets: torch.torch.Tensor,
        weights: BoucWenLoss.LossWeights | None = None,
        *,
        return_all: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, BWLoss1 | BWLoss2 | BWLoss3]:
        """
        Computes the loss function.

        Parameters
        ----------
        y_hat : BoucWenOutput1 | BoucWenOutput12 | BoucWenOutput123
            The output of the PhyLSTM model.
        targets : torch.Tensor
            The target values, i.e. the B field.
        weights : BoucWenLoss.LossWeights | None, optional
            The weights for the loss terms, by default None.
        return_all : bool, optional
            Whether to return all the loss terms, by default False.

        Returns
        -------
        torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]
            The loss value. For mathematical formulation see the module documentation.
        """
        if targets.ndim != 3:
            msg = (
                "target y must have 3 dimensions. "
                "Maybe you forgot the batch dimension?"
            )
            raise ValueError(msg)

        if weights is None:
            alpha = self.alpha
            beta = self.beta
            gamma = self.gamma
            eta = self.eta
            kappa = self.kappa
        else:
            alpha = weights.alpha
            beta = weights.beta
            gamma = weights.gamma
            eta = weights.eta
            kappa = weights.kappa

        loss_dict: dict[str, torch.Tensor] = {}

        loss1 = BoucWenLoss.loss1(y_hat, targets)
        loss2 = BoucWenLoss.loss2(y_hat, targets)

        loss_dict["loss1"] = alpha * loss1
        loss_dict["loss2"] = beta * loss2

        if "dz_dt" in y_hat and "g" in y_hat:
            y_hat = typing.cast(BoucWenOutput12, y_hat)
            loss_dict["loss3"] = gamma * BoucWenLoss.loss3(y_hat)
            loss_dict["loss4"] = eta * BoucWenLoss.loss4(y_hat)

            if "dr_dt" in y_hat:
                y_hat = typing.cast(BoucWenOutput123, y_hat)
                loss_dict["loss5"] = kappa * BoucWenLoss.loss5(y_hat)

        total_loss: torch.Tensor = torch.stack(list(loss_dict.values())).sum()

        if return_all:
            loss_dict["loss"] = total_loss
            return total_loss, loss_dict
        return total_loss

    @staticmethod
    def loss1(
        y_hat: BoucWenOutput1,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes the loss function for the first output of the model.

        :param y_hat: The output of the PhyLSTM model.
        :param targets: The target values, i.e. the B field. Shape (batch_size, seq_len, 1).

        :return: The loss value. For mathematical formulation see the module documentation.
        """
        B = BoucWenLoss.point_prediction(y_hat)

        return F.mse_loss(targets[..., 0], B, reduction="sum")

    @staticmethod
    def loss2(
        y_hat: BoucWenOutput1,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes the loss function for the second output of the model.

        :param y_hat: The output of the PhyLSTM model.
        :param targets: The target values, i.e. the B field. Shape (batch_size, seq_len, 1).
        :param weights: The weights for the loss terms.

        :return: The loss value. For mathematical formulation see the module documentation.
        """
        B_dot = torch.gradient(targets[..., 0], dim=1)[0]
        B_dot_hat = y_hat["z"][..., 1]

        if "b" in y_hat:
            B_dot_hat += y_hat["b"][..., 1]

        return F.mse_loss(B_dot, B_dot_hat, reduction="sum")

    @staticmethod
    def loss3(
        y_hat: BoucWenOutput12,
    ) -> torch.Tensor:
        """
        Computes the loss function for the third output of the model.

        :param y_hat: The output of the PhyLSTM model.

        :return: The loss value. For mathematical formulation see the module documentation.
        """
        B_hat_dot = y_hat["dz_dt"][..., 0]
        B_dot_hat = y_hat["z"][..., 1]

        if "b" in y_hat:
            B_hat_dot += torch.gradient(y_hat["b"][..., 0], dim=1)[0]
            B_dot_hat += y_hat["b"][..., 1]

        return F.mse_loss(B_hat_dot, B_dot_hat, reduction="sum")

    @staticmethod
    def loss4(
        y_hat: BoucWenOutput12,
    ) -> torch.Tensor:
        """
        Computes the loss function for the fourth output of the model.

        :param y_hat: The output of the PhyLSTM model.

        :return: The loss value. For mathematical formulation see the module documentation.
        """
        B_dot_hat_dot = y_hat["dz_dt"][..., 1]
        g = y_hat["g"]

        return F.mse_loss(B_dot_hat_dot[..., None], -g, reduction="sum")

    @staticmethod
    def loss5(
        y_hat: BoucWenOutput123,
    ) -> torch.Tensor:
        """
        Computes the loss function for the fifth output of the model.

        :param y_hat: The output of the PhyLSTM model.

        :return: The loss value. For mathematical formulation see the module documentation.
        """

        r_dot = y_hat["dz_dt"][..., 2]

        return F.mse_loss(y_hat["dr_dt"], r_dot[..., None], reduction="sum")
