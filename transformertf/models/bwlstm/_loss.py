from __future__ import annotations

import dataclasses
import typing

import torch
from torch import nn

from ...nn import functional as F
from . import typing as t


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
    |                        | :class:`t.BWOutput1`    | :class:`t.BWOutput12`   | :class:`t.BWOutput123` |
    +========================+=========================+=========================+=========================+
    | :math:`\\mathcal{L}_1` |   :white_check_mark:    |   :white_check_mark:    |   :white_check_mark:    |
    | :math:`\\mathcal{L}_2` |   :white_check_mark:    |   :white_check_mark:    |   :white_check_mark:    |
    | :math:`\\mathcal{L}_3` |                         |   :white_check_mark:    |   :white_check_mark:    |
    | :math:`\\mathcal{L}_4` |                         |   :white_check_mark:    |   :white_check_mark:    |
    | :math:`\\mathcal{L}_5` |                         |                         |   :white_check_mark:    |
    +========================+=========================+=========================+=========================+

    The loss functions are broken down in the methods :meth:`loss1`, :meth:`loss2`,
    :meth:`loss3`, :meth:`loss4`, and :meth:`loss5`. The total loss is computed in
    the :meth:`forward` method which computes a weighted sum of the individual loss*
    depending on the inputs provided.

    """

    @dataclasses.dataclass
    class LossWeights:
        """
        This class contains the weights for the loss function.

        Attributes:
            alpha: Weight for the first loss term (BWLSTM1).
            beta: Weight for the second loss term (BWLSTM1).
            gamma: Weight for the third loss term (BWLSTM2).
            eta: Weight for the fourth loss term (BWLSTM2).
            kappa: weight for the fifth loss term (BWLSTM3).
        """

        alpha: float = 1.0
        beta: float = 1.0
        gamma: float = 1.0
        eta: float = 1.0
        kappa: float = 1.0

    def __init__(self, loss_weights: BoucWenLoss.LossWeights | None = None):
        """
        Initializes the loss function.

        Parameters
        ----------
        loss_weights : BoucWenLoss.LossWeights | None, optional
            The weights for the loss terms, by default None. If None, the default
            weights are used from the BoucWenLoss.LossWeights class.
        """
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
    def trainable(self) -> bool:
        """
        Returns whether the weights are trainable.

        If training a self-adaptive PINN, the weights should be trainable.

        Returns
        -------
        bool
            Whether the weights are trainable.
        """
        return all(p.requires_grad for p in self.parameters())

    @trainable.setter
    def trainable(self, value: bool) -> None:
        """
        Sets whether the weights are trainable.

        Parameters
        ----------
        value : bool
            Whether the weights are trainable.
        """
        for p in self.parameters():
            p.requires_grad = value

    def invert_gradients(self) -> None:
        """
        Inverts the gradients of the weights.

        This is useful when training a self-adaptive PINN.
        """
        for p in self.parameters():
            if p.grad is not None:
                p.grad = -p.grad

    @property
    def weights(self) -> BoucWenLoss.LossWeights:
        """
        Returns the weights for the current loss function.

        Returns
        -------
        BoucWenLoss.LossWeights
            The weights for the loss function.
        """
        return BoucWenLoss.LossWeights(
            alpha=self.alpha.item(),
            beta=self.beta.item(),
            gamma=self.gamma.item(),
            eta=self.eta.item(),
            kappa=self.kappa.item(),
        )

    @staticmethod
    def point_prediction(
        y_hat: t.BWOutput1 | t.BWOutput12 | t.BWOutput123,
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
        y_hat: t.BWOutput1 | t.BWOutput12 | t.BWOutput123,
        targets: torch.torch.Tensor,
        weights: BoucWenLoss.LossWeights | None = None,
        *,
        return_all: typing.Literal[False],
    ) -> torch.Tensor: ...

    @typing.overload
    def forward(
        self,
        y_hat: t.BWOutput1 | t.BWOutput12 | t.BWOutput123,
        targets: torch.torch.Tensor,
        weights: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        *,
        return_all: typing.Literal[True],
    ) -> tuple[torch.Tensor, t.BWLoss1 | t.BWLoss2 | t.BWLoss3]: ...

    @typing.overload
    def forward(
        self,
        y_hat: t.BWOutput1 | t.BWOutput12 | t.BWOutput123,
        targets: torch.torch.Tensor,
        weights: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        *,
        return_all: typing.Literal[False] = False,
    ) -> torch.Tensor: ...

    def forward(
        self,
        y_hat: t.BWOutput1 | t.BWOutput12 | t.BWOutput123,
        targets: torch.torch.Tensor,
        weights: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        *,
        return_all: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, t.BWLoss1 | t.BWLoss2 | t.BWLoss3]:
        """
        Computes the loss function.

        Parameters
        ----------
        y_hat : BoucWenOutput1 | BoucWenOutput12 | BoucWenOutput123
            The output of the PhyLSTM model.
        targets : torch.Tensor
            The target values, i.e. the B field of shape [batch_size, seq_len, 1].
        weights : torch.Tensor | None, optional
            The loss weights for each sample, by default None.
            Should be shape [batch_size, 1].
        mask : torch.Tensor | None, optional
            The mask for the loss, by default None.
            Should be the same shape as the targets.
        return_all : bool, optional
            Whether to return all the loss terms, by default False.

        Returns
        -------
        torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]
            The loss value. If return_all is True, a tuple is returned with the
            loss value and a dictionary containing all the loss terms.
        """
        if targets.ndim != 3:
            msg = (
                "target y must have 3 dimensions. Maybe you forgot the batch dimension?"
            )
            raise ValueError(msg)

        loss_dict: dict[str, torch.Tensor] = {}

        loss1 = BoucWenLoss.loss1(y_hat, targets)
        loss2 = BoucWenLoss.loss2(y_hat, targets)

        loss_dict["loss1"] = loss1
        loss_dict["loss2"] = loss2

        if "dz_dt" in y_hat and "g_gamma_x" in y_hat:
            y_hat = typing.cast(t.BWOutput12, y_hat)
            loss_dict["loss3"] = BoucWenLoss.loss3(y_hat)
            loss_dict["loss4"] = BoucWenLoss.loss4(y_hat)

            if "dr_dt" in y_hat:
                y_hat = typing.cast(t.BWOutput123, y_hat)
                loss_dict["loss5"] = BoucWenLoss.loss5(y_hat)

        total_loss: torch.Tensor = (
            self.alpha * loss1
            + self.beta * loss2
            + self.gamma * loss_dict.get("loss3", 0.0)
            + self.eta * loss_dict.get("loss4", 0.0)
            + self.kappa * loss_dict.get("loss5", 0.0)
        )

        if return_all:
            loss_dict["loss_unweighted"] = typing.cast(
                torch.Tensor, sum(loss_dict.values())
            )
            loss_dict["loss"] = total_loss
            return total_loss, loss_dict
        return total_loss

    @staticmethod
    def loss1(
        y_hat: t.BWOutput1,
        targets: torch.Tensor,
        weights: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Computes the loss function for the first output of the model.

        :param y_hat: The output of the PhyLSTM model.
        :param targets: The target values, i.e. the B field. Shape (batch_size, seq_len, 1).

        :return: The loss value. For mathematical formulation see the module documentation.
        """
        B = BoucWenLoss.point_prediction(y_hat)

        return F.mse_loss(targets, B, weight=weights, mask=mask, reduction="mean")

    @staticmethod
    def loss2(
        y_hat: t.BWOutput1,
        targets: torch.Tensor,
        weights: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
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
            B_dot_hat = B_dot_hat + y_hat["b"][..., 1]

        return F.mse_loss(B_dot, B_dot_hat, weight=weights, mask=mask, reduction="mean")

    @staticmethod
    def loss3(
        y_hat: t.BWOutput12,
        weights: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Computes the loss function for the third output of the model.

        :param y_hat: The output of the PhyLSTM model.

        :return: The loss value. For mathematical formulation see the module documentation.
        """
        B_hat_dot = y_hat["dz_dt"][..., 0]
        B_dot_hat = y_hat["z"][..., 1]

        # avoid inplace operation
        if "b" in y_hat:
            B_hat_dot = B_hat_dot + torch.gradient(y_hat["b"][..., 0], dim=1)[0]
            B_dot_hat = B_dot_hat + y_hat["b"][..., 1]

        return F.mse_loss(
            B_hat_dot, B_dot_hat, weight=weights, mask=mask, reduction="mean"
        )

    @staticmethod
    def loss4(
        y_hat: t.BWOutput12,
        weights: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Computes the loss function for the fourth output of the model.

        :param y_hat: The output of the PhyLSTM model.

        :return: The loss value. For mathematical formulation see the module documentation.
        """
        B_dot_hat_dot = y_hat["dz_dt"][..., 1]
        if "b" in y_hat:
            B_dot_hat_dot = B_dot_hat_dot + torch.gradient(y_hat["b"][..., 1], dim=1)[0]
        g = y_hat["g_gamma_x"]

        return F.mse_loss(
            B_dot_hat_dot[..., None], -g, weight=weights, mask=mask, reduction="mean"
        )

    @staticmethod
    def loss5(
        y_hat: t.BWOutput123,
        weights: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Computes the loss function for the fifth output of the model.

        :param y_hat: The output of the PhyLSTM model.

        :return: The loss value. For mathematical formulation see the module documentation.
        """

        r_dot = y_hat["dz_dt"][..., 2]

        return F.mse_loss(
            y_hat["dr_dt"],
            r_dot[..., None],
            weight=weights,
            mask=mask,
            reduction="mean",
        )
