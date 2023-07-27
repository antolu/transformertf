from __future__ import annotations

import dataclasses
import functools
import typing

import torch
from torch import nn
from torch.nn import functional as F

from ._output import PhyLSTM1Output, PhyLSTM2Output, PhyLSTM3Output


if typing.TYPE_CHECKING:
    from ._config import PhyLSTMConfig


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


class PhyLSTMLoss(nn.Module):
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

    def __init__(self, loss_weights: LossWeights | None = None):
        super().__init__()

        loss_weights = loss_weights or LossWeights()
        assert isinstance(loss_weights, LossWeights)

        self.alpha = loss_weights.alpha
        self.beta = loss_weights.beta
        self.gamma = loss_weights.gamma
        self.eta = loss_weights.eta
        self.kappa = loss_weights.kappa

    @staticmethod
    def from_config(config: PhyLSTMConfig) -> PhyLSTMLoss:
        return PhyLSTMLoss(loss_weights=config.loss_weights)

    @property
    def weights(self) -> LossWeights:
        return LossWeights(
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
            eta=self.eta,
            kappa=self.kappa,
        )

    @typing.overload
    def forward(
            self,
            y_hat: PhyLSTM1Output | PhyLSTM2Output | PhyLSTM3Output,
            targets: torch.torch.Tensor,
            weights: LossWeights | None = None,
            *,
            return_all: typing.Literal[False],
    ) -> torch.Tensor:
        ...

    @typing.overload
    def forward(
            self,
            y_hat: PhyLSTM1Output | PhyLSTM2Output | PhyLSTM3Output,
            targets: torch.torch.Tensor,
            weights: LossWeights | None = None,
            *,
            return_all: typing.Literal[True],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        ...

    @typing.overload
    def forward(
            self,
            y_hat: PhyLSTM1Output | PhyLSTM2Output | PhyLSTM3Output,
            targets: torch.torch.Tensor,
            weights: LossWeights | None = None,
            return_all: typing.Literal[False] = False,
    ) -> torch.Tensor:
        ...

    def forward(
            self,
            y_hat: PhyLSTM1Output | PhyLSTM2Output | PhyLSTM3Output,
            targets: torch.torch.Tensor,
            weights: LossWeights | None = None,
            return_all: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Computes the loss function.

        :param y_hat: The output of the PhyLSTM model.
        :param targets: The target values, i.e. the B field.
        :param weights: The weights for the loss terms.
        :param return_all: Whether to return all the loss terms.

        :return: The loss value. For mathematical formulation see the module documentation.
        """
        return self.forward_explicit(
            z=y_hat["z_raw"],
            y=targets,
            dz_dt=y_hat.get("dz_dt"),
            dr_dt=y_hat.get("dr_dt"),
            gx=y_hat.get("g_gamma_x"),
            weights=weights,
            return_all=return_all,
        )

    @typing.overload
    def forward_explicit(
            self,
            z: torch.Tensor,
            y: torch.Tensor,
            dz_dt: torch.Tensor | None,
            gx: torch.Tensor | None,
            dr_dt: torch.Tensor | None,
            weights: LossWeights | None = None,
            *,
            return_all: typing.Literal[False],
    ) -> torch.Tensor:
        ...

    @typing.overload
    def forward_explicit(
            self,
            z: torch.Tensor,
            y: torch.Tensor,
            dz_dt: torch.Tensor | None,
            gx: torch.Tensor | None,
            dr_dt: torch.Tensor | None,
            weights: LossWeights | None = None,
            *,
            return_all: typing.Literal[True],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        ...

    @typing.overload
    def forward_explicit(
            self,
            z: torch.Tensor,
            y: torch.Tensor,
            dz_dt: torch.Tensor | None,
            gx: torch.Tensor | None,
            dr_dt: torch.Tensor | None,
            weights: LossWeights | None = None,
            return_all: typing.Literal[False] = False,
    ) -> torch.Tensor:
        ...

    @typing.overload
    def forward_explicit(
            self,
            z: torch.Tensor,
            y: torch.Tensor,
            dz_dt: torch.Tensor | None,
            gx: torch.Tensor | None,
            dr_dt: torch.Tensor | None,
            weights: LossWeights | None = None,
            return_all: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        ...

    def forward_explicit(
            self,
            z: torch.Tensor,
            y: torch.Tensor,
            dz_dt: torch.Tensor | None,
            gx: torch.Tensor | None,
            dr_dt: torch.Tensor | None,
            weights: LossWeights | None = None,
            return_all: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Computes the loss function.

        :param z: The output of the PhyLSTM1 model.
        :param y: The target values, i.e. the B field and its derivative.
        :param dz_dt: The time derivative of the output of the PhyLSTM2 model.
        :param gx: The output of the MLP, computed from PhyLSTM2.
        :param dr_dt: The time derivative of the hysteretic parameter r, from PhyLSTM3.
        :param weights: The weights for the loss terms.
        :param return_all: Whether to return all the loss terms.

        :return: The loss value. For mathematical formulation see the module documentation.
        """
        if y.ndim != 3:
            raise ValueError(
                "target y must have 3 dimensions. "
                "Maybe you forgot the batch dimension?"
            )

        if weights is None:
            alpha = self.alpha
            beta = self.beta
            gamma = self.gamma
            eta = self.eta
        else:
            alpha = weights.alpha
            beta = weights.beta
            gamma = weights.gamma
            eta = weights.eta

        mse = functools.partial(F.mse_loss, reduction="sum")
        loss_dict: dict[str, torch.Tensor] = {}

        loss_dict["loss1"] = alpha * mse(z[..., 0], y[..., 0])  # ||z1 - y1||^2
        loss_dict["loss2"] = beta * mse(z[..., 1], y[..., 1])  # ||z2 - y2||^2

        if dz_dt is not None:
            loss_dict["loss3"] = gamma * mse(
                dz_dt[:, :, 0], z[:, :, 1]
            )  # ||dz1/dt - z2||^2

        if gx is not None and dz_dt is not None:
            loss_dict["loss4"] = gamma * mse(
                dz_dt[:, :, 1].unsqueeze(-1), -gx
            )  # ||dz2/dt + MLP(g, i)||^2

        if dr_dt is not None and dz_dt is not None:
            loss_dict["loss5"] = eta * mse(
                dr_dt, dz_dt[:, :, 2].unsqueeze(-1)
            )  # ||dr/dt - dz3/dt||^2

        total_loss: torch.Tensor = torch.stack(list(loss_dict.values())).sum()

        if return_all:
            loss_dict["loss"] = total_loss
            return total_loss, loss_dict
        else:
            return total_loss
