from __future__ import annotations

import collections
import logging
import typing

import torch
from einops import einops
from torch import nn

try:
    from torch.optim.lr_scheduler import LRScheduler
except ImportError:
    from torch.optim.lr_scheduler import _LRScheduler as LRScheduler  # noqa

from ...utils import ops
from .typing import (
    BWOutput1,
    BWOutput2,
    BWOutput3,
    LSTMState,
)

__all__ = ["BWLSTM1Model", "BWLSTM2Model", "BWLSTM3Model"]


log = logging.getLogger(__name__)


class BWLSTM1Model(nn.Module):
    def __init__(
        self,
        n_features: int = 1,
        num_layers: int = 3,
        n_dim_model: int = 350,
        n_dim_fc: int | None = None,
        dropout: float = 0.2,
    ):
        """
        This is a PyTorch implementation of the Physics inspired neural network
        for modeling magnetic hysteresis. The model is based on the paper
        "Physics-Informed Multi-LSTM Networks for Metamodeling of Nonlinear
        Structures" https://arxiv.org/abs/2002.10253.

        Specifically this model implements a variant of the Bouc-Wen model for
        hysteresis. Instead of structural engineering this is used for magnetic
        flux.

        The physics equation this model is based on is the standard Bouc-Wen
        model, with the :math:`u` variable replaced by the magnetic flux
        :math:`B`, and the :math:`f(t)` input function replaced by the current
        :math:`i(t)`.

        .. math::
            a\\dot{B}(t) + b(B, \\dot{B}) + r(B, \\dot{B}, B(\\tau)) = \\Gamma i(t)

        Which can be rewritten as: :math:`\\ddot{B} + g = \\Gamma i(t)`

        This model is designed to be used with the :class:`BWLSTM` class,
        but is separated out for modularity.

        The model is compile-able with torchdynamo in Pytorch 2.0.

        This class only implements the first LSTM, and does not compute the
        physics constraints that the below models :class:`BWLSTMModel2` and
        :class:`BWLSTMModel3` do.

        :param num_layers: Number of LSTM layers.
        :param n_dim_model: Number of hidden units in each LSTM layer.
        :param dropout: Dropout probability.

        Parameters
        ----------
        num_layers : int
            Number of LSTM layers.
        n_dim_model : int
            Number of hidden units in each LSTM layer.
        n_dim_fc : int, optional
            Number of hidden units in the fully connected layer. If None, defaults to n_dim_model // 2.
        dropout : float
            Dropout probability.
        """
        super().__init__()

        # state space variable modeling
        self.lstm1 = nn.LSTM(
            input_size=n_features,
            hidden_size=n_dim_model,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        n_dim_fc_ = n_dim_fc or n_dim_model // 2
        self.fc1 = nn.Sequential(
            collections.OrderedDict([
                ("fc11", nn.Linear(n_dim_model, n_dim_fc_)),
                ("lrelu1", nn.LeakyReLU()),
                ("ln1", nn.LayerNorm(n_dim_fc_)),
                ("fc12", nn.Linear(n_dim_fc_, 3)),
            ])
        )
        self.apply(self.weights_init)

    @staticmethod
    def weights_init(module: nn.Module) -> None:
        """Initialize the weights of the given layer."""
        if isinstance(module, nn.LSTM):
            hidden_size = module.hidden_size
            for name, param in module.named_parameters():
                if "weight_ih" in name:
                    nn.init.xavier_normal_(param.data)
                elif "weight_hh" in name:
                    nn.init.xavier_normal_(param.data[:hidden_size, ...])
                    nn.init.xavier_normal_(
                        param.data[hidden_size : 2 * hidden_size, ...]
                    )
                    nn.init.xavier_normal_(
                        param.data[2 * hidden_size : 3 * hidden_size, ...]
                    )
                    nn.init.xavier_normal_(param.data[3 * hidden_size :, ...])
                elif "bias" in name:
                    pass
                    # nn.init.zeros_(param.data)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight.data)
            if module.bias is not None:
                nn.init.zeros_(module.bias.data)
        elif (
            isinstance(module, nn.ConvTranspose2d | nn.Linear)
            and module.weight.requires_grad
        ):  # guard for linear_field
            gain = nn.init.calculate_gain("leaky_relu", 0.01)
            nn.init.orthogonal_(module.weight, gain=gain)

    @typing.overload
    def forward(
        self,
        x: torch.Tensor,
        hx: LSTMState | None = None,
        *,
        return_states: typing.Literal[False] = False,
    ) -> BWOutput1: ...

    @typing.overload
    def forward(
        self,
        x: torch.Tensor,
        hx: LSTMState | None = None,
        *,
        return_states: typing.Literal[True],
    ) -> tuple[BWOutput1, LSTMState]: ...

    def forward(
        self,
        x: torch.Tensor,
        hx: LSTMState | None = None,
        *,
        return_states: bool = False,
    ) -> BWOutput1 | tuple[BWOutput1, LSTMState]:
        """
        Forward pass of the model.
        :param x: Input torch.Tensor of shape (batch_size, seq_len, 1).
        :param return_states: If True, return the hidden states of the LSTM.
        :param hx: Optional initial hidden state of the LSTM.

        :return: Output torch.Tensor of shape (batch_size, seq_len, 1).
        """
        o_lstm1, h_lstm1 = self.lstm1(x, hx=hx)

        z = self.fc1(o_lstm1)

        assert isinstance(h_lstm1, tuple)
        output: BWOutput1 = {"z": z}

        if return_states:
            return output, ops.detach(h_lstm1)
        return output


class BWLSTM2Model(nn.Module):
    def __init__(
        self,
        num_layers: int = 3,
        n_dim_model: int = 350,
        n_dim_fc: int | None = None,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.lstm2 = nn.LSTM(
            input_size=3,
            hidden_size=n_dim_model,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        n_dim_fc_ = n_dim_fc or n_dim_model // 2

        self.fc2 = nn.Sequential(
            collections.OrderedDict([
                ("fc21", nn.Linear(n_dim_model, n_dim_fc_)),
                ("lrelu2", nn.LeakyReLU()),
                ("ln1", nn.LayerNorm(n_dim_fc_)),
                ("fc22", nn.Linear(n_dim_fc_, 1)),
            ])
        )

        self.g_plus_x = nn.Sequential(
            nn.Linear(2, n_dim_fc_),
            nn.LayerNorm(n_dim_fc_),
            nn.ReLU(),
            nn.Linear(n_dim_fc_, 1),
        )

    @typing.overload  # type: ignore[override]
    def forward(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        hx: LSTMState | None = None,
        *,
        return_states: typing.Literal[False] = False,
    ) -> BWOutput2: ...

    @typing.overload  # type: ignore[override]
    def forward(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        hx: LSTMState | None = None,
        *,
        return_states: typing.Literal[True],
    ) -> tuple[BWOutput2, LSTMState]:  # type: ignore[override]
        ...

    def forward(  # type: ignore[override]
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        hx: LSTMState | None = None,
        *,
        return_states: bool = False,
    ) -> BWOutput2 | tuple[BWOutput2, LSTMState]:
        """
        Forward pass of the model.
        :param x: Input torch.Tensor of shape (batch_size, seq_len, 1), from :class:`BWLSTM1`.
        :param z: Input torch.Tensor of shape (batch_size, seq_len, 3), from :class:`BWLSTM1`.
        :param return_states: If True, return the hidden states of the LSTM.
        :param hx: Optional initial hidden state of the LSTM.

        :return: Output torch.Tensor of shape (batch_size, seq_len, 1).
        """
        if x.shape[-1] > 1:
            x = x[..., 1, None]
        dz_dt = torch.gradient(z, dim=1)[0]

        o_lstm2, h_lstm2 = self.lstm2(z, hx=hx)

        g = self.fc2(o_lstm2)
        g_gamma_x = self.g_plus_x(torch.cat([g, x], dim=2))

        output = typing.cast(
            BWOutput2,
            {
                "dz_dt": dz_dt,
                "g": g,
                "g_gamma_x": g_gamma_x,
            },
        )
        if return_states:
            return output, h_lstm2
        return output


class BWLSTM3Model(nn.Module):
    def __init__(
        self,
        num_layers: int = 3,
        n_dim_model: int = 350,
        n_dim_fc: int | None = None,
        dropout: float | tuple[float] = 0.2,
    ):
        super().__init__()
        # hysteric parameter modeling
        self.lstm3 = nn.LSTM(
            input_size=2,
            hidden_size=n_dim_model,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        n_dim_fc = n_dim_fc or n_dim_model // 2

        self.fc3 = nn.Sequential(
            collections.OrderedDict([
                ("fc31", nn.Linear(n_dim_model, n_dim_fc)),
                ("lrelu3", nn.LeakyReLU()),
                ("ln3", nn.LayerNorm(n_dim_fc)),
                ("fc32", nn.Linear(n_dim_fc, 1)),
            ])
        )

    @typing.overload  # type: ignore[override]
    def forward(
        self,
        z: torch.Tensor,
        dz_dt: torch.Tensor,
        hx: LSTMState | None = None,
        *,
        return_states: typing.Literal[False] = False,
    ) -> BWOutput3: ...

    @typing.overload  # type: ignore[override]
    def forward(
        self,
        z: torch.Tensor,
        dz_dt: torch.Tensor,
        hx: LSTMState | None = None,
        *,
        return_states: typing.Literal[True],
    ) -> tuple[BWOutput3, LSTMState]: ...

    def forward(  # type: ignore[override]
        self,
        z: torch.Tensor,
        dz_dt: torch.Tensor,
        hx: LSTMState | None = None,
        *,
        return_states: bool = False,
    ) -> BWOutput3 | tuple[BWOutput3, LSTMState]:
        """
        This forward pass can be used for both training and inference.

        During inference the previous hidden and cell states can be passed to the model
        so the model "remembers" the previously input sequence and can continue from there.


        :param hx: The previous hidden and cell states of the LSTM layers. Leave as None for training
                             and first batch of inference.
        :param return_states: Returns the hidden and cell states of the LSTM layers as well as the output.

        :return: The output of the model, and optionally the hidden and cell states of the LSTM layers.
        """
        dz_dt_0 = einops.repeat(
            dz_dt[:, 0, 1, None],
            "b f -> b t f",
            t=z.shape[1],
        )
        delta_z_dot = dz_dt[..., 1, None] - dz_dt_0
        phi = torch.cat([delta_z_dot, z[..., 2, None]], dim=2)

        o_lstm3, h_lstm3 = self.lstm3(phi, hx=hx)

        dr_dt = self.fc3(o_lstm3)

        output = typing.cast(
            BWOutput3,
            {
                "dr_dt": dr_dt,
            },
        )

        if return_states:
            return output, ops.detach(h_lstm3)
        return output
