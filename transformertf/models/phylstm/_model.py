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

from ...nn import GateAddNorm, GatedResidualNetwork, VariableSelection
from ...utils import ops
from ._output import (
    PhyLSTM1Output,
    PhyLSTM1States,
    PhyLSTM2Output,
    PhyLSTM2States,
    PhyLSTM3Output,
    PhyLSTM3States,
)

__all__ = ["PhyLSTM1Model", "PhyLSTM2Model", "PhyLSTM3Model"]


STATE1 = typing.TypeVar("STATE1", bound=PhyLSTM1States)
STATE2 = typing.TypeVar("STATE2", bound=PhyLSTM2States)
STATE3 = typing.TypeVar("STATE3", bound=PhyLSTM3States)


log = logging.getLogger(__name__)


class PhyLSTM1Model(nn.Module):
    def __init__(
        self,
        num_layers: int | tuple[int, ...] = 3,
        sequence_length: int = 500,
        hidden_dim: int | tuple[int, ...] = 350,
        hidden_dim_fc: int | tuple[int, ...] | None = None,
        dropout: float | tuple[float, ...] = 0.2,
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

        This model is designed to be used with the :class:`PhyLSTMModule` class,
        but is separated out for modularity.

        The model is compile-able with torchdynamo in Pytorch 2.0.

        This class only implements the first LSTM, and does not compute the
        physics constraints that the below models :class:`PhyLSTM2` and
        :class:`PhyLSTM3` do.

        :param num_layers: Number of LSTM layers.
        :param sequence_length: Length of the input sequence.
        :param hidden_dim: Number of hidden units in each LSTM layer.
        :param dropout: Dropout probability.
        """
        super().__init__()

        num_layers_ = _parse_vararg(num_layers, 1)
        hidden_dim_ = _parse_vararg(hidden_dim, 1)
        dropout_ = _parse_vararg(dropout, 1)

        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim_

        self.variable_selection = VariableSelection(
            n_features=1,
            hidden_dim=hidden_dim_,
            n_dim_model=hidden_dim_,
            context_size=None,
            dropout=dropout_,
        )

        # state space variable modeling
        self.lstm1 = nn.LSTM(
            input_size=hidden_dim_,
            hidden_size=hidden_dim_,
            num_layers=num_layers_,
            batch_first=True,
            dropout=dropout_,
        )

        self.post_lstm_gate = GateAddNorm(
            input_dim=hidden_dim_,
            output_dim=hidden_dim_,
            dropout=dropout_,
        )

        self.ff_z = GatedResidualNetwork(
            input_dim=hidden_dim_,
            output_dim=hidden_dim_,
            dropout=dropout_,
        )

        self.ff_idot = GatedResidualNetwork(
            input_dim=1,
            output_dim=hidden_dim_,
            dropout=dropout_,
        )

        self.ff_bi = GatedResidualNetwork(
            input_dim=hidden_dim_,
            output_dim=hidden_dim_,
            context_dim=hidden_dim_,
            dropout=dropout_,
        )
        self.ff_b = GatedResidualNetwork(
            input_dim=hidden_dim_,
            output_dim=hidden_dim_,
            dropout=dropout_,
            activation="relu",
        )

        self.fc_z = nn.Linear(hidden_dim_, 3)
        self.fc_b = nn.Linear(hidden_dim_, 2)

        if hidden_dim_fc is None:
            hidden_dim_ // 2
        else:
            _parse_vararg(hidden_dim_fc, 1)

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
        hx: STATE1 | None = None,
        *,
        return_states: typing.Literal[False] = False,
    ) -> PhyLSTM1Output: ...

    @typing.overload
    def forward(
        self,
        x: torch.Tensor,
        hx: STATE1 | None = None,
        *,
        return_states: typing.Literal[True],
    ) -> tuple[PhyLSTM1Output, PhyLSTM1States]: ...

    def forward(
        self,
        x: torch.Tensor,
        hx: STATE1 | None = None,
        *,
        return_states: bool = False,
    ) -> PhyLSTM1Output | tuple[PhyLSTM1Output, PhyLSTM1States]:
        """
        Forward pass of the model.
        :param x: Input torch.Tensor of shape (batch_size, sequence_length, 1).
        :param return_states: If True, return the hidden states of the LSTM.
        :param hx: Optional initial hidden state of the LSTM.

        :return: Output torch.Tensor of shape (batch_size, sequence_length, 1).
        """
        x_vs = self.variable_selection(x)[0]

        h_lstm1 = hx["lstm1"] if hx is not None else None

        o_lstm1, h_lstm1 = self.lstm1(x_vs, hx=h_lstm1)

        o_lstm1 = self.post_lstm_gate(o_lstm1, x_vs)

        z = self.ff_z(o_lstm1)
        z = self.fc_z(z)

        idot = self.ff_idot(torch.gradient(x, dim=1)[0])

        b = self.ff_bi(o_lstm1, idot)
        b = self.ff_b(b)
        b = self.fc_b(b)

        assert isinstance(h_lstm1, tuple)
        output: PhyLSTM1Output = {"z": z, "b": b}
        states: PhyLSTM1States = {"lstm1": tuple(o.detach() for o in h_lstm1)}

        if return_states:
            return output, states
        return output


class PhyLSTM2Model(PhyLSTM1Model):
    def __init__(
        self,
        num_layers: int | tuple[int, ...] = 3,
        sequence_length: int = 500,
        hidden_dim: int | tuple[int, ...] = 350,
        hidden_dim_fc: int | tuple[int, ...] | None = None,
        dropout: float | tuple[float, ...] = 0.2,
    ) -> None:
        super().__init__(
            num_layers=num_layers,
            sequence_length=sequence_length,
            hidden_dim=hidden_dim,
            hidden_dim_fc=hidden_dim_fc,
            dropout=dropout,
        )

        hidden_dim_ = _parse_vararg(hidden_dim, 2)
        num_layers_ = _parse_vararg(num_layers, 2)
        dropout_ = _parse_vararg(dropout, 2)

        self.lstm2 = nn.LSTM(
            input_size=3,
            hidden_size=hidden_dim_,
            num_layers=num_layers_,
            batch_first=True,
            dropout=dropout_,
        )

        if hidden_dim_fc is None:
            hidden_dim_fc_ = hidden_dim_ // 2
        else:
            hidden_dim_fc_ = _parse_vararg(hidden_dim_fc, 2)
        self.fc2 = nn.Sequential(
            collections.OrderedDict([
                ("fc21", nn.Linear(hidden_dim_, hidden_dim_fc_)),
                ("lrelu2", nn.LeakyReLU()),
                ("ln1", nn.LayerNorm(hidden_dim_fc_)),
                ("fc22", nn.Linear(hidden_dim_fc_, 1)),
            ])
        )

        self.g_plus_x = nn.Sequential(
            nn.Linear(2, hidden_dim_fc_),
            nn.LayerNorm(hidden_dim_fc_),
            nn.ReLU(),
            nn.Linear(hidden_dim_fc_, 1),
        )

    @typing.overload  # type: ignore[override]
    def forward(
        self,
        x: torch.Tensor,
        hx: STATE2 | None = None,
        *,
        return_states: typing.Literal[False] = False,
    ) -> PhyLSTM2Output: ...

    @typing.overload  # type: ignore[override]
    def forward(
        self,
        x: torch.Tensor,
        hx: STATE2 | None = None,
        *,
        return_states: typing.Literal[True],
    ) -> tuple[PhyLSTM2Output, PhyLSTM2States]:  # type: ignore[override]
        ...

    def forward(  # type: ignore[override]
        self,
        x: torch.Tensor,
        hx: STATE2 | None = None,
        *,
        return_states: bool = False,
    ) -> PhyLSTM2Output | tuple[PhyLSTM2Output, PhyLSTM2States]:
        """
        Forward pass of the model.
        :param x: Input torch.Tensor of shape (batch_size, sequence_length, 1).
        :param return_states: If True, return the hidden states of the LSTM.
        :param hx: Optional initial hidden state of the LSTM.

        :return: Output torch.Tensor of shape (batch_size, sequence_length, 1).
        """
        phylstm1_output: PhyLSTM1Output
        hidden1: PhyLSTM1States | None = None
        if return_states:
            phylstm1_output, hidden1 = super().forward(x, hx=hx, return_states=True)
        else:
            phylstm1_output = super().forward(x, hx=hx, return_states=False)

        z = phylstm1_output["z"]

        h_lstm2 = None if hx is None else hx.get("lstm2", None)

        dz_dt = torch.gradient(z, dim=1)[0]

        o_lstm2, h_lstm2 = self.lstm2(z, hx=h_lstm2)

        g = self.fc2(o_lstm2)
        g_gamma_x = self.g_plus_x(torch.cat([g, x], dim=2))

        output = typing.cast(
            PhyLSTM2Output,
            phylstm1_output
            | {
                "dz_dt": dz_dt,
                "g": g,
                "g_gamma_x": g_gamma_x,
            },
        )
        if return_states:
            assert hidden1 is not None
            states = hidden1 | {"lstm2": ops.detach(h_lstm2)}  # type: ignore[type-var]
            return output, typing.cast(PhyLSTM2States, states)
        return output


class PhyLSTM3Model(PhyLSTM2Model):
    def __init__(
        self,
        num_layers: int | tuple[int, ...] = 3,
        sequence_length: int = 500,
        hidden_dim: int | tuple[int, ...] = 350,
        hidden_dim_fc: int | tuple[int, ...] | None = None,
        dropout: float | tuple[float] = 0.2,
    ):
        super().__init__(
            num_layers=num_layers,
            sequence_length=sequence_length,
            hidden_dim=hidden_dim,
            hidden_dim_fc=hidden_dim_fc,
            dropout=dropout,
        )

        hidden_dim_ = _parse_vararg(hidden_dim, 3)
        num_layers_ = _parse_vararg(num_layers, 3)
        dropout_ = _parse_vararg(dropout, 3)

        # hysteric parameter modeling
        self.lstm3 = nn.LSTM(
            input_size=2,
            hidden_size=hidden_dim_,
            num_layers=num_layers_,
            batch_first=True,
            dropout=dropout_,
        )

        if hidden_dim_fc is None:
            hidden_dim_fc_ = hidden_dim_ // 2
        else:
            hidden_dim_fc_ = _parse_vararg(hidden_dim_fc, 3)
        self.fc3 = nn.Sequential(
            collections.OrderedDict([
                ("fc31", nn.Linear(hidden_dim_, hidden_dim_fc_)),
                ("lrelu3", nn.LeakyReLU()),
                ("ln3", nn.LayerNorm(hidden_dim_fc_)),
                ("fc32", nn.Linear(hidden_dim_fc_, 1)),
            ])
        )

    @typing.overload  # type: ignore[override]
    def forward(
        self,
        x: torch.Tensor,
        hx: STATE3 | None = None,
        *,
        return_states: typing.Literal[False] = False,
    ) -> PhyLSTM3Output: ...

    @typing.overload  # type: ignore[override]
    def forward(
        self,
        x: torch.Tensor,
        hx: STATE3 | None = None,
        *,
        return_states: typing.Literal[True],
    ) -> tuple[PhyLSTM3Output, PhyLSTM3States]: ...

    def forward(  # type: ignore[override]
        self,
        x: torch.Tensor,
        hx: STATE3 | None = None,
        *,
        return_states: bool = False,
    ) -> PhyLSTM3Output | tuple[PhyLSTM3Output, PhyLSTM3States]:
        """
        This forward pass can be used for both training and inference.

        During inference the previous hidden and cell states can be passed to the model
        so the model "remembers" the previously input sequence and can continue from there.


        :param x: Input sequence of shape (batch_size, sequence_length, 1)
        :param hx: The previous hidden and cell states of the LSTM layers. Leave as None for training
                             and first batch of inference.
        :param return_states: Returns the hidden and cell states of the LSTM layers as well as the output.

        :return: The output of the model, and optionally the hidden and cell states of the LSTM layers.
        """
        phylstm2_output: PhyLSTM2Output
        hidden2: PhyLSTM2States | None = None
        if return_states:
            phylstm2_output, hidden2 = super().forward(x, hx=hx, return_states=True)
        else:
            phylstm2_output = super().forward(x, hx=hx, return_states=False)

        z = phylstm2_output["z"]
        dz_dt = phylstm2_output["dz_dt"]

        h_lstm3 = None if hx is None else hx.get("lstm3", None)

        dz_dt_0 = einops.repeat(
            dz_dt[:, 0, 1, None],
            "b f -> b t f",
            t=self.sequence_length,
        )
        delta_z_dot = dz_dt[..., 1, None] - dz_dt_0
        phi = torch.cat([delta_z_dot, z[..., 2, None]], dim=2)

        o_lstm3, h_lstm3 = self.lstm3(phi, hx=h_lstm3)

        dr_dt = self.fc3(o_lstm3)

        output = typing.cast(
            PhyLSTM3Output,
            phylstm2_output
            | {
                "dr_dt": dr_dt,
            },
        )

        if return_states:
            assert hidden2 is not None
            states = hidden2 | {"lstm3": ops.detach(h_lstm3)}  # type: ignore[type-var]
            return output, typing.cast(PhyLSTM3States, states)
        return output


T = typing.TypeVar("T")


def _parse_vararg(vararg: T | tuple[T, ...], num_args: int) -> T:
    """
    Extract the `num_args`-th argument from `vararg`.

    Parameters
    ----------
    vararg
    num_args

    Returns
    -------

    """
    if not isinstance(vararg, tuple):
        return vararg

    if len(vararg) < num_args:
        msg = f"Expected at least {num_args} arguments, got {len(vararg)}"
        raise ValueError(msg)
    return vararg[num_args - 1]
