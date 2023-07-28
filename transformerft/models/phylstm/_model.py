from __future__ import annotations

import logging
import typing
import torch
from einops import einops
from sklearn.utils.validation import NotFittedError, check_is_fitted
from torch import nn
from torch.nn import functional as F
from ._normalizer import RunningNormalizer

try:
    from torch.optim.lr_scheduler import LRScheduler
except ImportError:
    from torch.optim.lr_scheduler import _LRScheduler as LRScheduler  # noqa

from ._output import (
    PhyLSTM1Output,
    PhyLSTM1States,
    PhyLSTM2Output,
    PhyLSTM2States,
    PhyLSTM3Output,
    PhyLSTM3States,
)
from ._utils import GradientTorch
from ...utils import ops


__all__ = ["PhyLSTM1", "PhyLSTM2", "PhyLSTM3"]


STATE1 = typing.TypeVar("STATE1", bound=PhyLSTM1States)
STATE2 = typing.TypeVar("STATE2", bound=PhyLSTM2States)
STATE3 = typing.TypeVar("STATE3", bound=PhyLSTM3States)


log = logging.getLogger(__name__)


class PhyLSTM1(nn.Module):
    def __init__(
            self,
            num_layers: int = 3,
            sequence_length: int = 500,
            hidden_dim: int = 350,
            dropout: float = 0.2,
            i2b_k: torch.Tensor | float | None = None,
            i2b_m: torch.Tensor | float | None = None,
            running_normalizer: bool = True,
            input_center: torch.Tensor | float | None = None,
            input_scale: torch.Tensor | float | None = None,
            target_center: torch.Tensor | float | None = None,
            target_scale: torch.Tensor | float | None = None,
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
            a\\dot{B}(t) + b(B, \\dot{B}) + r(B, \\dot{B}, B(\tau)) = \\Gamma i(t)

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
        :param i2b_k: The proportionality constant between the current and
            the field without hysteresis.
        """
        super().__init__()

        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim

        # state space variable modeling
        self.lstm1 = nn.LSTM(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.ln1 = nn.LayerNorm(hidden_dim // 2)
        self.fc11 = nn.Linear(hidden_dim // 2, 3)

        # linear component of inputs
        self.linear = nn.Linear(1, 1)
        if i2b_k is not None and i2b_m is not None:
            self.linear.requires_grad_(False)
        if i2b_k is not None:
            self.linear.weight.data = torch.Tensor([[i2b_k]])
        if i2b_m is not None:
            self.linear.bias.data = torch.Tensor([i2b_m])

        self.running_normalizer = running_normalizer
        self.input_scaler = RunningNormalizer(
            center=input_center or 0.0,
            scale=input_scale or 1.0,
            num_features=2,
        )
        self.target_scaler = RunningNormalizer(
            center=target_center or 0.0,
            scale=target_scale or 1.0,
            num_features=1,
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
                isinstance(module, (nn.ConvTranspose2d, nn.Linear))
                and module.weight.requires_grad
        ):  # guard for linear_field
            gain = nn.init.calculate_gain("leaky_relu", 0.01)
            nn.init.orthogonal_(module.weight, gain=gain)

    @typing.overload
    def forward(
            self,
            x: torch.Tensor,
            return_states: typing.Literal[False] = False,
            hidden_state: typing.Optional[STATE1] = None,
    ) -> PhyLSTM1Output:
        ...

    @typing.overload
    def forward(
            self,
            x: torch.Tensor,
            return_states: typing.Literal[True],
            hidden_state: typing.Optional[STATE1] = None,
    ) -> tuple[PhyLSTM1Output, PhyLSTM1States]:
        ...

    def forward(
            self,
            x: torch.Tensor,
            return_states: bool = False,
            hidden_state: typing.Optional[STATE1] = None,
    ) -> PhyLSTM1Output | tuple[PhyLSTM1Output, PhyLSTM1States]:
        """
        Forward pass of the model.
        :param x: Input torch.Tensor of shape (batch_size, sequence_length, 1).
        :param return_states: If True, return the hidden states of the LSTM.
        :param hidden_state: Optional initial hidden state of the LSTM.

        :return: Output torch.Tensor of shape (batch_size, sequence_length, 1).
        """
        if hidden_state is not None:
            h_lstm1 = hidden_state["lstm1"]
        else:
            h_lstm1 = None

        try:
            check_is_fitted(self.input_scaler)
            x_scaled = self.input_scaler.transform(x)  # type: ignore
        except NotFittedError:  # on the first batch, the scaler is not fitted
            # one-off fit of the scaler
            x_scaled = RunningNormalizer(num_features=2).fit_transform(x)
            # x_scaled = x

        x1_scaled = x_scaled[:, :, 0, None]
        o_lstm1, h_lstm1 = self.lstm1(x1_scaled, hx=h_lstm1)
        o_lstm1 = F.leaky_relu(self.fc1(o_lstm1))

        o_lstm1 = self.ln1(o_lstm1)
        z = self.fc11(o_lstm1)

        # postprocessing
        z_raw = z

        # undo the scaling before adding to the proportional component
        try:
            check_is_fitted(self.target_scaler)
            z1 = self.target_scaler(z[..., 0])
            z2 = z[..., 1] * self.target_scaler.scale_
        except (
                NotFittedError
        ):  # on the first batch when the scaler is not fitted yet
            z1 = z[..., 0]
            z2 = z[..., 1]

        z1 = z1 + self.linear(x[..., 0, None])[..., 0]
        z2 = z2 + self.linear.weight * x[..., 1]

        z = torch.stack([z1, z2, z[..., 2]], dim=-1)

        assert isinstance(h_lstm1, tuple)
        output: PhyLSTM1Output = {"z": z, "z_raw": z_raw, "x_scaled": x_scaled}
        states: PhyLSTM1States = {"lstm1": tuple(o.detach() for o in h_lstm1)}

        if return_states:
            return output, states
        else:
            return output

    def fit_scaler(self, x: torch.Tensor, y: torch.Tensor) -> None:
        if self.training and self.running_normalizer:
            self.input_scaler.fit(x)

            # do not fit Bdot
            y_ = y[..., 0] - self.linear(x[..., 0, None])[..., 0]
            self.target_scaler.fit(y_[..., None].detach())
        else:
            log.warning(
                "Not fitting target scaler, model is not in training mode."
            )


class PhyLSTM2(PhyLSTM1):
    def __init__(
            self,
            num_layers: int = 3,
            sequence_length: int = 500,
            hidden_dim: int = 350,
            dropout: float = 0.2,
            i2b_k: torch.Tensor | float | None = None,
            i2b_m: torch.Tensor | float | None = None,
            running_normalizer: bool = True,
            input_center: torch.Tensor | float | None = None,
            input_scale: torch.Tensor | float | None = None,
            target_center: torch.Tensor | float | None = None,
            target_scale: torch.Tensor | float | None = None,
    ) -> None:
        super().__init__(
            num_layers=num_layers,
            sequence_length=sequence_length,
            hidden_dim=hidden_dim,
            dropout=dropout,
            i2b_k=i2b_k,
            i2b_m=i2b_m,
            running_normalizer=running_normalizer,
            input_center=input_center,
            input_scale=input_scale,
            target_center=target_center,
            target_scale=target_scale,
        )

        self.gradient = GradientTorch()

        # ??
        self.lstm2 = nn.LSTM(
            input_size=3,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.ln2 = nn.LayerNorm(hidden_dim // 2)
        self.fc21 = nn.Linear(hidden_dim // 2, 1)

        self.g_plus_x = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    @typing.overload  # type: ignore[override]
    def forward(
            self,
            x: torch.Tensor,
            return_states: typing.Literal[False] = False,
            hidden_state: typing.Optional[STATE2] = None,
    ) -> PhyLSTM2Output:
        ...

    @typing.overload  # type: ignore[override]
    def forward(
            self,
            x: torch.Tensor,
            return_states: typing.Literal[True],
            hidden_state: typing.Optional[STATE2] = None,
    ) -> tuple[PhyLSTM2Output, PhyLSTM2States]:   # type: ignore[override]
        ...

    def forward(
            self,
            x: torch.Tensor,
            return_states: bool = False,
            hidden_state: typing.Optional[STATE2] = None,
    ) -> PhyLSTM2Output | tuple[PhyLSTM2Output, PhyLSTM2States]:
        """
        Forward pass of the model.
        :param x: Input torch.Tensor of shape (batch_size, sequence_length, 1).
        :param return_states: If True, return the hidden states of the LSTM.
        :param hidden_state: Optional initial hidden state of the LSTM.

        :return: Output torch.Tensor of shape (batch_size, sequence_length, 1).
        """
        phylstm1_output: PhyLSTM1Output
        hidden1: PhyLSTM1States | None = None
        if return_states:
            phylstm1_output, hidden1 = super().forward(
                x, hidden_state=hidden_state, return_states=True
            )
        else:
            phylstm1_output = super().forward(
                x, hidden_state=hidden_state, return_states=False
            )

        x_scaled = phylstm1_output["x_scaled"]
        x1_scaled = x_scaled[..., 0, None]
        z = phylstm1_output["z_raw"]

        if hidden_state is None:
            h_lstm2 = None
        else:
            h_lstm2 = hidden_state["lstm2"]

        dz_dt = self.gradient(z)

        o_lstm2, h_lstm2 = self.lstm2(z, hx=h_lstm2)
        o_lstm2 = F.leaky_relu(self.fc2(o_lstm2))
        o_lstm2 = self.ln2(o_lstm2)
        g = self.fc21(o_lstm2)

        g_gamma_x = self.g_plus_x(torch.cat([g, x1_scaled], dim=2))

        output = typing.cast(PhyLSTM2Output, {
            **phylstm1_output,
            "dz_dt": dz_dt,
            "g": g,
            "g_gamma_x": g_gamma_x,
        })
        if return_states:
            assert hidden1 is not None
            states = {**hidden1, "lstm2": ops.detach(h_lstm2)}
            return output, typing.cast(PhyLSTM2States, states)
        else:
            return output


class PhyLSTM3(PhyLSTM2):
    def __init__(
            self,
            num_layers: int = 3,
            sequence_length: int = 500,
            hidden_dim: int = 350,
            dropout: float = 0.2,
            i2b_k: torch.Tensor | float | None = None,
            i2b_m: torch.Tensor | float | None = None,
            running_normalizer: bool = True,
            input_center: torch.Tensor | float | None = None,
            input_scale: torch.Tensor | float | None = None,
            target_center: torch.Tensor | float | None = None,
            target_scale: torch.Tensor | float | None = None,
    ):
        super().__init__(
            num_layers=num_layers,
            sequence_length=sequence_length,
            hidden_dim=hidden_dim,
            dropout=dropout,
            i2b_k=i2b_k,
            i2b_m=i2b_m,
            running_normalizer=running_normalizer,
            input_center=input_center,
            input_scale=input_scale,
            target_center=target_center,
            target_scale=target_scale,
        )

        # hysteric parameter modeling
        self.lstm3 = nn.LSTM(
            input_size=2,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.ln3 = nn.LayerNorm(hidden_dim // 2)
        self.fc31 = nn.Linear(hidden_dim // 2, 1)

    @typing.overload  # type: ignore[override]
    def forward(
            self,
            x: torch.Tensor,
            return_states: typing.Literal[False] = False,
            hidden_state: typing.Optional[STATE3] = None,
    ) -> PhyLSTM3Output:
        ...

    @typing.overload  # type: ignore[override]
    def forward(
            self,
            x: torch.Tensor,
            return_states: typing.Literal[True],
            hidden_state: typing.Optional[STATE3] = None,
    ) -> tuple[PhyLSTM3Output, PhyLSTM3States]:
        ...

    def forward(
            self,
            x: torch.Tensor,
            return_states: bool = False,
            hidden_state: typing.Optional[STATE3] = None,
    ) -> PhyLSTM3Output | tuple[PhyLSTM3Output, PhyLSTM3States]:
        """
        This forward pass can be used for both training and inference.

        During inference the previous hidden and cell states can be passed to the model
        so the model "remembers" the previously input sequence and can continue from there.


        :param x: Input sequence of shape (batch_size, sequence_length, 1)
        :param hidden_state: The previous hidden and cell states of the LSTM layers. Leave as None for training
                             and first batch of inference.
        :param return_states: Returns the hidden and cell states of the LSTM layers as well as the output.

        :return: The output of the model, and optionally the hidden and cell states of the LSTM layers.
        """
        phylstm2_output: PhyLSTM2Output
        hidden2: PhyLSTM2States | None = None
        if return_states:
            phylstm2_output, hidden2 = super().forward(
                x, hidden_state=hidden_state, return_states=True
            )
        else:
            phylstm2_output = super().forward(
                x, hidden_state=hidden_state, return_states=False
            )

        z = phylstm2_output["z_raw"]
        dz_dt = phylstm2_output["dz_dt"]

        if hidden_state is None:
            h_lstm3 = None
        else:
            h_lstm3 = hidden_state["lstm3"]

        dz_dt_0 = einops.repeat(
            dz_dt[:, 0, 1, None],
            "b f -> b t f",
            t=self.sequence_length,
        )
        delta_z_dot = dz_dt[..., 1, None] - dz_dt_0
        phi = torch.cat([delta_z_dot, z[..., 2, None]], dim=2)

        o_lstm3, h_lstm3 = self.lstm3(phi, hx=h_lstm3)
        o_lstm3 = F.leaky_relu(self.fc3(o_lstm3))

        o_lstm3 = self.ln3(o_lstm3)
        dr_dt = self.fc31(o_lstm3)

        output = typing.cast(PhyLSTM3Output, {
            **phylstm2_output,
            "dr_dt": dr_dt,
        })

        if return_states:
            assert hidden2 is not None
            states = {**hidden2, "lstm3": ops.detach(h_lstm3)}
            return output, typing.cast(PhyLSTM3States, states)
        else:
            return output
