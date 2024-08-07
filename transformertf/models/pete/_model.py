from __future__ import annotations

import typing

import einops
import torch

from ...nn import (
    GateAddNorm,
    GatedResidualNetwork,
    InterpretableMultiHeadAttention,
    VariableSelection,
)
from ..bwlstm.typing import BWState3
from ..tsmixer import BasicTSMixerModel


class PETEModel(torch.nn.Module):
    def __init__(
        self,
        seq_len: int,
        num_features: int,
        n_dim_selection: int = 32,
        n_dim_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        n_layers_encoded: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_features = num_features
        self.n_dim_selection = n_dim_selection
        self.n_dim_model = n_dim_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.n_layers_encoded = n_layers_encoded
        self.dropout = dropout

        self.vs = VariableSelection(
            n_features=num_features,
            hidden_dim=n_dim_selection,
            n_dim_model=n_dim_model,
            dropout=dropout,
        )
        # self.lstm = torch.nn.LSTM(
        #     n_dim_model, n_dim_model, num_layers=n_layers, batch_first=True
        # )
        self.block = BasicTSMixerModel(
            seq_len=seq_len,
            num_features=n_dim_model,
            num_blocks=n_layers,
            hidden_dim=n_dim_model,
            activation="tanh",
            norm="layer",
        )
        self.attention = InterpretableMultiHeadAttention(
            n_dim_model=n_dim_model, n_heads=n_heads, dropout=dropout
        )
        self.gate = GateAddNorm(
            input_dim=n_dim_model, output_dim=n_dim_model, dropout=dropout
        )

        self.grn1h = GatedResidualNetwork(
            input_dim=seq_len, output_dim=n_layers_encoded, dropout=dropout
        )
        self.grn1o = GatedResidualNetwork(
            input_dim=seq_len, output_dim=n_layers_encoded, dropout=dropout
        )
        self.grn2h = GatedResidualNetwork(
            input_dim=seq_len, output_dim=n_layers_encoded, dropout=dropout
        )
        self.grn2o = GatedResidualNetwork(
            input_dim=seq_len, output_dim=n_layers_encoded, dropout=dropout
        )
        self.grn3h = GatedResidualNetwork(
            input_dim=seq_len, output_dim=n_layers_encoded, dropout=dropout
        )
        self.grn3o = GatedResidualNetwork(
            input_dim=seq_len, output_dim=n_layers_encoded, dropout=dropout
        )

    def forward(self, x: torch.Tensor) -> BWState3:
        x, _weights = self.vs(x)
        # x, _ = self.lstm(x)
        x = self.block(x)
        x = self.gate(self.attention(x, x, x)[0], x)

        x = einops.rearrange(x, "b s f -> b f s")
        h_lstm1 = self.grn1h(x)
        h_lstm1 = einops.rearrange(h_lstm1, "b f l -> l b f").contiguous()
        o_lstm1 = self.grn1o(x)
        o_lstm1 = einops.rearrange(o_lstm1, "b f l -> l b f").contiguous()

        h_lstm2 = self.grn2h(x)
        h_lstm2 = einops.rearrange(h_lstm2, "b f l -> l b f").contiguous()
        o_lstm2 = self.grn2o(x)
        o_lstm2 = einops.rearrange(o_lstm2, "b f l -> l b f").contiguous()

        h_lstm3 = self.grn3h(x)
        h_lstm3 = einops.rearrange(h_lstm3, "b f l -> l b f").contiguous()
        o_lstm3 = self.grn3o(x)
        o_lstm3 = einops.rearrange(o_lstm3, "b f l -> l b f").contiguous()

        return typing.cast(
            BWState3,
            {
                "hx": (h_lstm1, o_lstm1),
                "hx2": (h_lstm2, o_lstm2),
                "hx3": (h_lstm3, o_lstm3),
            },
        )
