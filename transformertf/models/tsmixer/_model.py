from __future__ import annotations

import torch
import einops
import typing


class ResidualBlock(torch.nn.Module):
    def __init__(
        self,
        num_features: int,
        seq_len: int,
        norm: typing.Literal["batch", "layer"],
        activation: typing.Literal["relu", "gelu"],
        dropout: float,
        fc_dim: int,
    ):
        super().__init__()

        if norm == "batch":
            norm_type = torch.nn.BatchNorm2d
        elif norm == "layer":
            norm_type = torch.nn.LayerNorm
        else:
            raise ValueError(f"norm must be 'batch' or 'layer', not {norm}")

        if activation == "relu":
            activation_mod = torch.nn.ReLU
        elif activation == "gelu":
            activation_mod = torch.nn.GELU
        else:
            raise ValueError(
                f"activation must be 'relu' or 'gelu', not {activation}"
            )

        self.dropout = torch.nn.Dropout(dropout)
        self.activation = activation_mod()

        self.temp_norm = norm_type(1)
        self.fc1 = torch.nn.Linear(num_features, num_features)

        self.feat_norm = norm_type(1)
        self.feat_fc1 = torch.nn.Linear(num_features, fc_dim)
        self.feat_fc2 = torch.nn.Linear(fc_dim, num_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Temporal Linear
        # need to rearrange to get the features in the right place
        x = einops.rearrange(inputs, "b l f -> b 1 l f")
        x = self.temp_norm(x)
        x = einops.rearrange(x, "b 1 l f -> b l f")

        x = self.fc1(x)
        x = self.activation(x)
        # x = einops.rearrange(x, "b l c -> b c l")
        x = self.dropout(x)
        res = x + inputs

        # Feature Linear
        # need to rearrange to get the features in the right place
        x = einops.rearrange(res, "b l f -> b 1 l f")
        x = self.feat_norm(x)
        x = einops.rearrange(x, "b 1 l f -> b l f")

        x = self.feat_fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.feat_fc2(x)
        x = self.dropout(x)
        x = x + res

        return x


class TSMixer(torch.nn.Module):
    def __init__(
        self,
        num_features: int,
        seq_len: int,
        out_seq_len: int,
        dropout: float,
        activation: typing.Literal["relu", "gelu"],
        fc_dim: int,
        norm: typing.Literal["batch", "layer"] = "batch",
        num_blocks: int = 4,
    ):
        super().__init__()

        self.residual_blocks = torch.nn.Sequential(
            *[
                ResidualBlock(
                    num_features=num_features,
                    seq_len=seq_len,
                    dropout=dropout,
                    activation=activation,
                    fc_dim=fc_dim,
                    norm=norm,
                )
                for _ in range(num_blocks)
            ]
        )

        self.fc = torch.nn.Linear(seq_len, out_seq_len)

    def forward(
        self, x: torch.Tensor, target_slice: int | None = None
    ) -> torch.Tensor:
        x = self.residual_blocks(x)

        if target_slice is not None:
            x = x[:, :target_slice]

        x = einops.rearrange(x, "b l c -> b c l")
        x = self.fc(x)
        x = einops.rearrange(x, "b c l -> b l c")

        return x
