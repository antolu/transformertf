"""
Implementation of the TSMixer model from
http://arxiv.org/abs/2303.06053
"""
from __future__ import annotations

import collections
import typing

import einops
import torch


from ..tsmixer import TSMixer, BasicTSMixer
from ..phylstm import PhyLSTM3Output


class PhyTSMixer(torch.nn.Module):
    def __init__(
        self,
        num_features: int,
        num_future_features: int,
        input_len: int = 600,
        output_len: int = 200,
        fc_dim: int = 1024,
        hidden_dim: int | None = None,
        hidden_dim_2: int = 256,
        num_blocks: int = 8,
        dropout: float = 0.1,
        norm: typing.Literal["batch", "layer"] = "batch",
        activation: typing.Literal["relu", "gelu"] = "relu",
    ):
        super().__init__()

        self.output_len = output_len
        hidden_dim = hidden_dim or num_future_features

        self.ts1 = TSMixer(
            num_feat=num_features,
            num_future_feat=num_future_features,
            seq_len=input_len,
            out_seq_len=output_len,
            out_dim=hidden_dim_2,
            fc_dim=fc_dim,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            dropout=dropout,
            norm=norm,
            activation=activation,
        )

        self.fc1 = torch.nn.Sequential(
            collections.OrderedDict(
                [
                    (
                        "fc11",
                        torch.nn.Linear(hidden_dim_2 or num_features, fc_dim),
                    ),
                    ("lrelu1", torch.nn.LeakyReLU()),
                    ("ln1", torch.nn.LayerNorm(fc_dim)),
                    ("fc12", torch.nn.Linear(fc_dim, 3)),
                ]
            )
        )

        self.ts2 = BasicTSMixer(
            num_features=3,
            seq_len=output_len,
            fc_dim=fc_dim,
            hidden_dim=hidden_dim_2,
            num_blocks=num_blocks,
            dropout=dropout,
            norm=norm,
            activation=activation,
        )

        self.fc2 = torch.nn.Sequential(
            collections.OrderedDict(
                [
                    (
                        "fc21",
                        torch.nn.Linear(
                            hidden_dim_2 or num_features, hidden_dim
                        ),
                    ),
                    ("lrelu2", torch.nn.LeakyReLU()),
                    ("ln1", torch.nn.LayerNorm(hidden_dim)),
                    ("fc22", torch.nn.Linear(hidden_dim, 1)),
                ]
            )
        )

        self.g_plus_x = torch.nn.Sequential(
            torch.nn.Linear(2, hidden_dim_2),
            torch.nn.LayerNorm(hidden_dim_2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim_2, 1),
        )

        self.ts3 = BasicTSMixer(
            num_features=2,
            seq_len=output_len,
            fc_dim=fc_dim,
            hidden_dim=hidden_dim_2,
            num_blocks=num_blocks,
            dropout=dropout,
            norm=norm,
            activation=activation,
        )

        self.fc3 = torch.nn.Sequential(
            collections.OrderedDict(
                [
                    (
                        "fc31",
                        torch.nn.Linear(
                            hidden_dim_2 or num_features, hidden_dim
                        ),
                    ),
                    ("lrelu3", torch.nn.LeakyReLU()),
                    ("ln1", torch.nn.LayerNorm(hidden_dim)),
                    ("fc32", torch.nn.Linear(hidden_dim, 1)),
                ]
            )
        )

    def forward(
        self,
        past_covariates: torch.Tensor,
        future_covariates: torch.Tensor,
        static_covariates: torch.Tensor | None = None,
    ) -> PhyLSTM3Output:
        z = self.ts1(past_covariates, future_covariates, static_covariates)
        z = self.fc1(z)

        dz_dt = torch.gradient(z, dim=1)[0]

        g = self.fc2(self.ts2(z))
        g_gamma_x = self.g_plus_x(torch.cat([g, future_covariates], dim=2))

        dz_dt_0 = einops.repeat(
            dz_dt[:, 0, 1, None], "b f -> b t f", t=self.output_len
        )

        delta_z_dot = dz_dt[..., 1, None] - dz_dt_0
        phi = torch.cat([delta_z_dot, z[..., 2, None]], dim=2)

        dr_dt = self.fc3(self.ts3(phi))

        return typing.cast(
            PhyLSTM3Output,
            dict(
                z=z,
                dz_dt=dz_dt,
                g=g,
                g_gamma_x=g_gamma_x,
                dr_dt=dr_dt,
            ),
        )
