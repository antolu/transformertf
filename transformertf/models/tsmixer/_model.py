"""
Implementation of the TSMixer model from
http://arxiv.org/abs/2303.06053
"""
from __future__ import annotations

import typing

import einops
import torch


from ._modules import (
    MixerBlock,
    ConditionalFeatureMixer,
    ConditionalMixerBlock,
    TemporalProjection,
)


class BasicTSMixer(torch.nn.Module):
    """
    This TSMixer model is a basic implementation of the TSMixer model, that takes
    no auxiliary information in time series forecasting.
    """

    def __init__(
        self,
        num_features: int,
        seq_len: int,
        out_seq_len: int | typing.Literal["headless"],
        num_blocks: int = 4,
        fc_dim: int = 512,
        dropout: float = 0.2,
        activation: typing.Literal["relu", "gelu"] = "relu",
        norm: typing.Literal["batch", "layer"] = "batch",
    ):
        super().__init__()

        self.residual_blocks = torch.nn.Sequential(
            *[
                MixerBlock(
                    num_features=num_features,
                    dropout=dropout,
                    activation=activation,
                    fc_dim=fc_dim,
                    norm=norm,
                )
                for _ in range(num_blocks)
            ]
        )

        if out_seq_len == "headless":
            self.fc = None
        else:
            self.fc = TemporalProjection(seq_len, out_seq_len)

    def forward(
        self, x: torch.Tensor, target_slice: int | None = None
    ) -> torch.Tensor:
        x = self.residual_blocks(x)

        if target_slice is not None:
            x = x[..., :target_slice]

        if self.fc is not None:
            x = einops.rearrange(x, "b l c -> b c l")
            x = self.fc(x)
            x = einops.rearrange(x, "b c l -> b l c")

        return x


class TSMixer(torch.nn.Module):
    """
    This TSMixer model is a full implementation of the TSMixer model, that takes
    static and continuous auxiliary information in time series forecasting.
    """

    def __init__(
        self,
        num_features: int,
        num_static_features: int = 0,
        seq_len: int = 512,
        out_seq_len: int = 128,
        dropout: float = 0.2,
        activation: typing.Literal["relu", "gelu"] = "relu",
        fc_dim: int = 512,
        norm: typing.Literal["batch", "layer"] = "batch",
        num_blocks: int = 4,
    ):
        super().__init__()

        self.num_features = num_features
        self.num_static_features = num_static_features

        # self.past_mixer = ConditionalMixerBlock()
        self.future_mixer = ConditionalFeatureMixer(
            num_features=num_features,
            num_static_features=num_static_features,
            dropout=dropout,
            activation=activation,
            fc_dim=fc_dim,
            norm=norm,
        )

        residual_blocks = [
            ConditionalMixerBlock(
                num_features=2 * num_features,
                num_static_features=num_static_features,
                dropout=dropout,
                activation=activation,
                fc_dim=fc_dim,
                norm=norm,
                out_num_features=num_features,
            )
        ]
        residual_blocks += [
            ConditionalMixerBlock(
                num_features=num_features,
                num_static_features=num_static_features,
                dropout=dropout,
                activation=activation,
                fc_dim=fc_dim,
                norm=norm,
            )
            for _ in range(num_blocks - 1)
        ]
        self.residual_blocks = torch.nn.Sequential(*residual_blocks)

        self.fc = TemporalProjection(num_features, out_seq_len)

    def forward(
        self,
        past_covariates: torch.Tensor,
        future_covariates: torch.Tensor,
        static_covariates: torch.Tensor | None = None,
        target_slice: int | slice | None = None,
    ) -> torch.Tensor:
        if static_covariates is None:
            static_covariates = torch.zeros(
                (past_covariates.size(0), self.num_static_features)
            )

        x_p = past_covariates
        z_p = self.future_mixer(future_covariates, static_covariates)

        y = self.residual_blocks(torch.cat([x_p, z_p], dim=-1))

        if target_slice is not None:
            y = y[..., target_slice]

        y = einops.rearrange(y, "b l c -> b c l")
        y = self.fc(y)
        y = einops.rearrange(y, "b c l -> b l c")

        return y
