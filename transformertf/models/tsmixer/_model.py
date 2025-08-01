"""
Implementation of the TSMixer model from
http://arxiv.org/abs/2303.06053
"""

from __future__ import annotations

import collections
import typing

import einops
import torch

from ...nn import VALID_ACTIVATIONS
from ._modules import (
    ConditionalFeatureMixer,
    ConditionalMixerBlock,
    FeatureProjection,
    MixerBlock,
    TemporalProjection,
)


class BasicTSMixerModel(torch.nn.Module):
    """
    This TSMixer model is a basic implementation of the TSMixer model, that takes
    no auxiliary information in time series forecasting.
    """

    def __init__(
        self,
        num_features: int,
        seq_len: int,
        out_seq_len: int | None = None,
        num_blocks: int = 4,
        d_fc: int = 512,
        d_hidden: int | None = None,
        dropout: float = 0.2,
        activation: VALID_ACTIVATIONS = "relu",
        norm: typing.Literal["batch", "layer"] = "batch",
    ):
        super().__init__()
        d_hidden = d_hidden or num_features

        self.residual_blocks = torch.nn.Sequential(*[
            MixerBlock(
                input_len=seq_len,
                num_features=num_features if i == 0 else d_hidden,
                dropout=dropout,
                activation=activation,
                d_fc=d_fc,
                norm=norm,
                out_num_features=d_hidden,
            )
            for i in range(num_blocks)
        ])

        if out_seq_len is None:
            self.fc = None
        else:
            self.fc = FeatureProjection(seq_len, out_seq_len)

    def forward(self, x: torch.Tensor, target_slice: int | None = None) -> torch.Tensor:
        x = self.residual_blocks(x)

        if target_slice is not None:
            x = x[..., :target_slice]

        if self.fc is not None:
            x = einops.rearrange(x, "b l c -> b c l")
            x = self.fc(x)
            x = einops.rearrange(x, "b c l -> b l c")

        return x


class TSMixerModel(torch.nn.Module):
    fc: torch.nn.Module | None
    """
    This TSMixer model is a full implementation of the TSMixer model, that takes
    static and continuous auxiliary information in time series forecasting.
    """

    def __init__(
        self,
        ctxt_seq_len: int,
        tgt_seq_len: int,
        num_feat: int,
        num_future_feat: int = 0,
        num_static_real_feat: int = 0,
        d_hidden: int | None = None,
        d_fc: int = 512,
        dropout: float = 0.2,
        norm: typing.Literal["batch", "layer"] = "batch",
        activation: typing.Literal["relu", "gelu"] = "relu",
        num_blocks: int = 4,
        output_dim: int | None = None,
    ):
        super().__init__()

        self.ctxt_seq_len = ctxt_seq_len
        self.tgt_seq_len = tgt_seq_len

        self.num_features = num_feat
        self.num_future_features = num_future_feat
        self.num_static_real_feat = num_static_real_feat
        self.d_hidden = d_hidden or num_feat
        d_hidden_ = self.d_hidden

        self.past_proj = TemporalProjection(ctxt_seq_len, tgt_seq_len)
        self.past_mixer = ConditionalFeatureMixer(
            input_len=tgt_seq_len,
            num_features=num_feat,
            num_static_features=num_static_real_feat,
            dropout=dropout,
            activation=activation,
            d_fc=d_fc,
            norm=norm,
            out_num_features=d_hidden_,
        )
        self.future_mixer = ConditionalFeatureMixer(
            input_len=tgt_seq_len,
            num_features=num_future_feat,
            num_static_features=num_static_real_feat,
            d_hidden=d_hidden_,
            dropout=dropout,
            activation=activation,
            d_fc=d_fc,
            norm=norm,
            out_num_features=d_hidden_,
        )

        residual_blocks = [
            ConditionalMixerBlock(
                input_len=tgt_seq_len,
                num_features=2 * d_hidden_ if i == 0 else d_hidden_,
                num_static_features=num_static_real_feat,
                d_hidden=d_hidden_,
                dropout=dropout,
                activation=activation,
                d_fc=d_fc,
                norm=norm,
                out_num_features=d_hidden_,
            )
            for i in range(num_blocks)
        ]
        self.residual_blocks = torch.nn.ModuleList(residual_blocks)

        if output_dim is not None:
            # use MLP for final prediction
            self.fc = torch.nn.Sequential(
                collections.OrderedDict({
                    "fc1": torch.nn.Linear(d_hidden_, d_fc),
                    "relu": torch.nn.ReLU(),
                    "fc2": torch.nn.Linear(d_fc, output_dim),
                })
            )
        else:
            self.fc = None

    def forward(
        self,
        past_covariates: torch.Tensor,
        future_covariates: torch.Tensor,
        static_covariates: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if static_covariates is None:
            static_covariates = torch.zeros((
                past_covariates.size(0),
                self.num_static_real_feat,
            ))

        # project past covariates to target sequence length
        past_covariates = self.past_proj(past_covariates)

        x_p = self.past_mixer(past_covariates, static_covariates)
        z_p = self.future_mixer(future_covariates, static_covariates)

        y = torch.cat([x_p, z_p], dim=-1)
        for block in self.residual_blocks:
            y = block(y, static_features=static_covariates)

        if self.fc is not None:
            y = self.fc(y)

        return y
