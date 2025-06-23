from __future__ import annotations

import einops.layers
import einops.layers.torch
import torch

from ...nn import (
    AddNorm,
    GatedLinearUnit,
    GatedResidualNetwork,
    InterpretableMultiHeadAttention,
    VariableSelection,
)
from .._base_transformer import get_attention_mask
from ._conv import DownsampleBlock, UpsampleBlock

__all__ = ["xTFTConvModel"]


class xTFTConvModel(torch.nn.Module):  # noqa: N801
    def __init__(
        self,
        num_past_features: int,
        num_future_features: int,
        n_dim_model: int = 300,
        hidden_continuous_dim: int = 8,
        num_heads: int = 4,
        num_lstm_layers: int = 2,
        dropout: float = 0.1,
        output_dim: int = 7,
        downsample_factor: int = 2,
        *,
        causal_attention: bool = True,
    ):
        """
        Implementation of the Temporal Fusion Transformer architecture.

        Parameters
        ----------
        num_features : int
            Number of continuous features / covariates (time series)
        num_static_features : int, optional
            Number of static features, by default 0. Currently not used.
        n_dim_model : int, optional
            Dimension of the model, by default 300. The most important
            hyperparameter, as it determines the model capacity.
        variable_selection_dim : int, optional
            Dimension of the variable selection network, by default 100.
        hidden_continuous_dim : int, optional
            Dimension of the hidden continuous features, by default 8.
        num_heads : int, optional
            Number of attention heads, by default 4.
        num_lstm_layers : int, optional
            Number of LSTM layers, by default 2.
        dropout : float, optional
            Dropout rate, by default 0.1.
        output_dim : int, optional
            Output dimension, by default 7. The output dimension is
            typically 7 for quantile regression, but can be different
            for other tasks. For MSE loss, the output dimension should
            be 1.
        """
        super().__init__()
        self.num_past_features = num_past_features
        self.num_future_features = num_future_features
        self.n_dim_model = n_dim_model
        self.hidden_continuous_dim = hidden_continuous_dim
        self.num_heads = num_heads
        self.num_lstm_layers = num_lstm_layers
        self.dropout = dropout
        self.output_dim = output_dim
        self.downsample_factor = downsample_factor
        self.causal_attention = causal_attention

        enc_vs_out_dim = min(n_dim_model, num_past_features * hidden_continuous_dim)
        self.enc_vs = VariableSelection(
            n_features=num_past_features,
            hidden_dim=hidden_continuous_dim,
            n_dim_model=enc_vs_out_dim,
            dropout=dropout,
        )

        dec_vs_out_dim = min(n_dim_model, num_future_features * hidden_continuous_dim)
        self.dec_vs = VariableSelection(
            n_features=num_future_features,
            hidden_dim=hidden_continuous_dim,
            n_dim_model=dec_vs_out_dim,
            dropout=dropout,
        )

        self.flip_conv = einops.layers.torch.Rearrange("b t f -> b f t")
        self.reverse_flip_conv = einops.layers.torch.Rearrange("b f t -> b t f")

        self.dec_ds = torch.nn.Sequential(
            *[
                DownsampleBlock(
                    in_channels=dec_vs_out_dim,
                    out_channels=dec_vs_out_dim
                    if i < downsample_factor - 2
                    else n_dim_model,
                    kernel_size=3,
                    downsample=2,
                )
                for i in range(downsample_factor - 1)
            ],
        )
        self.enc_ds = torch.nn.Sequential(
            *[
                DownsampleBlock(
                    in_channels=enc_vs_out_dim,
                    out_channels=enc_vs_out_dim
                    if i < downsample_factor - 2
                    else n_dim_model,
                    kernel_size=3,
                    downsample=2,
                )
                for i in range(downsample_factor - 1)
            ],
        )

        self.enc_lstm = torch.nn.LSTM(
            input_size=n_dim_model,
            hidden_size=n_dim_model,
            num_layers=num_lstm_layers,
            dropout=dropout if num_lstm_layers > 1 else 0.0,
            batch_first=True,
        )

        self.dec_lstm = torch.nn.LSTM(
            input_size=n_dim_model,
            hidden_size=n_dim_model,
            num_layers=num_lstm_layers,
            dropout=dropout if num_lstm_layers > 1 else 0.0,
            batch_first=True,
        )

        self.enc_gate1 = GatedLinearUnit(n_dim_model, dropout=dropout)
        self.dec_gate1 = self.enc_gate1

        self.enc_norm1 = AddNorm(n_dim_model, trainable_add=False)
        self.dec_norm1 = self.enc_norm1

        self.attn = InterpretableMultiHeadAttention(
            n_heads=num_heads,
            n_dim_model=n_dim_model,
            dropout=dropout,
        )

        self.attn_gate1 = GatedLinearUnit(n_dim_model, dropout=dropout)
        self.attn_norm1 = AddNorm(n_dim_model, trainable_add=False)
        self.attn_grn = GatedResidualNetwork(
            input_dim=n_dim_model,
            hidden_dim=n_dim_model,
            output_dim=n_dim_model,
            dropout=dropout,
            projection="interpolate",
        )

        self.attn_gate2 = GatedLinearUnit(n_dim_model, dropout=0.0)
        self.attn_norm2 = AddNorm(n_dim_model, trainable_add=False)

        upsample_dim = min(n_dim_model, num_future_features * hidden_continuous_dim)
        self.output_layer = torch.nn.Linear(n_dim_model, upsample_dim)

        self.dec_us = torch.nn.Sequential(
            *[
                UpsampleBlock(n_channels=upsample_dim, kernel_size=3, upsample=2)
                for _ in range(downsample_factor - 1)
            ],
        )

        self.output_layer2 = torch.nn.Linear(upsample_dim, output_dim)

    def forward(
        self,
        past_covariates: torch.Tensor,
        future_covariates: torch.Tensor,
        encoder_lengths: torch.Tensor | None = None,
        decoder_lengths: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass.

        Parameters
        ----------
        past_covariates : torch.Tensor
            [batch_size, ctxt_seq_len, num_features]
        future_covariates : torch.Tensor
            [batch_size, tgt_seq_len, num_features]

        Returns
        -------
        dict[str, torch.Tensor]
            Predictions and attention weights.
            Keys: "output", "enc_weights", "dec_weights", "attn_weights"
            correspond to the output predictions, encoder variable selection
            weights, decoder variable selection weights, and attention
            weights, respectively.
        """
        past_covariates.shape[0]
        past_covariates.shape[1]
        future_covariates.shape[1]

        # variable selection
        enc_vs, enc_weights = self.enc_vs(past_covariates)
        dec_vs, dec_weights = self.dec_vs(future_covariates)

        enc_vs_ds = self.flip_conv(enc_vs)
        enc_vs_ds = self.enc_ds(enc_vs_ds)
        enc_vs_ds = self.reverse_flip_conv(enc_vs_ds)

        dec_vs_ds = self.flip_conv(dec_vs)
        seq_lens = [dec_vs_ds.shape[-1]]
        for block in self.dec_ds:
            dec_vs_ds = block(dec_vs_ds)
            seq_lens.append(dec_vs_ds.shape[-1])
        dec_vs_ds = self.reverse_flip_conv(dec_vs_ds)

        enc_input, hx = self.enc_lstm(enc_vs_ds)
        dec_input, hx = self.dec_lstm(dec_vs_ds, hx)

        # encoder and decoder post-processing
        enc_output = self.enc_gate1(enc_input)
        enc_output = self.enc_norm1(enc_output, enc_vs_ds)

        dec_output = self.dec_gate1(dec_input)
        dec_output = self.dec_norm1(dec_output, dec_vs_ds)

        attn_mask = (
            get_attention_mask(
                torch.round(encoder_lengths / self.downsample_factor),
                torch.round(decoder_lengths / self.downsample_factor),
                max_encoder_length=enc_vs_ds.shape[1],
                max_decoder_length=dec_vs_ds.shape[1],
                causal_attention=self.causal_attention,
            )
            if encoder_lengths is not None and decoder_lengths is not None
            else None
        )

        attn_input = torch.cat([enc_output, dec_output], dim=1)

        # multi-head attention and post-processing
        attn_output, attn_weights = self.attn(
            attn_input[:, enc_vs_ds.shape[1] :], attn_input, attn_input, mask=attn_mask
        )
        attn_output = self.attn_gate1(attn_output)
        attn_output = self.attn_norm1(attn_output, attn_input[:, enc_vs_ds.shape[1] :])
        attn_output = self.attn_grn(attn_output)

        # final post-processing and output
        attn_output = self.attn_gate2(attn_output)
        attn_output = self.attn_norm2(attn_output, dec_output)

        output = self.output_layer(attn_output)

        output = self.flip_conv(output)
        for seq_len, block in zip(
            list(reversed(seq_lens))[1:], self.dec_us, strict=True
        ):
            output = block(output, target_length=seq_len)
        output = self.reverse_flip_conv(output)

        output = self.output_layer2(output)

        return {
            "output": output,
            "enc_weights": enc_weights,
            "dec_weights": dec_weights,
            "attn_weights": attn_weights,
        }
