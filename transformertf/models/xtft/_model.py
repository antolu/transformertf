from __future__ import annotations

import torch

from ...nn import (
    AddNorm,
    GatedLinearUnit,
    GatedResidualNetwork,
    InterpretableMultiHeadAttention,
    VariableSelection,
)
from .._base_transformer import get_attention_mask

__all__ = ["xTFTModel"]


class xTFTModel(torch.nn.Module):  # noqa: N801
    def __init__(
        self,
        num_past_features: int,
        num_future_features: int,
        d_model: int = 300,
        hidden_continuous_dim: int = 8,
        num_heads: int = 4,
        num_lstm_layers: int = 2,
        dropout: float = 0.1,
        output_dim: int = 7,
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
        d_model : int, optional
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
        self.d_model = d_model
        self.hidden_continuous_dim = hidden_continuous_dim
        self.num_heads = num_heads
        self.num_lstm_layers = num_lstm_layers
        self.dropout = dropout
        self.output_dim = output_dim
        self.causal_attention = causal_attention

        self.enc_vs = VariableSelection(
            num_features=num_past_features,
            d_hidden=hidden_continuous_dim,
            d_model=d_model,
            dropout=dropout,
        )

        self.dec_vs = VariableSelection(
            num_features=num_future_features,
            d_hidden=hidden_continuous_dim,
            d_model=d_model,
            dropout=dropout,
        )

        self.enc_lstm = torch.nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_lstm_layers,
            dropout=dropout if num_lstm_layers > 1 else 0.0,
            batch_first=True,
        )

        self.dec_lstm = torch.nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_lstm_layers,
            dropout=dropout if num_lstm_layers > 1 else 0.0,
            batch_first=True,
        )

        self.enc_gate1 = GatedLinearUnit(d_model, dropout=dropout)
        self.dec_gate1 = self.enc_gate1

        self.enc_norm1 = AddNorm(d_model, trainable_add=False)
        self.dec_norm1 = self.enc_norm1

        self.attn = InterpretableMultiHeadAttention(
            num_heads=num_heads,
            d_model=d_model,
            dropout=dropout,
        )

        self.attn_gate1 = GatedLinearUnit(d_model, dropout=dropout)
        self.attn_norm1 = AddNorm(d_model, trainable_add=False)
        self.attn_grn = GatedResidualNetwork(
            input_dim=d_model,
            d_hidden=d_model,
            output_dim=d_model,
            dropout=dropout,
            projection="interpolate",
        )

        self.attn_gate2 = GatedLinearUnit(d_model, dropout=0.0)
        self.attn_norm2 = AddNorm(d_model, trainable_add=False)

        self.output_layer = torch.nn.Linear(d_model, output_dim)

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
        enc_seq_len = past_covariates.shape[1]
        dec_seq_len = future_covariates.shape[1]

        # variable selection
        enc_vs, enc_weights = self.enc_vs(past_covariates)
        dec_vs, dec_weights = self.dec_vs(future_covariates)

        # initialize LSTM states with static features
        enc_input, hx = self.enc_lstm(enc_vs)
        dec_input, hx = self.dec_lstm(dec_vs, hx)

        # encoder and decoder post-processing
        enc_output = self.enc_gate1(enc_input)
        enc_output = self.enc_norm1(enc_output, enc_vs)

        dec_output = self.dec_gate1(dec_input)
        dec_output = self.dec_norm1(dec_output, dec_vs)

        attn_mask = get_attention_mask(
            encoder_lengths,
            decoder_lengths,
            max_encoder_length=enc_seq_len,
            max_decoder_length=dec_seq_len,
            causal_attention=self.causal_attention,
            encoder_alignment="right",
        )

        attn_input = torch.cat([enc_output, dec_output], dim=1)

        # multi-head attention and post-processing
        attn_output, attn_weights = self.attn(
            attn_input[:, enc_seq_len:], attn_input, attn_input, mask=attn_mask
        )
        attn_output = self.attn_gate1(attn_output)
        attn_output = self.attn_norm1(attn_output, attn_input[:, enc_seq_len:])
        attn_output = self.attn_grn(attn_output)

        # final post-processing and output
        attn_output = self.attn_gate2(attn_output)
        attn_output = self.attn_norm2(attn_output, dec_output)

        output = self.output_layer(attn_output)

        return {
            "output": output,
            "enc_weights": enc_weights,
            "dec_weights": dec_weights,
            "attn_weights": attn_weights,
        }
