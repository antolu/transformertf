from __future__ import annotations

import einops
import torch

from ...nn import (
    AddNorm,
    GatedLinearUnit,
    GatedResidualNetwork,
    InterpretableMultiHeadAttention,
    VariableSelection,
)

__all__ = ["TemporalFusionTransformerModel"]


class TemporalFusionTransformerModel(torch.nn.Module):
    def __init__(
        self,
        num_past_features: int,
        num_future_features: int,
        ctxt_seq_len: int,
        tgt_seq_len: int,
        num_static_features: int = 0,
        n_dim_model: int = 300,
        hidden_continuous_dim: int = 8,
        num_heads: int = 4,
        num_lstm_layers: int = 2,
        dropout: float = 0.1,
        output_dim: int = 7,
    ):
        """
        Implementation of the Temporal Fusion Transformer architecture.

        Parameters
        ----------
        num_features : int
            Number of continuous features / covariates (time series)
        ctxt_seq_len : int
            Length of the context sequence, in other words the encoder
            sequence length.
        tgt_seq_len : int
            Length of the target sequence, in other words the decoder
            sequence length. This is the prediction horizon.
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
        self.ctxt_seq_len = ctxt_seq_len
        self.tgt_seq_len = tgt_seq_len
        self.num_static_features = num_static_features  # not used
        self.n_dim_model = n_dim_model
        self.hidden_continuous_dim = hidden_continuous_dim
        self.num_heads = num_heads
        self.num_lstm_layers = num_lstm_layers
        self.dropout = dropout
        self.output_dim = output_dim

        # TODO: static covariate embeddings
        self.static_vs = VariableSelection(
            n_features=num_static_features,
            hidden_dim=hidden_continuous_dim,
            n_dim_model=n_dim_model,
            dropout=dropout,
        )

        self.enc_vs = VariableSelection(
            n_features=num_past_features,
            hidden_dim=hidden_continuous_dim,
            n_dim_model=n_dim_model,
            context_size=n_dim_model,
            dropout=dropout,
        )

        self.dec_vs = VariableSelection(
            n_features=num_future_features,
            hidden_dim=hidden_continuous_dim,
            n_dim_model=n_dim_model,
            context_size=n_dim_model,
            dropout=dropout,
        )

        self.static_ctxt_vs = basic_grn(n_dim_model, dropout)
        self.static_ctxt_enrichment = basic_grn(n_dim_model, dropout)

        self.lstm_init_hidden = basic_grn(n_dim_model, dropout)
        self.lstm_init_cell = basic_grn(n_dim_model, dropout)

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

        self.static_enrichment = GatedResidualNetwork(
            input_dim=n_dim_model,
            hidden_dim=n_dim_model,
            output_dim=n_dim_model,
            context_dim=n_dim_model,
            dropout=dropout,
            projection="interpolate",
        )

        self.attn = InterpretableMultiHeadAttention(
            n_heads=num_heads,
            n_dim_model=n_dim_model,
            dropout=dropout,
        )

        self.attn_gate1 = GatedLinearUnit(n_dim_model, dropout=dropout)
        self.attn_norm1 = AddNorm(n_dim_model, trainable_add=False)
        self.attn_grn = basic_grn(n_dim_model, dropout)

        self.attn_gate2 = GatedLinearUnit(n_dim_model, dropout=0.0)
        self.attn_norm2 = AddNorm(n_dim_model, trainable_add=False)

        self.output_layer = torch.nn.Linear(n_dim_model, output_dim)

    def forward(
        self,
        past_covariates: torch.Tensor,
        future_covariates: torch.Tensor,
        static_covariates: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass.

        Parameters
        ----------
        past_covariates : torch.Tensor
            [batch_size, ctxt_seq_len, num_features]
        future_covariates : torch.Tensor
            [batch_size, tgt_seq_len, num_features]
        static_covariates : torch.Tensor, optional
            [batch_size, num_static_features]

        Returns
        -------
        dict[str, torch.Tensor]
            Predictions and attention weights.
            Keys: "output", "enc_weights", "dec_weights", "attn_weights"
            correspond to the output predictions, encoder variable selection
            weights, decoder variable selection weights, and attention
            weights, respectively.
        """
        batch_size = past_covariates.shape[0]
        enc_seq_len = past_covariates.shape[1]
        dec_seq_len = future_covariates.shape[1]

        if static_covariates is None:
            # normally static embedding is computed using an
            # embedding layer, but for simplicity we just use zeros
            static_embedding = torch.zeros(
                batch_size,
                self.n_dim_model,
                device=past_covariates.device,
            )  # static covariate embeddings
            static_variable_selection = torch.zeros(
                batch_size, 0, device=past_covariates.device
            )  # static variable selection weights
        else:
            static_embedding = static_covariates
            static_embedding, static_variable_selection = self.static_vs(
                static_embedding
            )

        # static variable selection
        static_ctxt_vs = self.static_ctxt_vs(static_embedding)
        enc_static_context = einops.repeat(
            static_ctxt_vs,
            "b f -> b t f",
            t=enc_seq_len,
        )
        dec_static_context = einops.repeat(
            static_ctxt_vs,
            "b f -> b t f",
            t=dec_seq_len,
        )

        # variable selection
        enc_vs, enc_weights = self.enc_vs(past_covariates, enc_static_context)
        dec_vs, dec_weights = self.dec_vs(
            future_covariates,
            dec_static_context,  # skip target feature since it's zeros
        )

        # initialize LSTM states with static features
        lstm_hidden = self.lstm_init_hidden(static_embedding)
        lstm_hidden = einops.repeat(
            lstm_hidden, "h i -> n h i", n=self.num_lstm_layers
        ).contiguous()
        lstm_cell = self.lstm_init_cell(static_embedding)
        lstm_cell = einops.repeat(
            lstm_cell, "h i -> n h i", n=self.num_lstm_layers
        ).contiguous()

        # encoder and decoder LSTM
        enc_input, hx = self.enc_lstm(enc_vs, (lstm_hidden, lstm_cell))
        dec_input, _ = self.dec_lstm(dec_vs, hx)

        # encoder and decoder post-processing
        enc_output = self.enc_gate1(enc_input)
        enc_output = self.enc_norm1(enc_output, enc_vs)

        dec_output = self.dec_gate1(dec_input)
        dec_output = self.dec_norm1(dec_output, dec_vs)

        # add static features LSTM outputs
        static_context = self.static_ctxt_enrichment(static_embedding)
        attn_input = self.static_enrichment(
            torch.cat([enc_output, dec_output], dim=1),
            einops.repeat(
                static_context,
                "b f -> b t f",
                t=enc_seq_len + dec_seq_len,
            ),
        )

        # multi-head attention and post-processing
        attn_output, attn_weights = self.attn(
            attn_input[:, enc_seq_len:], attn_input, attn_input
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
            "static_weights": static_variable_selection,
        }


def basic_grn(dim: int, dropout: float) -> GatedResidualNetwork:
    """
    Utility function improve readability of TemporalFusionTransformer.__init__.

    Parameters
    ----------
    dim : int
        Dimension of the model, in, hidden and output dimensions.
    dropout : float
        Dropout rate.

    Returns
    -------
    GatedResidualNetwork
        Gated residual network with input, hidden, and output dimensions equal to `dim`.
    """
    return GatedResidualNetwork(
        input_dim=dim,
        hidden_dim=dim,
        output_dim=dim,
        dropout=dropout,
        projection="interpolate",
    )
