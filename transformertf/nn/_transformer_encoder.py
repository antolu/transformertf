from __future__ import annotations

import numpy as np
import torch

from ._get_activation import VALID_ACTIVATIONS
from ._mlp import MLP

__all__ = ["SimplePositionalEncoding", "TransformerEncoder", "generate_mask"]


class TransformerEncoder(torch.nn.Module):
    """
    A transformer encoder model for sequence-to-sequence tasks.

    This model consists of:
    - Feature embedding layer to project input features to model dimension
    - Positional encoding for sequence position information
    - Multi-layer transformer encoder
    - MLP head for final predictions

    Parameters
    ----------
    num_features : int
        Number of input features per time step
    seq_len : int
        Length of input sequences
    d_model : int, default=128
        Model dimension (embedding size)
    num_heads : int, default=8
        Number of attention heads
    num_encoder_layers : int, default=6
        Number of transformer encoder layers
    dropout : float, default=0.1
        Dropout probability
    activation : VALID_ACTIVATIONS, default="relu"
        Activation function to use
    fc_dim : int | tuple[int, ...], default=1024
        Dimension(s) of the MLP head
    output_dim : int, default=7
        Output dimension
    """

    def __init__(
        self,
        num_features: int,
        seq_len: int,
        d_model: int = 128,
        num_heads: int = 8,
        num_encoder_layers: int = 6,
        dropout: float = 0.1,
        activation: VALID_ACTIVATIONS = "relu",
        fc_dim: int | tuple[int, ...] = 1024,
        output_dim: int = 7,
    ):
        super().__init__()

        self.num_features = num_features
        self.seq_len = seq_len
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_encoder_layers = num_encoder_layers
        self.dropout = dropout
        self.activation = activation
        self.fc_dim = fc_dim

        self.feature_embedding = torch.nn.Linear(
            self.num_features, self.d_model
        )  # [bs, seq_len, n_dim_model]

        self.pos_encoder = SimplePositionalEncoding(
            dim_model=self.d_model, dropout=self.dropout
        )

        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.num_heads,
            dropout=self.dropout,
            activation=self.activation,
            batch_first=True,
        )
        norm = torch.nn.LayerNorm(self.d_model)
        self.transformer = torch.nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=self.num_encoder_layers,
            norm=norm,
        )
        self.fc = MLP(
            input_dim=self.d_model,
            hidden_dim=self.fc_dim,
            output_dim=output_dim,
            dropout=self.dropout,
            activation=self.activation,
        )  # [bs, seq_len, output_dim]

        self.src_mask: torch.Tensor
        self.register_buffer("src_mask", generate_mask(self.seq_len))

    def forward(
        self,
        source: torch.Tensor,
        src_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass through the transformer encoder.

        Parameters
        ----------
        source : torch.Tensor
            Input tensor of shape [batch_size, seq_len, num_features]
        src_mask : torch.Tensor | None, optional
            Source mask tensor. If None, uses the default registered mask.

        Returns
        -------
        torch.Tensor
            Output tensor of shape [batch_size, seq_len, output_dim]
        """
        x = self.feature_embedding(source)
        x = self.pos_encoder(x)

        encoded = self.transformer(x)

        return self.fc(encoded)


class SimplePositionalEncoding(torch.nn.Module):
    """
    Simple positional encoding using sine and cosine functions.

    Parameters
    ----------
    dim_model : int
        Model dimension
    dropout : float, default=0.1
        Dropout probability
    max_len : int, default=5000
        Maximum sequence length
    """

    def __init__(self, dim_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()

        self.dropout = torch.nn.Dropout(p=dropout) if dropout > 0 else None
        pe = torch.zeros(max_len, dim_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim_model, 2) * (-np.log(10000.0) / dim_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1).to(torch.float32)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Creates a basic positional encoding"""
        x += self.pe[: x.size(0), :]
        return self.dropout(x) if self.dropout is not None else x


def generate_mask(size: int) -> torch.Tensor:
    """
    Generate a square mask for the sequence.

    Parameters
    ----------
    size : int
        Size of the square mask

    Returns
    -------
    torch.Tensor
        Mask tensor
    """
    mask = torch.triu(torch.ones(size, size), diagonal=1)
    return mask.masked_fill(mask == 1, float("-inf"))
