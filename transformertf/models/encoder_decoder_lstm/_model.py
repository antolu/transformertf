from __future__ import annotations

import typing

import torch

from ...nn import MLP, VALID_ACTIVATIONS

__all__ = ["EncoderDecoderLSTMModel"]

HIDDEN_STATE = tuple[torch.Tensor, torch.Tensor]


class EncoderDecoderLSTMModel(torch.nn.Module):
    """
    Encoder-Decoder LSTM model for sequence-to-sequence time series forecasting.

    This model implements an encoder-decoder architecture using LSTM networks for
    time series forecasting tasks. The encoder processes past features to create
    a context representation, which is then used to initialize the decoder that
    generates predictions for future time steps.

    The architecture consists of:
    1. Encoder LSTM: Processes past feature sequences
    2. Decoder LSTM: Generates future predictions using encoder context
    3. MLP Head: Configurable multi-layer perceptron for final output projection

    Parameters
    ----------
    num_past_features : int
        Number of input features in the past sequence (encoder input).
    num_future_features : int
        Number of input features in the future sequence (decoder input).
        Can be different from num_past_features to handle different feature sets.
    encoder_hidden_size : int, default=128
        Hidden size of the encoder LSTM layers.
    decoder_hidden_size : int, default=128
        Hidden size of the decoder LSTM layers.
    num_encoder_layers : int, default=2
        Number of LSTM layers in the encoder.
    num_decoder_layers : int, default=2
        Number of LSTM layers in the decoder.
    dropout : float, default=0.1
        Dropout probability for LSTM layers.
    mlp_hidden_dim : int | tuple[int, ...] | None, default=None
        Hidden dimensions for the MLP head. If None, uses a single linear layer.
        If int, creates one hidden layer. If tuple, creates multiple hidden layers.
    output_dim : int, default=1
        Output dimension of the model.
    mlp_activation : VALID_ACTIVATIONS, default="relu"
        Activation function for the MLP head.
    mlp_dropout : float, default=0.1
        Dropout probability for the MLP head.

    Attributes
    ----------
    encoder : torch.nn.LSTM
        The encoder LSTM network.
    decoder : torch.nn.LSTM
        The decoder LSTM network.
    mlp_head : torch.nn.Linear | MLP
        The output projection head.

    Examples
    --------
    >>> import torch
    >>> from transformertf.models.encoder_decoder_lstm import EncoderDecoderLSTMModel
    >>>
    >>> # Create model with different past/future features
    >>> model = EncoderDecoderLSTMModel(
    ...     num_past_features=10,
    ...     num_future_features=5,
    ...     encoder_hidden_size=64,
    ...     decoder_hidden_size=64,
    ...     mlp_hidden_dim=(32, 16),
    ...     output_dim=3
    ... )
    >>>
    >>> # Forward pass
    >>> batch_size, past_len, future_len = 32, 100, 50
    >>> past_seq = torch.randn(batch_size, past_len, 10)
    >>> future_seq = torch.randn(batch_size, future_len, 5)
    >>> output = model(past_seq, future_seq)
    >>> print(output.shape)  # torch.Size([32, 50, 3])

    Notes
    -----
    **Training vs Inference:**

    Both training and inference require past and future sequences to be provided:
    - Encoder processes the past sequence to create context representation
    - Decoder uses the provided future sequence with encoder context to generate predictions
    - Future features are always required for both training and inference

    **State Transfer:**

    The encoder's final hidden states are used to initialize the decoder,
    enabling the transfer of learned representations from past to future.

    **Input Requirements:**

    - past_sequence: (batch_size, past_seq_len, num_past_features)
    - future_sequence: (batch_size, future_seq_len, num_future_features)

    The model supports different sequence lengths and feature dimensions
    for maximum flexibility in various forecasting scenarios.
    """

    def __init__(
        self,
        num_past_features: int,
        num_future_features: int,
        encoder_hidden_size: int = 128,
        decoder_hidden_size: int = 128,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        dropout: float = 0.1,
        mlp_hidden_dim: int | tuple[int, ...] | None = None,
        output_dim: int = 1,
        mlp_activation: VALID_ACTIVATIONS = "relu",
        mlp_dropout: float = 0.1,
    ):
        super().__init__()

        self.num_past_features = num_past_features
        self.num_future_features = num_future_features
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dropout = dropout
        self.output_dim = output_dim

        # Encoder LSTM
        self.encoder = torch.nn.LSTM(
            input_size=num_past_features,
            hidden_size=encoder_hidden_size,
            num_layers=num_encoder_layers,
            dropout=dropout if num_encoder_layers > 1 else 0.0,
            batch_first=True,
        )

        # Decoder LSTM
        self.decoder = torch.nn.LSTM(
            input_size=num_future_features,
            hidden_size=decoder_hidden_size,
            num_layers=num_decoder_layers,
            dropout=dropout if num_decoder_layers > 1 else 0.0,
            batch_first=True,
        )

        # State projection layer (if encoder and decoder have different hidden sizes)
        self.state_projection: torch.nn.Module | None = None
        if encoder_hidden_size != decoder_hidden_size:
            self.state_projection = torch.nn.ModuleDict({
                "h_proj": torch.nn.Linear(encoder_hidden_size, decoder_hidden_size),
                "c_proj": torch.nn.Linear(encoder_hidden_size, decoder_hidden_size),
            })

        # MLP head for output projection
        if mlp_hidden_dim is not None:
            self.mlp_head = MLP(
                input_dim=decoder_hidden_size,
                hidden_dim=mlp_hidden_dim,
                output_dim=output_dim,
                dropout=mlp_dropout,
                activation=mlp_activation,
            )
        else:
            self.mlp_head = torch.nn.Linear(decoder_hidden_size, output_dim)

    def _project_encoder_states(self, encoder_states: HIDDEN_STATE) -> HIDDEN_STATE:
        """Project encoder states to decoder dimensions if needed."""
        if self.state_projection is None:
            return encoder_states

        h_enc, c_enc = encoder_states
        h_proj = self.state_projection["h_proj"](h_enc)  # type: ignore[index]
        c_proj = self.state_projection["c_proj"](c_enc)  # type: ignore[index]
        return h_proj, c_proj

    @typing.overload
    def forward(
        self,
        past_sequence: torch.Tensor,
        future_sequence: torch.Tensor,
        *,
        return_encoder_states: typing.Literal[False] = False,
    ) -> torch.Tensor: ...

    @typing.overload
    def forward(
        self,
        past_sequence: torch.Tensor,
        future_sequence: torch.Tensor,
        *,
        return_encoder_states: typing.Literal[True],
    ) -> tuple[torch.Tensor, HIDDEN_STATE]: ...

    def forward(
        self,
        past_sequence: torch.Tensor,
        future_sequence: torch.Tensor,
        *,
        return_encoder_states: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, HIDDEN_STATE]:
        """
        Forward pass through the encoder-decoder LSTM model.

        Parameters
        ----------
        past_sequence : torch.Tensor
            Past sequence tensor of shape (batch_size, past_seq_len, num_past_features).
        future_sequence : torch.Tensor
            Future sequence tensor of shape (batch_size, future_seq_len, num_future_features).
        return_encoder_states : bool, default=False
            Whether to return the encoder's final hidden states.

        Returns
        -------
        torch.Tensor | tuple[torch.Tensor, HIDDEN_STATE]
            If return_encoder_states=False:
                Output tensor of shape (batch_size, future_seq_len, output_dim)
            If return_encoder_states=True:
                Tuple of (output, encoder_states) where encoder_states is (h_n, c_n)
        """
        # Encode past sequence
        _, encoder_states = self.encoder(past_sequence)

        # Project encoder states to decoder dimensions if needed
        decoder_init_states = self._project_encoder_states(encoder_states)

        # Decode future sequence using encoder context
        decoder_output, _ = self.decoder(future_sequence, decoder_init_states)

        # Apply MLP head
        output = self.mlp_head(decoder_output)

        if return_encoder_states:
            return output, encoder_states
        return output
