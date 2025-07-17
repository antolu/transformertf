"""
Attention-enhanced LSTM for sequence-to-sequence time series forecasting.

This module implements an LSTM architecture enhanced with self-attention mechanisms
for improved sequence modeling capabilities. The model combines the strengths of
LSTM networks for temporal processing with multi-head self-attention for capturing
long-range dependencies.

Classes
-------
AttentionLSTMModel : torch.nn.Module
    Core model implementation with attention-enhanced LSTM architecture.
AttentionLSTM : LightningModuleBase
    PyTorch Lightning wrapper for training and inference.

Key Features
------------
- Shared hyperparameters for encoder and decoder (simplified configuration)
- Self-attention mechanism applied to decoder outputs
- Optional gating mechanism for skip connections
- Dynamic sequence length support through attention masking
- Compatible with both point and probabilistic forecasting
- Minimal hyperparameter set for ease of use

Examples
--------
>>> from transformertf.models.attention_lstm import AttentionLSTM
>>> from transformertf.data import EncoderDecoderDataModule
>>> import lightning as L
>>>
>>> # Create model with shared LSTM parameters
>>> model = AttentionLSTM(
...     num_past_features=10,
...     num_future_features=5,
...     hidden_size=128,
...     num_layers=2,
...     n_heads=4,
...     use_gating=True,
...     dropout=0.1
... )
>>>
>>> # Train with data module
>>> datamodule = EncoderDecoderDataModule(...)
>>> trainer = L.Trainer(max_epochs=100)
>>> trainer.fit(model, datamodule)

Notes
-----
The model architecture consists of:
1. Encoder LSTM: Processes past sequences with shared parameters
2. Decoder LSTM: Processes future sequences with shared parameters
3. Self-attention: Captures long-range dependencies in decoder outputs
4. Skip connection: Optional gating or simple residual connection
5. Linear output: Single linear layer for final predictions

The shared parameter design (same hidden_size and num_layers for both encoder
and decoder) reduces configuration complexity while maintaining model effectiveness.
"""

from ._lightning import AttentionLSTM
from ._model import AttentionLSTMModel

__all__ = [
    "AttentionLSTM",
    "AttentionLSTMModel",
]
