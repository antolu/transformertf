from __future__ import annotations

import numpy as np
import torch

__all__ = ["SimplePositionalEncoding"]


class SimplePositionalEncoding(torch.nn.Module):
    def __init__(self, dim_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()

        self.dropout = torch.nn.Dropout(p=dropout)
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
        return self.dropout(x)
