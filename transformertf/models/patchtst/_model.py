from __future__ import annotations

import torch

from ._modules import (
    LSTMDecoderWithAttention,
    PatchEmbedding,
    STEncoderBlock,
)

__all__ = ["PatchTSTModel"]


class PatchTSTModel(torch.nn.Module):
    def __init__(
        self,
        num_future_features: int,
        ctxt_seq_len: int,
        patch_len: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        d_ff: int,
        d_lstm: int,
        num_lstm_layers: int,
        dropout: float,
        use_norm: bool,
    ) -> None:
        super().__init__()
        if ctxt_seq_len % patch_len != 0:
            msg = f"ctxt_seq_len ({ctxt_seq_len}) must be divisible by patch_len ({patch_len})"
            raise ValueError(msg)

        self.use_norm = use_norm
        patch_num = ctxt_seq_len // patch_len

        self.patch_embedding = PatchEmbedding(
            patch_len=patch_len,
            d_model=d_model,
            patch_num=patch_num,
            dropout=dropout,
        )
        self.encoder_blocks = torch.nn.ModuleList([
            STEncoderBlock(
                d_model=d_model, num_heads=num_heads, d_ff=d_ff, dropout=dropout
            )
            for _ in range(num_layers)
        ])

        self.h0_proj = torch.nn.Linear(d_model, d_lstm * num_lstm_layers)
        self.c0_proj = torch.nn.Linear(d_model, d_lstm * num_lstm_layers)

        self.decoder_proj = torch.nn.Linear(num_future_features, d_model)
        self.decoder = LSTMDecoderWithAttention(
            d_model=d_model,
            d_lstm=d_lstm,
            num_lstm_layers=num_lstm_layers,
            dropout=dropout,
        )
        self._num_lstm_layers = num_lstm_layers
        self._d_lstm = d_lstm

    def forward(
        self,
        encoder_input: torch.Tensor,
        decoder_input: torch.Tensor,
    ) -> torch.Tensor:
        B = encoder_input.shape[0]

        if self.use_norm:
            target = encoder_input[:, :, -1:]
            mean = target.mean(dim=1, keepdim=True)
            std = target.std(dim=1, keepdim=True) + 1e-5
            normalized_target = (target - mean) / std
            encoder_input = torch.cat(
                [encoder_input[:, :, :-1], normalized_target], dim=-1
            )

        x = self.patch_embedding(encoder_input)  # (B, C_enc, patch_num, d_model)
        for block in self.encoder_blocks:
            x = block(x)

        _, C, P, D = x.shape
        patch_memory = x.reshape(B, C * P, D)  # (B, C_enc*patch_num, d_model)

        pooled = patch_memory.mean(dim=1)  # (B, d_model)
        h0 = (
            self.h0_proj(pooled)
            .reshape(B, self._num_lstm_layers, self._d_lstm)
            .permute(1, 0, 2)
            .contiguous()
        )
        c0 = (
            self.c0_proj(pooled)
            .reshape(B, self._num_lstm_layers, self._d_lstm)
            .permute(1, 0, 2)
            .contiguous()
        )

        decoder_embed = self.decoder_proj(decoder_input)  # (B, tgt_seq_len, d_model)
        out = self.decoder(decoder_embed, patch_memory, h0, c0)  # (B, tgt_seq_len, 1)

        if self.use_norm:
            out = out * std + mean

        return out
