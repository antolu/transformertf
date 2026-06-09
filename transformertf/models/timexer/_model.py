from __future__ import annotations

import torch

from ._modules import (
    EncoderLayer,
    EnEmbedding,
    ExEmbedding,
    FlattenHead,
)

__all__ = ["TimeXerModel"]


class TimeXerModel(torch.nn.Module):
    def __init__(
        self,
        ctxt_seq_len: int,
        tgt_seq_len: int,
        num_past_covariates: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 3,
        patch_len: int = 16,
        d_ff: int = 512,
        dropout: float = 0.1,
        *,
        use_norm: bool = True,
    ) -> None:
        super().__init__()
        if ctxt_seq_len % patch_len != 0:
            msg = f"ctxt_seq_len ({ctxt_seq_len}) must be divisible by patch_len ({patch_len})"
            raise ValueError(msg)
        self.use_norm = use_norm
        self.tgt_seq_len = tgt_seq_len
        patch_num = ctxt_seq_len // patch_len

        self.en_embedding = EnEmbedding(d_model, patch_len, dropout)
        self.ex_embedding = ExEmbedding(ctxt_seq_len + tgt_seq_len, d_model, dropout)
        self.encoder = torch.nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.norm = torch.nn.LayerNorm(d_model)
        self.head = FlattenHead(
            nf=d_model * (patch_num + 1),
            pred_len=tgt_seq_len,
        )

    def forward(self, x_enc: torch.Tensor, x_ex: torch.Tensor) -> torch.Tensor:
        if self.use_norm:
            mean = x_enc.mean(dim=1, keepdim=True)
            stdev = x_enc.std(dim=1, keepdim=True) + 1e-5
            x_enc = (x_enc - mean) / stdev

        en_out = self.en_embedding(x_enc)
        ex_out = self.ex_embedding(x_ex)

        for layer in self.encoder:
            en_out = layer(en_out, ex_out)

        en_out = self.norm(en_out)

        en_out = en_out.unsqueeze(1).permute(0, 1, 3, 2)  # (B, 1, d_model, patch_num+1)
        out = self.head(en_out)  # (B, 1, tgt_seq_len)
        out = out.permute(0, 2, 1)  # (B, tgt_seq_len, 1)

        if self.use_norm:
            out = out * stdev + mean

        return out
