from __future__ import annotations

import math

import torch
import torch.nn as nn


class _PositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pe[:, : x.size(1)]  # type: ignore[index]


class _EnEmbedding(nn.Module):
    def __init__(self, d_model: int, patch_len: int, dropout: float) -> None:
        super().__init__()
        self.patch_len = patch_len
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.position_embedding = _PositionalEmbedding(d_model)
        self.glb_token = nn.Parameter(torch.randn(1, 1, 1, d_model))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        x = x.permute(0, 2, 1)  # (B, 1, ctxt_seq_len)
        x = x.unfold(-1, self.patch_len, self.patch_len)  # (B, 1, patch_num, patch_len)
        n_vars = x.size(1)
        patch_num = x.size(2)
        x = x.reshape(B * n_vars, patch_num, self.patch_len)
        x = self.value_embedding(x) + self.position_embedding(x)
        x = x.reshape(B, n_vars, patch_num, -1)
        glb = self.glb_token.expand(B, n_vars, 1, -1)
        x = torch.cat([x, glb], dim=2)  # (B, 1, patch_num+1, d_model)
        x = x.reshape(B, patch_num + 1, -1)  # (B, patch_num+1, d_model)
        return self.dropout(x)


class _ExEmbedding(nn.Module):
    def __init__(self, total_seq_len: int, d_model: int, dropout: float) -> None:
        super().__init__()
        self.value_embedding = nn.Linear(total_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)  # (B, num_covariates, total_seq_len)
        x = self.value_embedding(x)  # (B, num_covariates, d_model)
        return self.dropout(x)


class _EncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, cross: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))

        glb = x[:, -1:, :]
        cross_out, _ = self.cross_attn(glb, cross, cross)
        glb = self.norm2(glb + self.dropout(cross_out))
        x = torch.cat([x[:, :-1, :], glb], dim=1)

        residual = x
        x = x.transpose(1, 2)  # (B, d_model, seq)
        x = torch.relu(self.conv1(x))
        x = self.conv2(x)
        x = x.transpose(1, 2)  # (B, seq, d_model)
        return self.norm3(residual + self.dropout(x))


class _FlattenHead(nn.Module):
    def __init__(self, nf: int, pred_len: int, dropout: float) -> None:
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, pred_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)  # (B, 1, nf)
        x = self.linear(x)  # (B, 1, pred_len)
        return self.dropout(x)


class TimeXerModel(nn.Module):
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

        self.en_embedding = _EnEmbedding(d_model, patch_len, dropout)
        self.ex_embedding = _ExEmbedding(ctxt_seq_len + tgt_seq_len, d_model, dropout)
        self.encoder = nn.ModuleList([
            _EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = _FlattenHead(
            nf=d_model * (patch_num + 1),
            pred_len=tgt_seq_len,
            dropout=dropout,
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
