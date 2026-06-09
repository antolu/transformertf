from __future__ import annotations

import torch

__all__ = [
    "BahdanauAttention",
    "LSTMDecoderWithAttention",
    "PatchEmbedding",
    "STEncoderBlock",
    "SpatialAttentionBlock",
    "TemporalAttentionBlock",
]


class PatchEmbedding(torch.nn.Module):
    def __init__(
        self,
        patch_len: int,
        d_model: int,
        patch_num: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.patch_len = patch_len
        self.projection = torch.nn.Linear(patch_len, d_model, bias=False)
        self.pos_embedding = torch.nn.Parameter(torch.zeros(1, 1, patch_num, d_model))
        torch.nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)  # (B, C, T)
        x = x.unfold(-1, self.patch_len, self.patch_len)  # (B, C, patch_num, patch_len)
        x = self.projection(x)  # (B, C, patch_num, d_model)
        x = x + self.pos_embedding
        return self.dropout(x)


class TemporalAttentionBlock(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.norm = torch.nn.LayerNorm(d_model)
        self.attn = torch.nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, P, D = x.shape
        x_flat = x.reshape(B * C, P, D)
        x_norm = self.norm(x_flat)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x_flat = x_flat + self.dropout(attn_out)
        return x_flat.reshape(B, C, P, D)


class SpatialAttentionBlock(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.norm = torch.nn.LayerNorm(d_model)
        self.attn = torch.nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, P, D = x.shape
        x_flat = x.permute(0, 2, 1, 3).reshape(B * P, C, D)
        x_norm = self.norm(x_flat)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x_flat = x_flat + self.dropout(attn_out)
        return x_flat.reshape(B, P, C, D).permute(0, 2, 1, 3)


class STEncoderBlock(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.temporal = TemporalAttentionBlock(d_model, num_heads, dropout)
        self.spatial = SpatialAttentionBlock(d_model, num_heads, dropout)
        self.norm_ffn = torch.nn.LayerNorm(d_model)
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_ff),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(d_ff, d_model),
            torch.nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.temporal(x)
        x = self.spatial(x)
        return x + self.ffn(self.norm_ffn(x))


class BahdanauAttention(torch.nn.Module):
    def __init__(self, query_dim: int, memory_dim: int) -> None:
        super().__init__()
        self.W_q = torch.nn.Linear(query_dim, memory_dim, bias=False)
        self.W_k = torch.nn.Linear(memory_dim, memory_dim, bias=False)
        self.v = torch.nn.Linear(memory_dim, 1, bias=False)

    def forward(self, query: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        q = self.W_q(query).unsqueeze(1)  # (B, 1, memory_dim)
        k = self.W_k(memory)  # (B, S, memory_dim)
        scores = self.v(torch.tanh(q + k)).squeeze(-1)  # (B, S)
        weights = torch.softmax(scores, dim=-1)  # (B, S)
        return (weights.unsqueeze(-1) * memory).sum(dim=1)  # (B, memory_dim)


class LSTMDecoderWithAttention(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        lstm_hidden: int,
        lstm_num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.lstm = torch.nn.LSTM(
            input_size=d_model * 2,
            hidden_size=lstm_hidden,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0.0,
        )
        self.attention = BahdanauAttention(query_dim=lstm_hidden, memory_dim=d_model)
        self.norm = torch.nn.LayerNorm(lstm_hidden)
        self.output_head = torch.nn.Linear(lstm_hidden, 1)

    def forward(
        self,
        decoder_embed: torch.Tensor,
        memory: torch.Tensor,
        h0: torch.Tensor,
        c0: torch.Tensor,
    ) -> torch.Tensor:
        B, T, _ = decoder_embed.shape
        D_mem = memory.shape[-1]

        hx: tuple[torch.Tensor, torch.Tensor] = (h0, c0)
        context = torch.zeros(
            B, D_mem, device=decoder_embed.device, dtype=decoder_embed.dtype
        )
        outputs = []

        for t in range(T):
            inp = torch.cat([decoder_embed[:, t, :], context], dim=-1).unsqueeze(1)
            _out, hx = self.lstm(inp, hx)
            h_top = hx[0][-1]  # (B, lstm_hidden)
            context = self.attention(h_top, memory)
            outputs.append(self.output_head(self.norm(h_top)))

        return torch.stack(outputs, dim=1)  # (B, T, 1)
