from __future__ import annotations

import torch

B, C, T, P, D = 2, 3, 64, 16, 32  # batch, channels, time, patch_len, d_model
PATCH_NUM = T // P  # 4


def test_patch_embedding_output_shape():
    from transformertf.models.patchtst._modules import PatchEmbedding

    emb = PatchEmbedding(patch_len=P, d_model=D, patch_num=PATCH_NUM, dropout=0.0)
    x = torch.randn(B, T, C)
    out = emb(x)
    assert out.shape == (B, C, PATCH_NUM, D)


def test_patch_embedding_finite():
    from transformertf.models.patchtst._modules import PatchEmbedding

    emb = PatchEmbedding(patch_len=P, d_model=D, patch_num=PATCH_NUM, dropout=0.0)
    x = torch.randn(B, T, C)
    out = emb(x)
    assert torch.isfinite(out).all()


def test_temporal_attention_block_shape():
    from transformertf.models.patchtst._modules import TemporalAttentionBlock

    block = TemporalAttentionBlock(d_model=D, num_heads=4, dropout=0.0)
    x = torch.randn(B, C, PATCH_NUM, D)
    out = block(x)
    assert out.shape == (B, C, PATCH_NUM, D)


def test_spatial_attention_block_shape():
    from transformertf.models.patchtst._modules import SpatialAttentionBlock

    block = SpatialAttentionBlock(d_model=D, num_heads=4, dropout=0.0)
    x = torch.randn(B, C, PATCH_NUM, D)
    out = block(x)
    assert out.shape == (B, C, PATCH_NUM, D)


def test_st_encoder_block_shape():
    from transformertf.models.patchtst._modules import STEncoderBlock

    block = STEncoderBlock(d_model=D, num_heads=4, d_ff=64, dropout=0.0)
    x = torch.randn(B, C, PATCH_NUM, D)
    out = block(x)
    assert out.shape == (B, C, PATCH_NUM, D)


def test_bahdanau_attention_shape():
    from transformertf.models.patchtst._modules import BahdanauAttention

    S = C * PATCH_NUM  # memory sequence length
    attn = BahdanauAttention(query_dim=D, memory_dim=D)
    query = torch.randn(B, D)
    memory = torch.randn(B, S, D)
    context = attn(query, memory)
    assert context.shape == (B, D)


def test_lstm_decoder_with_attention_shape():
    from transformertf.models.patchtst._modules import LSTMDecoderWithAttention

    S = C * PATCH_NUM
    TGT = 20
    H = 48
    dec = LSTMDecoderWithAttention(
        d_model=D, lstm_hidden=H, lstm_num_layers=2, dropout=0.0
    )
    decoder_embed = torch.randn(B, TGT, D)
    memory = torch.randn(B, S, D)
    h0 = torch.zeros(2, B, H)
    c0 = torch.zeros(2, B, H)
    out = dec(decoder_embed, memory, h0, c0)
    assert out.shape == (B, TGT, 1)
