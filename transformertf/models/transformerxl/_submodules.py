from __future__ import annotations

import typing

import torch
import torch.nn.functional as F
from torch import nn


class PositionalEmbedding(nn.Module):
    inv_freq: torch.Tensor

    def __init__(self, dim_embedding: int):
        super().__init__()

        self.d_emb = dim_embedding

        inv_freq = 1 / (
            10000 ** (torch.arange(0.0, dim_embedding, 2.0) / dim_embedding)
        )
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, pos_seq: torch.Tensor, bsz: int | None = None) -> torch.Tensor:
        sinusoid_inp = torch.outer(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        if bsz is not None:
            return pos_emb[:, None, :].expand(-1, bsz, -1)
        return pos_emb[:, None, :]


class PositionwiseFF(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_inner: int,
        dropout: float,
        *,
        pre_lnorm: bool = False,
    ):
        super().__init__()

        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout

        self.CoreNet = nn.Sequential(
            nn.Linear(d_model, d_inner),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_inner, d_model),
            nn.Dropout(dropout),
        )

        self.layer_norm = nn.LayerNorm(d_model)

        self.pre_lnorm = pre_lnorm

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        if self.pre_lnorm:
            # layer normalization + positionwise feed-forward
            core_out = self.CoreNet(self.layer_norm(inp))

            # residual connection
            output = core_out + inp
        else:
            # positionwise feed-forward
            core_out = self.CoreNet(inp)

            # residual connection + layer normalization
            output = self.layer_norm(inp + core_out)

        return output


class MultiHeadAttn(nn.Module):
    def __init__(
        self,
        n_head: int,
        d_model: int,
        d_head: int,
        dropout: float,
        dropatt: float = 0,
        *,
        pre_lnorm: bool = False,
    ):
        super().__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        self.q_net = nn.Linear(d_model, n_head * d_head, bias=False)
        self.kv_net = nn.Linear(d_model, 2 * n_head * d_head, bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head**0.5)

        self.pre_lnorm = pre_lnorm

    def forward(
        self,
        h: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        mems: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # multihead attention
        # [hlen x bsz x n_head x d_head]

        c = torch.cat([mems, h], 0) if mems is not None else h

        if self.pre_lnorm:
            # layer normalization
            c = self.layer_norm(c)

        head_q = self.q_net(h)
        head_k, head_v = torch.chunk(self.kv_net(c), 2, -1)

        head_q = head_q.view(h.size(0), h.size(1), self.n_head, self.d_head)
        head_k = head_k.view(c.size(0), c.size(1), self.n_head, self.d_head)
        head_v = head_v.view(c.size(0), c.size(1), self.n_head, self.d_head)

        # [qlen x klen x bsz x n_head]
        attn_score = torch.einsum("ibnd,jbnd->ijbn", (head_q, head_k))
        attn_score.mul_(self.scale)
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score.masked_fill_(attn_mask[None, :, :, None], -float("inf"))
            elif attn_mask.dim() == 3:
                attn_score.masked_fill_(attn_mask[:, :, :, None], -float("inf"))

        # [qlen x klen x bsz x n_head]
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)

        # [qlen x klen x bsz x n_head] + [klen x bsz x n_head x d_head] -> [qlen x bsz x n_head x d_head]  # E501
        attn_vec = torch.einsum("ijbn,jbnd->ibnd", (attn_prob, head_v))
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head
        )

        # linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        return h + attn_out if self.pre_lnorm else self.layer_norm(h + attn_out)


class RelMultiHeadAttn(nn.Module):
    def __init__(
        self,
        n_head: int,
        d_model: int,
        d_head: int,
        dropout: float,
        dropatt: float = 0,
        tgt_len: int | None = None,
        ext_len: int | None = None,
        mem_len: int | None = None,
        *,
        pre_lnorm: bool = False,
    ):
        super().__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        self.qkv_net = nn.Linear(d_model, 3 * n_head * d_head, bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head**0.5)

        self.pre_lnorm = pre_lnorm

    def _parallelogram_mask(
        self, h: int, w: int, *, left: bool = False
    ) -> torch.Tensor:
        mask = torch.ones((h, w)).byte()
        m = min(h, w)
        mask[:m, :m] = torch.triu(mask[:m, :m])
        mask[-m:, -m:] = torch.tril(mask[-m:, -m:])

        if left:
            return mask
        return mask.flip(0)

    def _shift(
        self,
        x: torch.Tensor,
        qlen: int,
        klen: int,
        mask: torch.Tensor,
        *,
        left: bool = False,
    ) -> torch.Tensor:
        if qlen > 1:
            zero_pad = torch.zeros(
                (x.size(0), qlen - 1, x.size(2), x.size(3)),
                device=x.device,
                dtype=x.dtype,
            )
        else:
            zero_pad = torch.zeros(0, device=x.device, dtype=x.dtype)

        if left:
            mask = mask.flip(1)
            x_padded = torch.cat([zero_pad, x], dim=1).expand(qlen, -1, -1, -1)
        else:
            x_padded = torch.cat([x, zero_pad], dim=1).expand(qlen, -1, -1, -1)

        return x_padded.masked_select(mask[:, :, None, None]).view(
            qlen, klen, x.size(2), x.size(3)
        )

    def _rel_shift(self, x: torch.Tensor, *, zero_triu: bool = False) -> torch.Tensor:
        zero_pad = torch.zeros(
            (x.size(0), 1, *x.size()[2:]), device=x.device, dtype=x.dtype
        )
        x_padded = torch.cat([zero_pad, x], dim=1)

        x_padded = x_padded.view(x.size(1) + 1, x.size(0), *x.size()[2:])

        x = x_padded[1:].view_as(x)

        if zero_triu:
            ones = torch.ones((x.size(0), x.size(1)))
            x *= torch.tril(ones, x.size(1) - x.size(0))[:, :, None, None]

        return x

    def forward(
        self,
        w: torch.Tensor,
        r: torch.Tensor,
        w_bias: torch.Tensor,
        r_bias: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        mems: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError


class RelPartialLearnableMultiHeadAttn(RelMultiHeadAttn):
    def __init__(
        self,
        n_head: int,
        d_model: int,
        d_head: int,
        dropout: float,
        dropatt: float = 0,
        *,
        pre_lnorm: bool = False,
    ):
        super().__init__(
            n_head=n_head,
            d_model=d_model,
            d_head=d_head,
            dropout=dropout,
            dropatt=dropatt,
            pre_lnorm=pre_lnorm,
        )

        self.r_net = nn.Linear(self.d_model, self.n_head * self.d_head, bias=False)

    def forward(
        self,
        w: torch.Tensor,
        r: torch.Tensor,
        r_w_bias: torch.Tensor,
        r_r_bias: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        mems: torch.Tensor | None = None,
    ) -> torch.Tensor:
        qlen, rlen, bsz = w.size(0), r.size(0), w.size(1)

        if mems is not None:
            cat = torch.cat([mems, w], 0)
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(cat))
            else:
                w_heads = self.qkv_net(cat)
            r_head_k = self.r_net(r)

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)
            w_head_q = w_head_q[-qlen:]
        else:
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(w))
            else:
                w_heads = self.qkv_net(w)
            r_head_k = self.r_net(r)

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

        klen = w_head_k.size(0)

        w_head_q = w_head_q.view(
            qlen, bsz, self.n_head, self.d_head
        )  # qlen x bsz x n_head x d_head
        w_head_k = w_head_k.view(
            klen, bsz, self.n_head, self.d_head
        )  # qlen x bsz x n_head x d_head
        w_head_v = w_head_v.view(
            klen, bsz, self.n_head, self.d_head
        )  # qlen x bsz x n_head x d_head

        r_head_k = r_head_k.view(
            rlen, self.n_head, self.d_head
        )  # qlen x n_head x d_head

        # compute attention score
        rw_head_q = w_head_q + r_w_bias  # qlen x bsz x n_head x d_head
        AC = torch.einsum(
            "ibnd,jbnd->ijbn", (rw_head_q, w_head_k)
        )  # qlen x klen x bsz x n_head

        rr_head_q = w_head_q + r_r_bias
        BD = torch.einsum(
            "ibnd,jnd->ijbn", (rr_head_q, r_head_k)
        )  # qlen x klen x bsz x n_head
        BD = self._rel_shift(BD)

        # [qlen x klen x bsz x n_head]
        attn_score = AC + BD
        attn_score.mul_(self.scale)

        # compute attention probability
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score = (
                    attn_score.float()
                    .masked_fill(attn_mask[None, :, :, None], -float("inf"))
                    .type_as(attn_score)
                )
            elif attn_mask.dim() == 3:
                attn_score = (
                    attn_score.float()
                    .masked_fill(attn_mask[:, :, :, None], -float("inf"))
                    .type_as(attn_score)
                )

        # [qlen x klen x bsz x n_head]
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)

        # compute attention vector
        attn_vec = torch.einsum("ijbn,jbnd->ibnd", (attn_prob, w_head_v))

        # [qlen x bsz x n_head x d_head]
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head
        )

        # linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)
        return w + attn_out if self.pre_lnorm else self.layer_norm(w + attn_out)


class RelLearnableMultiHeadAttn(RelMultiHeadAttn):
    def forward(
        self,
        w: torch.Tensor,
        r_emb: torch.Tensor,
        r_w_bias: torch.Tensor,
        r_bias: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        mems: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # r_emb: [klen, n_head, d_head], used for term B
        # r_w_bias: [n_head, d_head], used for term C
        # r_bias: [klen, n_head], used for term D

        qlen, bsz = w.size(0), w.size(1)

        if mems is not None:
            cat = torch.cat([mems, w], 0)
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(cat))
            else:
                w_heads = self.qkv_net(cat)
            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

            w_head_q = w_head_q[-qlen:]
        else:
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(w))
            else:
                w_heads = self.qkv_net(w)
            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

        klen = w_head_k.size(0)

        w_head_q = w_head_q.view(qlen, bsz, self.n_head, self.d_head)
        w_head_k = w_head_k.view(klen, bsz, self.n_head, self.d_head)
        w_head_v = w_head_v.view(klen, bsz, self.n_head, self.d_head)

        if klen > r_emb.size(0):
            r_emb_pad = r_emb[0:1].expand(klen - r_emb.size(0), -1, -1)
            r_emb = torch.cat([r_emb_pad, r_emb], 0)
            r_bias_pad = r_bias[0:1].expand(klen - r_bias.size(0), -1)
            r_bias = torch.cat([r_bias_pad, r_bias], 0)
        else:
            r_emb = r_emb[-klen:]
            r_bias = r_bias[-klen:]

        # compute attention score
        rw_head_q = w_head_q + r_w_bias[None]  # qlen x bsz x n_head x d_head

        AC = torch.einsum(
            "ibnd,jbnd->ijbn", (rw_head_q, w_head_k)
        )  # qlen x klen x bsz x n_head
        B_ = torch.einsum(
            "ibnd,jnd->ijbn", (w_head_q, r_emb)
        )  # qlen x klen x bsz x n_head
        D_ = r_bias[None, :, None]  # 1    x klen x 1   x n_head
        BD = self._rel_shift(B_ + D_)

        # [qlen x klen x bsz x n_head]
        attn_score = AC + BD
        attn_score.mul_(self.scale)

        # compute attention probability
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score.masked_fill_(attn_mask[None, :, :, None], -float("inf"))
            elif attn_mask.dim() == 3:
                attn_score.masked_fill_(attn_mask[:, :, :, None], -float("inf"))

        # [qlen x klen x bsz x n_head]
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)

        # compute attention vector
        attn_vec = torch.einsum("ijbn,jbnd->ibnd", (attn_prob, w_head_v))

        # [qlen x bsz x n_head x d_head]
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head
        )

        # linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        return w + attn_out if self.pre_lnorm else self.layer_norm(w + attn_out)


class DecoderLayer(nn.Module):
    def __init__(
        self,
        n_head: int,
        d_model: int,
        d_head: int,
        d_inner: int,
        dropout: float,
        dropatt: float,
        *,
        prel_norm: bool = False,
    ):
        super().__init__()

        self.dec_attn = MultiHeadAttn(
            n_head, d_model, d_head, dropout, dropatt, pre_lnorm=prel_norm
        )
        self.pos_ff = PositionwiseFF(d_model, d_inner, dropout, pre_lnorm=prel_norm)

    def forward(
        self,
        dec_inp: torch.Tensor,
        dec_attn_mask: torch.Tensor | None = None,
        mems: torch.Tensor | None = None,
    ) -> torch.Tensor:
        output = self.dec_attn(dec_inp, attn_mask=dec_attn_mask, mems=mems)
        return self.pos_ff(output)


class RelLearnableDecoderLayer(nn.Module):
    def __init__(
        self,
        n_head: int,
        d_model: int,
        d_head: int,
        d_inner: int,
        dropout: float,
        dropatt: float,
        *,
        pre_lnorm: bool = False,
    ):
        super().__init__()

        self.dec_attn = RelLearnableMultiHeadAttn(
            n_head, d_model, d_head, dropout, dropatt, pre_lnorm=pre_lnorm
        )
        self.pos_ff = PositionwiseFF(d_model, d_inner, dropout, pre_lnorm=pre_lnorm)

    def forward(
        self,
        dec_inp: torch.Tensor,
        r_emb: torch.Tensor,
        r_w_bias: torch.Tensor,
        r_bias: torch.Tensor,
        dec_attn_mask: torch.Tensor | None = None,
        mems: torch.Tensor | None = None,
    ) -> torch.Tensor:
        output = self.dec_attn(
            dec_inp,
            r_emb,
            r_w_bias,
            r_bias,
            attn_mask=dec_attn_mask,
            mems=mems,
        )
        return self.pos_ff(output)


class RelPartialLearnableDecoderLayer(nn.Module):
    def __init__(
        self,
        n_head: int,
        d_model: int,
        d_head: int,
        d_inner: int,
        dropout: float,
        dropatt: float,
        *,
        prel_norm: bool = False,
    ):
        super().__init__()

        self.dec_attn = RelPartialLearnableMultiHeadAttn(
            n_head, d_model, d_head, dropout, dropatt, pre_lnorm=prel_norm
        )
        self.pos_ff = PositionwiseFF(d_model, d_inner, dropout, pre_lnorm=prel_norm)

    def forward(
        self,
        dec_inp: torch.Tensor,
        r: torch.Tensor,
        r_w_bias: torch.Tensor,
        r_r_bias: torch.Tensor,
        dec_attn_mask: torch.Tensor | None = None,
        mems: torch.Tensor | None = None,
    ) -> torch.Tensor:
        output = self.dec_attn(
            dec_inp, r, r_w_bias, r_r_bias, attn_mask=dec_attn_mask, mems=mems
        )
        return self.pos_ff(output)


class AdaptiveEmbedding(nn.Module):
    def __init__(
        self,
        n_token: int,
        d_embed: int,
        d_proj: int,
        cutoffs: typing.Sequence[int],
        div_val: int = 1,
        *,
        sample_softmax: bool = False,
    ):
        super().__init__()

        self.n_token = n_token
        self.d_embed = d_embed

        self.cutoffs = [*list(cutoffs), n_token]
        self.div_val = div_val
        self.d_proj = d_proj

        self.emb_scale = d_proj**0.5

        self.cutoff_ends = [0, *self.cutoffs]

        self.emb_layers = nn.ModuleList()
        self.emb_projs = nn.ParameterList()
        if div_val == 1:
            self.emb_layers.append(
                nn.Embedding(n_token, d_embed, sparse=sample_softmax > 0)
            )
            if d_proj != d_embed:
                self.emb_projs.append(nn.Parameter(torch.Tensor(d_proj, d_embed)))
        else:
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]
                d_emb_i = d_embed // (div_val**i)
                self.emb_layers.append(nn.Embedding(r_idx - l_idx, d_emb_i))
                self.emb_projs.append(nn.Parameter(torch.Tensor(d_proj, d_emb_i)))

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        if self.div_val == 1:
            embed = self.emb_layers[0](inp)
            if self.d_proj != self.d_embed:
                embed = F.linear(embed, self.emb_projs[0])
        else:
            param = next(self.parameters())
            inp_flat = inp.view(-1)
            emb_flat = torch.zeros(
                [inp_flat.size(0), self.d_proj],
                dtype=param.dtype,
                device=param.device,
            )
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]

                mask_i = (inp_flat >= l_idx) & (inp_flat < r_idx)
                indices_i = mask_i.nonzero().squeeze()

                if indices_i.numel() == 0:
                    continue

                inp_i = inp_flat.index_select(0, indices_i) - l_idx
                emb_i = self.emb_layers[i](inp_i)
                emb_i = F.linear(emb_i, self.emb_projs[i])

                emb_flat.index_copy_(0, indices_i, emb_i)

            embed = emb_flat.view(*inp.size(), self.d_proj)

        embed.mul_(self.emb_scale)

        return embed


class ProjectedAdaptiveLogSoftmax(nn.Module):
    def __init__(
        self,
        n_token: int,
        d_embed: int,
        d_proj: int,
        cutoffs: typing.Sequence[int],
        div_val: int = 1,
        *,
        keep_order: bool = False,
    ):
        super().__init__()

        self.n_token = n_token
        self.d_embed = d_embed
        self.d_proj = d_proj

        self.cutoffs = [*list(cutoffs), n_token]
        self.cutoff_ends = [0, *self.cutoffs]
        self.div_val = div_val

        self.shortlist_size = self.cutoffs[0]
        self.n_clusters = len(self.cutoffs) - 1
        self.head_size = self.shortlist_size + self.n_clusters

        if self.n_clusters > 0:
            self.cluster_weight = nn.Parameter(
                torch.zeros(self.n_clusters, self.d_embed)
            )
            self.cluster_bias = nn.Parameter(torch.zeros(self.n_clusters))

        self.out_layers = nn.ModuleList()
        self.out_projs = nn.ParameterList()

        if div_val == 1:
            for _ in range(len(self.cutoffs)):
                if d_proj != d_embed:
                    self.out_projs.append(nn.Parameter(torch.Tensor(d_proj, d_embed)))
                else:
                    self.out_projs.append(None)

            self.out_layers.append(nn.Linear(d_embed, n_token))
        else:
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]
                d_emb_i = d_embed // (div_val**i)

                self.out_projs.append(nn.Parameter(torch.Tensor(d_proj, d_emb_i)))

                self.out_layers.append(nn.Linear(d_emb_i, r_idx - l_idx))

        self.keep_order = keep_order

    def _compute_logit(
        self,
        hidden: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        proj: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if proj is None:
            logit = F.linear(hidden, weight, bias=bias)
        else:
            # if CUDA_MAJOR <= 9 and CUDA_MINOR <= 1:
            proj_hid = F.linear(hidden, proj.t().contiguous())
            logit = F.linear(proj_hid, weight, bias=bias)
            # else:
            #     logit = torch.einsum('bd,de,ev->bv', (hidden, proj, weight.t()))
            #     if bias is not None:
            #         logit = logit + bias

        return logit

    def forward(
        self,
        hidden: torch.Tensor,
        target: torch.Tensor,
        *,
        keep_order: bool = False,
    ) -> torch.Tensor:
        """
        hidden :: [len*bsz x d_proj]
        target :: [len*bsz]
        """

        if hidden.size(0) != target.size(0):
            msg = "Input and target should have the same size in the batch dimension."
            raise RuntimeError(msg)

        if self.n_clusters == 0:
            logit = self._compute_logit(
                hidden,
                self.out_layers[0].weight,
                self.out_layers[0].bias,
                self.out_projs[0],
            )
            nll = (
                -F.log_softmax(logit, dim=-1).gather(1, target.unsqueeze(1)).squeeze(1)
            )
        else:
            # construct weights and biases
            weights, biases = [], []
            for i in range(len(self.cutoffs)):
                if self.div_val == 1:
                    l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]
                    weight_i = self.out_layers[0].weight[l_idx:r_idx]
                    bias_i = self.out_layers[0].bias[l_idx:r_idx]
                else:
                    weight_i = self.out_layers[i].weight
                    bias_i = self.out_layers[i].bias

                if i == 0:
                    weight_i = torch.cat([weight_i, self.cluster_weight], dim=0)
                    bias_i = torch.cat([bias_i, self.cluster_bias], dim=0)

                weights.append(weight_i)
                biases.append(bias_i)

            head_weight, head_bias, head_proj = (
                weights[0],
                biases[0],
                self.out_projs[0],
            )

            head_logit = self._compute_logit(hidden, head_weight, head_bias, head_proj)
            head_logprob = F.log_softmax(head_logit, dim=1)

            nll = torch.zeros_like(target, dtype=hidden.dtype, device=hidden.device)

            offset = 0
            cutoff_values = [0, *self.cutoffs]
            for i in range(len(cutoff_values) - 1):
                l_idx, r_idx = cutoff_values[i], cutoff_values[i + 1]

                mask_i = (target >= l_idx) & (target < r_idx)
                indices_i = mask_i.nonzero().squeeze()

                if indices_i.numel() == 0:
                    continue

                target_i = target.index_select(0, indices_i) - l_idx
                head_logprob_i = head_logprob.index_select(0, indices_i)

                if i == 0:
                    logprob_i = head_logprob_i.gather(1, target_i[:, None]).squeeze(1)
                else:
                    weight_i, bias_i, proj_i = (
                        weights[i],
                        biases[i],
                        self.out_projs[i],
                    )

                    hidden_i = hidden.index_select(0, indices_i)

                    tail_logit_i = self._compute_logit(
                        hidden_i, weight_i, bias_i, proj_i
                    )
                    tail_logprob_i = F.log_softmax(tail_logit_i, dim=1)

                    logprob_i = head_logprob_i[:, -i] + tail_logprob_i.gather(
                        1, target_i[:, None]
                    ).squeeze(1)

                if (hasattr(self, "keep_order") and self.keep_order) or keep_order:
                    nll.index_copy_(0, indices_i, -logprob_i)
                else:
                    nll[offset : offset + logprob_i.size(0)].copy_(-logprob_i)

                offset += logprob_i.size(0)

        return nll


class LogUniformSampler:
    def __init__(self, range_max: int, n_sample: int):
        """
        Reference : https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/ops/candidate_sampling_ops.py  # noqa E501
            `P(class) = (log(class + 2) - log(class + 1)) / log(range_max + 1)`

        expected count can be approximated by 1 - (1 - p)^n
        and we use a numerically stable version -expm1(num_tries * log1p(-p))

        Our implementation fixes num_tries at 2 * n_sample, and the actual
        #samples will vary from run to run
        """
        with torch.no_grad():
            self.range_max = range_max
            log_indices = torch.arange(1.0, range_max + 2.0, 1.0).log_()
            self.dist = (log_indices[1:] - log_indices[:-1]) / log_indices[-1]
            # print('P', self.dist.numpy().tolist()[-30:])

            self.log_q = (
                (-(-self.dist.double().log1p_() * 2 * n_sample).expm1_()).log_().float()
            )

        self.n_sample = n_sample

    def sample(
        self, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
            labels: [b1, b2]
        Return
            true_log_probs: [b1, b2]
            samp_log_probs: [n_sample]
            neg_samples: [n_sample]
        """

        # neg_samples = torch.empty(0).long()
        n_sample = self.n_sample
        n_tries = 2 * n_sample

        with torch.no_grad():
            neg_samples = torch.multinomial(
                self.dist, n_tries, replacement=True
            ).unique()
            device = labels.device
            neg_samples = neg_samples.to(device)
            true_log_probs = self.log_q[labels].to(device)
            samp_log_probs = self.log_q[neg_samples].to(device)
            return true_log_probs, samp_log_probs, neg_samples


def sample_logits(
    embedding: nn.Module,
    bias: torch.Tensor,
    labels: torch.Tensor,
    inputs: torch.Tensor,
    sampler: LogUniformSampler,
) -> torch.Tensor:
    """
        embedding: an nn.Embedding layer
        bias: [n_vocab]
        labels: [b1, b2]
        inputs: [b1, b2, n_emb]
        sampler: you may use a LogUniformSampler
    Return
        logits: [b1, b2, 1 + n_sample]
    """
    true_log_probs, samp_log_probs, neg_samples = sampler.sample(labels)
    n_sample = neg_samples.size(0)
    b1, b2 = labels.size(0), labels.size(1)
    all_ids = torch.cat([labels.view(-1), neg_samples])
    all_w = embedding(all_ids)
    true_w = all_w[:-n_sample].view(b1, b2, -1)
    sample_w = all_w[-n_sample:].view(n_sample, -1)

    all_b = bias[all_ids]
    true_b = all_b[:-n_sample].view(b1, b2)
    sample_b = all_b[-n_sample:]

    hit = (labels[:, :, None] == neg_samples).detach()

    true_logits = (
        torch.einsum("ijk,ijk->ij", [true_w, inputs]) + true_b - true_log_probs
    )
    sample_logits = (
        torch.einsum("lk,ijk->ijl", [sample_w, inputs]) + sample_b - samp_log_probs
    )
    sample_logits.masked_fill_(hit, -1e30)
    return torch.cat([true_logits[:, :, None], sample_logits], -1)
