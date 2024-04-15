from __future__ import annotations

import logging
import typing

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._submodules import (
    AdaptiveEmbedding,
    DecoderLayer,
    LogUniformSampler,
    PositionalEmbedding,
    ProjectedAdaptiveLogSoftmax,
    RelLearnableDecoderLayer,
    RelPartialLearnableDecoderLayer,
    sample_logits,
)

log = logging.getLogger(__name__)


class TransformerXL(nn.Module):
    def __init__(
        self,
        n_token: int,
        n_layer: int,
        n_head: int,
        d_model: int,
        d_head: int,
        d_inner: int,
        dropout: float,
        dropatt: float,
        tie_weight: bool = True,
        d_embed: int | None = None,
        div_val: int = 1,
        tie_projs: list[bool] | None = None,
        pre_lnorm: bool = False,
        tgt_len: int | None = None,
        ext_len: int | None = None,
        mem_len: int | None = None,
        cutoffs: list[int] | None = None,
        adapt_inp: bool = False,
        same_length: bool = False,
        attn_type: typing.Literal[0, 1, 2, 3] = 0,
        clamp_len: int = -1,
        sample_softmax: int = -1,
    ):
        super().__init__()
        tie_projs = tie_projs or [False]
        cutoffs = cutoffs or []

        self.n_token = n_token

        d_embed = d_model if d_embed is None else d_embed
        self.d_embed = d_embed
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head

        self.word_emb = AdaptiveEmbedding(
            n_token, d_embed, d_model, cutoffs, div_val=div_val
        )

        self.drop = nn.Dropout(dropout)

        self.n_layer = n_layer

        self.tgt_len = tgt_len
        self.mem_len = mem_len
        self.ext_len = ext_len
        self.max_klen = sum([k or 0 for k in (tgt_len, ext_len, mem_len)])

        self.attn_type = attn_type

        layer_kwargs = {
            "n_head": n_head,
            "d_model": d_model,
            "d_head": d_head,
            "d_inner": d_inner,
            "dropout": dropout,
            "dropatt": dropatt,
            "pre_lnorm": pre_lnorm,
        }
        layer_cls: typing.Type[nn.Module]
        if attn_type == 0:  # the default attention
            layer_cls = RelPartialLearnableDecoderLayer
            layer_kwargs.update(
                {"tgt_len": tgt_len, "ext_len": ext_len, "mem_len": mem_len}  # type: ignore[dict-item]  # noqa: E501
            )
        elif attn_type == 1:  # learnable embeddings
            layer_cls = RelLearnableDecoderLayer
            layer_kwargs.update(
                {"tgt_len": tgt_len, "ext_len": ext_len, "mem_len": mem_len}  # type: ignore[dict-item]  # noqa: E501
            )
        elif attn_type in [2, 3]:  # absolute embeddings
            layer_cls = DecoderLayer
        else:
            raise ValueError("Unknown attention type {}".format(attn_type))

        self.layers = nn.ModuleList(
            [layer_cls(**layer_kwargs) for _ in range(n_layer)]
        )

        self.sample_softmax = sample_softmax
        # use sampled softmax
        if sample_softmax > 0:
            self.out_layer = nn.Linear(d_model, n_token)
            if tie_weight:
                self.out_layer.weight = self.word_emb.weight  # type: ignore[assignment]  # noqa: E501
            self.tie_weight = tie_weight
            self.sampler = LogUniformSampler(n_token, sample_softmax)

        # use adaptive softmax (including standard softmax)
        else:
            self.crit = ProjectedAdaptiveLogSoftmax(
                n_token, d_embed, d_model, cutoffs, div_val=div_val
            )

            if tie_weight:
                for i in range(len(self.crit.out_layers)):
                    self.crit.out_layers[i].weight = self.word_emb.emb_layers[
                        i
                    ].weight

            if tie_projs:
                for i, tie_proj in enumerate(tie_projs):
                    if tie_proj and div_val == 1 and d_model != d_embed:
                        self.crit.out_projs[i] = self.word_emb.emb_projs[0]
                    elif tie_proj and div_val != 1:
                        self.crit.out_projs[i] = self.word_emb.emb_projs[i]

        self.same_length = same_length
        self.clamp_len = clamp_len

        self._create_params()

    def _create_params(self) -> None:
        if self.attn_type == 0:  # default attention
            self.pos_emb = PositionalEmbedding(self.d_model)
            self.r_w_bias = nn.Parameter(
                torch.Tensor(self.n_head, self.d_head)
            )
            self.r_r_bias = nn.Parameter(
                torch.Tensor(self.n_head, self.d_head)
            )
        elif self.attn_type == 1:  # learnable
            self.r_emb = nn.Parameter(
                torch.Tensor(
                    self.n_layer, self.max_klen, self.n_head, self.d_head
                )
            )
            self.r_w_bias = nn.Parameter(
                torch.Tensor(self.n_layer, self.n_head, self.d_head)
            )
            self.r_bias = nn.Parameter(
                torch.Tensor(self.n_layer, self.max_klen, self.n_head)
            )
        elif self.attn_type == 2:  # absolute standard
            self.pos_emb = PositionalEmbedding(self.d_model)
        elif self.attn_type == 3:  # absolute deeper SA
            self.r_emb = nn.Parameter(
                torch.Tensor(
                    self.n_layer, self.max_klen, self.n_head, self.d_head
                )
            )

    def reset_length(self, tgt_len: int, ext_len: int, mem_len: int) -> None:
        self.tgt_len = tgt_len
        self.mem_len = mem_len
        self.ext_len = ext_len

    def init_mems(self) -> list[torch.Tensor] | None:
        if self.mem_len is not None and self.mem_len > 0:
            mems = []
            param = next(self.parameters())
            for _ in range(self.n_layer + 1):
                empty = torch.empty(0, dtype=param.dtype, device=param.device)
                mems.append(empty)

            return mems
        else:
            return None

    def _update_mems(
        self,
        hids: typing.Sequence[torch.Tensor],
        mems: typing.Sequence[torch.Tensor] | None,
        qlen: int,
        mlen: int,
    ) -> list[torch.Tensor] | None:
        # does not deal with None
        if mems is None:
            return None

        # mems is not None
        assert len(hids) == len(mems), "len(hids) != len(mems)"

        # There are `mlen + qlen` steps that can be cached into mems
        # For the next step, the last `ext_len` of the `qlen` tokens
        # will be used as the extended context. Hence, we only cache
        # the tokens from `mlen + qlen - self.ext_len - self.mem_len`
        # to `mlen + qlen - self.ext_len`.
        with torch.no_grad():
            new_mems = []
            end_idx = mlen + max(0, qlen - 0 - (self.ext_len or 0))
            beg_idx = max(0, end_idx - (self.mem_len or 0))
            for i in range(len(hids)):
                cat = torch.cat([mems[i], hids[i]], dim=0)
                new_mems.append(cat[beg_idx:end_idx].detach())

        return new_mems

    def _forward(
        self,
        dec_inp: torch.Tensor,
        mems: typing.Sequence[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
        qlen, bsz = dec_inp.size()

        word_emb = self.word_emb(dec_inp)

        mlen = mems[0].size(0) if mems is not None else 0
        klen = mlen + qlen
        if self.same_length:
            all_ones = word_emb.new_ones(qlen, klen)
            mask_len = klen - (self.mem_len or 0)
            if mask_len > 0:
                mask_shift_len = qlen - mask_len
            else:
                mask_shift_len = qlen
            dec_attn_mask = (
                torch.triu(all_ones, 1 + mlen)
                + torch.tril(all_ones, -mask_shift_len)
            ).byte()[:, :, None]  # -1
        else:
            dec_attn_mask = torch.triu(
                word_emb.new_ones(qlen, klen), diagonal=1 + mlen
            ).byte()[:, :, None]

        hids = []
        if self.attn_type == 0:  # default
            pos_seq = torch.arange(
                klen - 1,
                -1,
                -1.0,
                device=word_emb.device,
                dtype=word_emb.dtype,
            )
            if self.clamp_len > 0:
                pos_seq.clamp_(max=self.clamp_len)
            pos_emb = self.pos_emb(pos_seq)

            core_out = self.drop(word_emb)
            pos_emb = self.drop(pos_emb)

            hids.append(core_out)
            for i, layer in enumerate(self.layers):
                mems_i = None if mems is None else mems[i]
                core_out = layer(
                    core_out,
                    pos_emb,
                    self.r_w_bias,
                    self.r_r_bias,
                    dec_attn_mask=dec_attn_mask,
                    mems=mems_i,
                )
                hids.append(core_out)
        elif self.attn_type == 1:  # learnable
            core_out = self.drop(word_emb)
            hids.append(core_out)
            for i, layer in enumerate(self.layers):
                if self.clamp_len > 0:
                    r_emb = self.r_emb[i][-self.clamp_len :]
                    r_bias = self.r_bias[i][-self.clamp_len :]
                else:
                    r_emb, r_bias = self.r_emb[i], self.r_bias[i]

                mems_i = None if mems is None else mems[i]
                core_out = layer(
                    core_out,
                    r_emb,
                    self.r_w_bias[i],
                    r_bias,
                    dec_attn_mask=dec_attn_mask,
                    mems=mems_i,
                )
                hids.append(core_out)
        elif self.attn_type == 2:  # absolute
            pos_seq = torch.arange(
                klen - 1,
                -1,
                -1.0,
                device=word_emb.device,
                dtype=word_emb.dtype,
            )
            if self.clamp_len > 0:
                pos_seq.clamp_(max=self.clamp_len)
            pos_emb = self.pos_emb(pos_seq)

            core_out = self.drop(word_emb + pos_emb[-qlen:])

            hids.append(core_out)
            for i, layer in enumerate(self.layers):
                mems_i = None if mems is None else mems[i]
                if mems_i is not None and i == 0:
                    mems_i += pos_emb[:mlen]
                core_out = layer(
                    core_out, dec_attn_mask=dec_attn_mask, mems=mems_i
                )
                hids.append(core_out)
        elif self.attn_type == 3:
            core_out = self.drop(word_emb)

            hids.append(core_out)
            for i, layer in enumerate(self.layers):
                mems_i = None if mems is None else mems[i]
                if mems_i is not None and mlen > 0:
                    cur_emb = self.r_emb[i][:-qlen]
                    cur_size = cur_emb.size(0)
                    if cur_size < mlen:
                        cur_emb_pad = cur_emb[0:1].expand(
                            mlen - cur_size, -1, -1
                        )
                        cur_emb = torch.cat([cur_emb_pad, cur_emb], 0)
                    else:
                        cur_emb = cur_emb[-mlen:]
                    mems_i += cur_emb.view(mlen, 1, -1)
                core_out += self.r_emb[i][-qlen:].view(qlen, 1, -1)

                core_out = layer(
                    core_out, dec_attn_mask=dec_attn_mask, mems=mems_i
                )
                hids.append(core_out)

        core_out = self.drop(core_out)

        new_mems = self._update_mems(hids, mems, mlen, qlen)

        return core_out, new_mems

    def forward(
        self, data: torch.Tensor, target: torch.Tensor, *mems: torch.Tensor
    ) -> list[torch.Tensor]:
        # nn.DataParallel does not allow size(0) tensors to be broadcasted.
        # So, have to initialize size(0) mems inside the model forward.
        # Moreover, have to return new_mems to allow nn.DataParallel to piece
        # them together.
        if len(mems) == 0:
            init_mems = self.init_mems()
            assert init_mems is not None
            mems = tuple(init_mems)

        tgt_len = target.size(0)
        hidden, new_mems = self._forward(data, mems=mems)

        pred_hid = hidden[-tgt_len:]
        if self.sample_softmax > 0 and self.training:
            assert self.tie_weight
            logit = sample_logits(
                self.word_emb,
                self.out_layer.bias,
                target,
                pred_hid,
                self.sampler,
            )
            loss = -F.log_softmax(logit, -1)[:, :, 0]
        else:
            loss = self.crit(
                pred_hid.view(-1, pred_hid.size(-1)), target.view(-1)
            )
            loss = loss.view(tgt_len, -1)

        if new_mems is None:
            return [loss]
        else:
            return [loss] + new_mems
