from __future__ import annotations

import typing

import torch

from transformertf.models.phytsmixer import PhyTSMixer


def test_phytsmixer_forward_pass(
    phytsmixer_module: PhyTSMixer, phytsmixer_module_config: dict[str, typing.Any]
) -> None:
    x_past = torch.rand(1, phytsmixer_module_config["ctxt_seq_len"], 2)
    x_future = torch.rand(1, phytsmixer_module_config["tgt_seq_len"], 1)

    batch = {
        "encoder_input": x_past,
        "decoder_input": torch.cat([x_future, x_future], dim=-1),
    }

    with torch.no_grad():
        y = phytsmixer_module(batch)

    assert y["z"].shape == (1, phytsmixer_module_config["tgt_seq_len"], 3)
