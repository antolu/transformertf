from __future__ import annotations

import typing

import torch

from transformertf.models.bwlstm import BWLSTM3


def test_bouc_wen_lstm_forward_pass(
    bouc_wen_module: BWLSTM3, bouc_wen_module_config: dict[str, typing.Any]
) -> None:
    x = torch.rand(1, bouc_wen_module_config["seq_len"], 1)

    with torch.no_grad():
        y = bouc_wen_module(x)

    assert y["z"].shape == (1, bouc_wen_module_config["seq_len"], 3)
