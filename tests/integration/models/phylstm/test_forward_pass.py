from __future__ import annotations

import typing

import torch

from transformertf.models.phylstm import PhyLSTM


def test_phylstm_forward_pass(
    phylstm_module: PhyLSTM, phylstm_module_config: dict[str, typing.Any]
) -> None:
    x = torch.rand(1, phylstm_module_config["sequence_length"], 1)

    with torch.no_grad():
        y = phylstm_module(x)

    assert y["z"].shape == (1, phylstm_module_config["sequence_length"], 3)
