from __future__ import annotations

import typing

from transformertf.models.sa_bwlstm import SABWLSTM


def test_sa_create_model(bouc_wen_module_config: dict[str, typing.Any]) -> None:
    model = SABWLSTM(**bouc_wen_module_config)
    assert model is not None

    opt = model.configure_optimizers()

    assert len(opt) == 2


# def test_sa_compile_model(bouc_wen_module_config: dict[str, typing.Any]) -> None:
#     bouc_wen_module_config["compile_model"] = True
#
#     model = SABWLSTM(**bouc_wen_module_config)
#     model = torch.compile(model)
#
#     sample = torch.rand((2, 10, 1))
#     out = model(sample)
#
#     assert out is not None
