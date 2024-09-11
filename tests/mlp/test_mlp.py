from __future__ import annotations

import pathlib

import numpy as np
import torch.nn

from sps_mlp_hystcomp import TFTPredictor


def test_tft_mlp_load_save_parameters(mlp_model_checkpoint: str, tmp_dir: str) -> None:
    predictor = TFTPredictor.load_from_checkpoint(mlp_model_checkpoint, device="cpu")

    tmp_path = tmp_dir + "/mlp_checkpoint.ckpt"

    predictor.export_parameters(pathlib.Path(tmp_path))

    predictor_loaded = TFTPredictor.load_from_checkpoint(tmp_path, device="cpu")

    for key, value in predictor._module.state_dict().items():  # noqa: SLF001
        assert (value == predictor_loaded._module.state_dict()[key]).all()  # noqa: SLF001
    for key, value in predictor.hparams.items():
        if isinstance(value, torch.nn.Module | dict):
            continue
        if isinstance(value, np.ndarray | torch.Tensor):
            assert (value == predictor_loaded.hparams[key]).all()
            continue

        assert predictor_loaded.hparams[key] == value

    predictor_loaded.load_parameters(pathlib.Path(tmp_path))

    for key, value in predictor._module.state_dict().items():  # noqa: SLF001
        if isinstance(value, torch.nn.Module):
            continue
        assert (value == predictor_loaded._module.state_dict()[key]).all()  # noqa: SLF001
    for key, value in predictor.hparams.items():
        if isinstance(value, torch.nn.Module | dict):
            continue
        if isinstance(value, np.ndarray | torch.Tensor):
            assert (value == predictor_loaded.hparams[key]).all()
            continue

        assert predictor_loaded.hparams[key] == value
