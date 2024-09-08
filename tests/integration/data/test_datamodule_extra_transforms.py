"""
Tests to makes sure that extra transforms
"""

from __future__ import annotations

import os
import pathlib
import typing

import lightning as L
import pytest

import transformertf.data.transform
from transformertf.data import EncoderDecoderDataModule, TransformCollection
from transformertf.models.pete import PETE


@pytest.fixture
def calibration_fn_path() -> str:
    return os.fspath(pathlib.Path(__file__).parent / "calibration_fn.csv")


@pytest.fixture(scope="module")
def encoder_decoder_datamodule_config(
    df_path: str, current_key: str, field_key: str
) -> dict[str, typing.Any]:
    return {
        "train_df_paths": df_path,
        "val_df_paths": df_path,
        "num_workers": 0,
        "known_covariates": [current_key],
        "target_covariate": field_key,
        "target_depends_on": current_key,
        "ctxt_seq_len": 100,
        "tgt_seq_len": 50,
    }


def test_datamodule_save_extra_transforms(
    encoder_decoder_datamodule_config: dict[str, typing.Any],
    calibration_fn_path: str,
    tmp_dir: str,
) -> None:
    encoder_decoder_datamodule_config["extra_transforms"] = {
        encoder_decoder_datamodule_config["target_covariate"]: [
            transformertf.data.transform.DiscreteFunctionTransform(
                xs_=calibration_fn_path
            )
        ]
    }

    datamodule = EncoderDecoderDataModule(**encoder_decoder_datamodule_config)
    datamodule.prepare_data()

    ckpt_path = os.path.join(tmp_dir, "test_extra_transforms.ckpt")

    model = PETE(
        num_past_features=datamodule.num_past_known_covariates,
        num_future_features=datamodule.num_future_known_covariates,
        ctxt_seq_len=datamodule.ctxt_seq_len,
        n_dim_model=16,
        n_dim_fc=8,
    )

    trainer = L.Trainer(accelerator="cpu", fast_dev_run=True, max_steps=1)
    trainer.fit(model, datamodule)
    trainer.save_checkpoint(ckpt_path)

    try:
        datamodule_loaded = EncoderDecoderDataModule.load_from_checkpoint(ckpt_path)
    except ValueError as e:  # extra transforms are not saved
        raise AssertionError from e

    for original_transforms, loaded_transforms in zip(
        datamodule.transforms.values(),
        datamodule_loaded.transforms.values(),
        strict=False,
    ):
        assert isinstance(original_transforms, TransformCollection)
        assert isinstance(loaded_transforms, TransformCollection)
        assert str(original_transforms) == str(loaded_transforms)
