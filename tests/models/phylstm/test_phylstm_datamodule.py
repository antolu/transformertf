from __future__ import annotations

import pytest
import torch.utils.data

from transformertf.models.phylstm import PhyLSTMConfig, PhyLSTMDataModule

from ...conftest import CURRENT, DF_PATH, FIELD

config = PhyLSTMConfig(input_columns=CURRENT, target_column=FIELD)


def test_phylstm_datamodule_create() -> None:
    dm = PhyLSTMDataModule.from_parquet(
        config=config,
        train_dataset=DF_PATH,
        val_dataset=DF_PATH,
        seq_len=500,
        stride=1,
    )
    assert dm is not None


def test_phylstm_datamodule_create_with_config() -> None:
    dm = PhyLSTMDataModule.from_parquet(
        config, train_dataset=DF_PATH, val_dataset=DF_PATH
    )

    assert dm.hparams["seq_len"] == 500
    assert dm.hparams["stride"] == 1
    assert dm.hparams["lowpass_filter"] is True
    assert dm.hparams["mean_filter"] is True
    assert dm.hparams["downsample"] == 1
    assert dm.hparams["batch_size"] == 128
    assert dm.hparams["num_workers"] == 4


def test_phylstm_datamodule_hparams_correct() -> None:
    kwargs = dict(
        seq_len=500,
        stride=1,
        lowpass_filter=True,
        mean_filter=True,
        downsample=50,
        batch_size=32,
        num_workers=4,
        remove_polynomial=True,
        polynomial_degree=2,
        polynomial_iterations=2000,
        input_columns=["a"],
        target_column="b",
        model_dir="model_dir",
    )

    dm = PhyLSTMDataModule.from_parquet(
        config,
        train_dataset=DF_PATH,
        val_dataset=DF_PATH,
        **kwargs,  # type: ignore[arg-type]
    )

    correct_hparams = {
        "seq_len": 500,
        "min_seq_len": None,
        "randomize_seq_len": False,
        "stride": 1,
        "lowpass_filter": True,
        "mean_filter": True,
        "downsample": 50,
        "batch_size": 32,
        "num_workers": 4,
        "input_columns": ["a"],
        "target_column": "b",
        "model_dir": "model_dir",
        "normalize": True,
        "remove_polynomial": True,
        "polynomial_degree": 2,
        "polynomial_iterations": 2000,
        "target_depends_on": None,
        "dtype": "float32",
    }

    hparams = dict(dm.hparams)
    for key, value in correct_hparams.items():
        assert hparams.pop(key) == value

    assert len(hparams) == 0


def test_phylstm_datamodule_prepare_data() -> None:
    dm = PhyLSTMDataModule.from_parquet(
        config=config,
        train_dataset=DF_PATH,
        val_dataset=DF_PATH,
    )

    dm.prepare_data()


def test_phylstm_datamodule_setup_before_prepare() -> None:
    dm = PhyLSTMDataModule.from_parquet(
        config=config,
        train_dataset=DF_PATH,
        val_dataset=DF_PATH,
    )

    dm.setup()

    with pytest.raises(ValueError):
        dm.train_dataloader()

    with pytest.raises(ValueError):
        dm.val_dataloader()


@pytest.fixture(scope="module")
def phylstm_datamodule() -> PhyLSTMDataModule:
    dm = PhyLSTMDataModule.from_parquet(
        config=config,
        train_dataset=DF_PATH,
        val_dataset=DF_PATH,
        seq_len=500,
        stride=1,
        lowpass_filter=True,
        mean_filter=True,
        downsample=50,
        batch_size=128,
        num_workers=4,
        model_dir="model_dir",
    )

    dm.prepare_data()
    dm.setup()

    return dm


def test_phylstm_datamodule_setup(
    phylstm_datamodule: PhyLSTMDataModule,
) -> None:
    assert phylstm_datamodule is not None


def test_phylstm_datamodule_train_dataset(
    phylstm_datamodule: PhyLSTMDataModule,
) -> None:
    dm = phylstm_datamodule

    dataset = dm.train_dataset
    assert dataset is not None
    assert isinstance(dataset, torch.utils.data.Dataset)

    dataloader = dm.train_dataloader()
    assert dataloader is not None
    assert isinstance(dataloader, torch.utils.data.DataLoader)


def test_phylstm_datamodule_val_dataset(
    phylstm_datamodule: PhyLSTMDataModule,
) -> None:
    dm = phylstm_datamodule

    dataset = dm.val_dataset
    assert dataset is not None
    assert isinstance(dataset, torch.utils.data.Dataset)

    dataloader = dm.val_dataloader()
    assert dataloader is not None
    assert isinstance(dataloader, torch.utils.data.DataLoader)

    samples = [sample for sample in dataloader]

    assert samples[-1]["input"].shape == (1, dm.hparams["seq_len"], 1)
    assert samples[-1]["target"].shape == (1, dm.hparams["seq_len"], 1)
