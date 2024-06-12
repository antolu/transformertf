from __future__ import annotations

import logging
import os
import typing

import jsonargparse
import lightning.pytorch.cli
import pytorch_optimizer  # noqa: F401
import rich
import rich.logging
import torch
from lightning.pytorch.cli import LightningArgumentParser

from transformertf.data import (
    DataModuleBase,
    EncoderDataModule,  # noqa: F401
    EncoderDecoderDataModule,  # noqa: F401
    TimeSeriesDataModule,  # noqa: F401
)
from transformertf.models import LightningModuleBase
from transformertf.models.bwlstm import BWLSTM1, BWLSTM2, BWLSTM3  # noqa: F401
from transformertf.models.lstm import LSTM  # noqa: F401
from transformertf.models.tsmixer import TSMixer  # noqa: F401


def setup_logger(logging_level: int = 0) -> None:
    log = logging.getLogger()

    ch = rich.logging.RichHandler()

    formatter = logging.Formatter(
        "%(message)s",
        "%Y-%m-%d %H:%M:%S",
    )
    ch.setFormatter(formatter)
    log.addHandler(ch)

    if logging_level >= 2:
        log.setLevel(logging.DEBUG)
        ch.setLevel(logging.DEBUG)
    elif logging_level >= 1:
        log.setLevel(logging.INFO)
        log.setLevel(logging.INFO)
    else:
        log.setLevel(logging.WARNING)
        log.setLevel(logging.WARNING)

    logging.getLogger("matplotlib").setLevel(logging.WARNING)


class LightningCLI(lightning.pytorch.cli.LightningCLI):
    model: LightningModuleBase
    datamodule: DataModuleBase

    def __init__(self, *args: typing.Any, **kwargs: typing.Any) -> None:
        super().__init__(*args, parser_kwargs={"parser_mode": "omegaconf"}, **kwargs)

    def before_instantiate_classes(self) -> None:
        if hasattr(self.config, "fit"):
            setup_logger(self.config.fit.verbose)

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.add_argument(
            "-v",
            dest="verbose",
            action="count",
            default=0,
            help="Verbose flag. Can be used more than once.",
        )

        parser.set_defaults({
            "trainer.logger": jsonargparse.lazy_instance(
                lightning.pytorch.loggers.TensorBoardLogger,
                save_dir="logs",
                name=None,
            ),
        })

        add_callback_defaults(parser)

        add_seq_len_link(parser)

        add_num_features_link(parser)

    def before_fit(self) -> None:
        # hijack model checkpoint callbacks to save to checkpoint_dir/version_{version}
        version = self.trainer.logger.version
        version_str = f"version_{version}"

        for callback in self.trainer.callbacks:
            if isinstance(callback, lightning.pytorch.callbacks.ModelCheckpoint):
                callback.dirpath = os.path.join(callback.dirpath, version_str)


def add_callback_defaults(parser: LightningArgumentParser) -> None:
    parser.add_lightning_class_args(
        lightning.pytorch.callbacks.LearningRateMonitor, "lr_monitor"
    )
    parser.set_defaults({"lr_monitor.logging_interval": "epoch"})

    parser.add_lightning_class_args(
        lightning.pytorch.callbacks.RichProgressBar, "progress_bar"
    )
    parser.set_defaults({"progress_bar.refresh_rate": 1})

    parser.add_lightning_class_args(
        lightning.pytorch.callbacks.RichModelSummary, "model_summary"
    )
    parser.set_defaults({"model_summary.max_depth": 2})

    parser.add_lightning_class_args(
        lightning.pytorch.callbacks.ModelCheckpoint, "fit.checkpoint_every"
    )
    parser.set_defaults({
        "fit.checkpoint_every.save_top_k": -1,
        "fit.checkpoint_every.monitor": "loss/validation",
        "fit.checkpoint_every.mode": "min",
        "fit.checkpoint_every.dirpath": "checkpoints",
        "fit.checkpoint_every.filename": "epoch={epoch}-valloss={loss/validation:.4f}",
        "fit.checkpoint_every.every_n_epochs": 50,
    })

    parser.add_lightning_class_args(
        lightning.pytorch.callbacks.ModelCheckpoint, "fit.checkpoint_best"
    )
    parser.set_defaults({
        "fit.checkpoint_best.save_top_k": 1,
        "fit.checkpoint_best.monitor": "loss/validation",
        "fit.checkpoint_best.mode": "min",
        "fit.checkpoint_best.dirpath": "checkpoints",
        "fit.checkpoint_best.filename": "epoch={epoch}-valloss={loss/validation:.4f}",
        "fit.checkpoint_best.save_last": "link",
        "fit.checkpoint_best.save_weights_only": False,
        "fit.checkpoint_best.auto_insert_metric_name": False,
        "fit.checkpoint_best.enable_version_counter": False,
    })


def add_seq_len_link(parser: LightningArgumentParser) -> None:
    parser.link_arguments(
        "data.seq_len",
        "model.init_args.seq_len",
        apply_on="instantiate",
    )

    parser.link_arguments(
        "data.ctxt_seq_len",
        "model.init_args.ctxt_seq_len",
        apply_on="instantiate",
    )

    parser.link_arguments(
        "data.tgt_seq_len",
        "model.init_args.tgt_seq_len",
        apply_on="instantiate",
    )


def add_num_features_link(
    parser: LightningArgumentParser,
) -> None:
    # encoder-decoder models
    parser.link_arguments(
        "data.num_past_known_covariates",
        "model.init_args.num_past_features",
        apply_on="instantiate",
    )
    parser.link_arguments(
        "data.num_future_known_covariates",
        "model.init_args.num_future_features",
        apply_on="instantiate",
    )

    # seq2seq models
    parser.link_arguments(
        "data.num_past_known_covariates",
        "model.init_args.num_features",
        apply_on="instantiate",
    )


def main() -> None:
    torch.set_float32_matmul_precision("high")
    LightningCLI(
        model_class=LightningModuleBase,
        datamodule_class=DataModuleBase,
        subclass_mode_model=True,
        subclass_mode_data=True,
    )


if __name__ == "__main__":
    main()
