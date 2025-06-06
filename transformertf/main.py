from __future__ import annotations

import logging
import os
import pathlib
import sys
import typing
import warnings

import einops._torch_specific
import lightning as L
import lightning.pytorch.cli
import pytorch_optimizer  # noqa: F401
import rich
import rich.logging
import torch
from lightning import LightningModule
from lightning.pytorch.cli import LightningArgumentParser, LRSchedulerTypeUnion

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

warnings.filterwarnings("ignore", category=UserWarning)

einops._torch_specific.allow_ops_in_compiled_graph()  # noqa: SLF001


class _FilterCallback(logging.Filterer):
    def filter(self, record: logging.LogRecord) -> bool:
        return not (
            record.name == "neptune"
            and record.getMessage().startswith(
                "Error occurred during asynchronous operation processing:"
            )
        )


class NeptuneLoggerSaveConfigCallback(lightning.pytorch.cli.SaveConfigCallback):
    def save_config(
        self, trainer: L.Trainer, pl_module: L.LightningModule, stage: str
    ) -> None:
        if isinstance(trainer.logger, L.pytorch.loggers.neptune.NeptuneLogger):
            config = self.parser.dump(self.config, skip_none=False)

            with open(self.config_filename, "w", encoding="utf-8") as f:
                f.write(config)
            trainer.logger.experiment["model/config"].upload(self.config_filename)
            return

        super().save_config(trainer, pl_module, stage=stage)


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
        if hasattr(self.config, "fit") and hasattr(self.config.fit, "verbose"):
            setup_logger(self.config.fit.verbose)

        if hasattr(self.config, "fit") and hasattr(
            self.config.fit, "no_auto_configure_optimizers"
        ):
            self.auto_configure_optimizers = (
                not self.config.fit.no_auto_configure_optimizers
            )

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.add_argument(
            "-v",
            dest="verbose",
            action="count",
            default=0,
            help="Verbose flag. Can be used more than once.",
        )

        parser.add_argument(
            "-n",
            "--experiment-name",
            dest="experiment_name",
            type=str,
            default=None,
            help="Name of the experiment.",
        )

        parser.add_argument(
            "--transfer-ckpt",
            dest="transfer_ckpt",
            type=str,
            default=None,
            help="Path to the checkpoint resume to do transfer learning.",
        )

        parser.add_argument(
            "--no-auto-configure-optimizers",
            action="store_true",
            dest="no_auto_configure_optimizers",
            help="Do not auto-configure optimizers.",
        )

        parser.add_argument(
            "--lr_step_interval",
            default="epoch",
            choices=["epoch", "step"],
            help="Learning rate scheduler step interval",
        )

        add_trainer_defaults(parser)

        add_callback_defaults(parser)

        add_seq_len_link(parser)

        add_num_features_link(parser)

    def before_fit(self) -> None:
        # hijack model checkpoint callbacks to save to checkpoint_dir/version_{version}
        if (
            hasattr(self.config, "fit")
            and hasattr(self.config.fit, "experiment_name")
            and self.config.fit.experiment_name
        ):
            logger_name = self.config.fit.experiment_name

            if isinstance(self.trainer.logger, L.pytorch.loggers.TensorBoardLogger):
                self.trainer.logger._name = logger_name  # noqa: SLF001
        else:
            logger_name = ""
        try:
            if isinstance(self.trainer.logger, L.pytorch.loggers.TensorBoardLogger):
                version = self.trainer.logger.version
            else:
                version = "na"
        except TypeError:
            version = 0
        version_str = f"version_{version}"

        # if logger is a neptune logger, save the config to a temporary file and upload it
        # also track artifacts from datamodule
        if isinstance(self.trainer.logger, L.pytorch.loggers.neptune.NeptuneLogger):
            import neptune  # noqa: PLC0415

            # filter out errors caused by logging epoch more than once
            neptune.internal.operation_processors.async_operation_processor.logger.addFilter(
                _FilterCallback()
            )

            if "train_df_paths" in self.datamodule.hparams:
                for train_df_path in self.datamodule.hparams["train_df_paths"]:
                    self.trainer.logger.experiment["train/dataset"].track_files(
                        os.fspath(pathlib.Path(train_df_path).expanduser())
                    )
            if "val_df_paths" in self.datamodule.hparams:
                if isinstance(self.datamodule.hparams["val_df_paths"], str):
                    self.trainer.logger.experiment["validation/dataset"].track_files(
                        os.fspath(
                            pathlib.Path(
                                self.datamodule.hparams["val_df_paths"]
                            ).expanduser()
                        )
                    )
                else:
                    for val_df_path in self.datamodule.hparams["val_df_paths"]:
                        self.trainer.logger.experiment[
                            "validation/dataset"
                        ].track_files(os.fspath(pathlib.Path(val_df_path).expanduser()))

            # log the command used to launch training to neptune
            self.trainer.logger.experiment["source_code/argv"] = " ".join(sys.argv)

            self.trainer.logger.experiment.sync()

        for callback in self.trainer.callbacks:
            if isinstance(callback, lightning.pytorch.callbacks.ModelCheckpoint):
                if logger_name:
                    dirpath = os.path.join(callback.dirpath, logger_name, version_str)
                else:
                    dirpath = os.path.join(callback.dirpath, version_str)
                callback.dirpath = dirpath

        # patch LR monitor to log at the correct interval
        for callback in self.trainer.callbacks:
            if isinstance(callback, lightning.pytorch.callbacks.LearningRateMonitor):
                callback.logging_interval = self.config.fit.lr_step_interval

        self.trainer.callbacks.append(
            NeptuneLoggerSaveConfigCallback(
                parser=self.parser, config=self.config, overwrite=True
            )
        )

        # load checkpoint for transfer learning
        if (
            hasattr(self.config, "fit")
            and hasattr(self.config.fit, "transfer_ckpt")
            and self.config.fit.transfer_ckpt is not None
        ):
            transfer_ckpt = os.fspath(
                pathlib.Path(self.config.fit.transfer_ckpt).expanduser()
            )
            state_dict = torch.load(
                transfer_ckpt, map_location="cpu", weights_only=False
            )
            self.model.load_state_dict(state_dict["state_dict"], strict=False)

    def configure_optimizers(
        self,
        lightning_module: LightningModule,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: LRSchedulerTypeUnion | None = None,
    ) -> torch.optim.Optimizer | dict[str, typing.Any]:
        if lr_scheduler is None:
            return optimizer
        if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": lr_scheduler,
                    "monitor": lr_scheduler.monitor,
                    "interval": self.config.fit.lr_step_interval,
                },
            }
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": self.config.fit.lr_step_interval,
            },
        }


def add_trainer_defaults(parser: LightningArgumentParser) -> None:
    parser.set_defaults({"trainer.use_distributed_sampler": False})


def add_callback_defaults(parser: LightningArgumentParser) -> None:
    parser.add_lightning_class_args(
        lightning.pytorch.callbacks.LearningRateMonitor, "lr_monitor"
    )
    parser.set_defaults({"lr_monitor.logging_interval": "epoch"})

    # parser.add_lightning_class_args(
    #     lightning.pytorch.callbacks.RichProgressBar, "progress_bar"
    # )
    # parser.set_defaults({"progress_bar.refresh_rate": 1})

    parser.add_lightning_class_args(
        lightning.pytorch.callbacks.RichModelSummary, "model_summary"
    )
    parser.set_defaults({"model_summary.max_depth": 2})

    parser.add_lightning_class_args(
        lightning.pytorch.callbacks.ModelCheckpoint, "fit.checkpoint_every"
    )
    parser.set_defaults({
        "fit.checkpoint_every.save_top_k": -1,
        "fit.checkpoint_every.monitor": "validation/loss",
        "fit.checkpoint_every.mode": "min",
        "fit.checkpoint_every.dirpath": "checkpoints",
        "fit.checkpoint_every.filename": "epoch={epoch}-valloss={validation/loss:.4f}",
        "fit.checkpoint_every.every_n_epochs": 50,
    })

    parser.add_lightning_class_args(
        lightning.pytorch.callbacks.ModelCheckpoint, "fit.checkpoint_best"
    )
    parser.set_defaults({
        "fit.checkpoint_best.save_top_k": 1,
        "fit.checkpoint_best.monitor": "validation/loss",
        "fit.checkpoint_best.mode": "min",
        "fit.checkpoint_best.dirpath": "checkpoints",
        "fit.checkpoint_best.filename": "epoch={epoch}-valloss={validation/loss:.4f}",
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
        save_config_kwargs={"overwrite": True},
    )


if __name__ == "__main__":
    main()
