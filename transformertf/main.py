"""
Command-line interface and training orchestration for TransformerTF.

This module provides the main entry point for the TransformerTF framework through a
Lightning CLI interface. It handles configuration parsing, logging setup, model training,
and experiment management. The CLI supports all model types and data modules in the
framework with extensive configuration options.

Classes
-------
LightningCLI : lightning.pytorch.cli.LightningCLI
    Extended Lightning CLI with TransformerTF-specific features
NeptuneLoggerSaveConfigCallback : lightning.pytorch.cli.SaveConfigCallback
    Custom callback for saving configurations to Neptune logger

Functions
---------
setup_logger : Callable[[int], None]
    Configure logging with Rich formatting
main : Callable[[], None]
    Main entry point for the CLI application
add_trainer_defaults : Callable[[LightningArgumentParser], None]
    Add default trainer configuration
add_callback_defaults : Callable[[LightningArgumentParser], None]
    Add default callback configuration
add_seq_len_link : Callable[[LightningArgumentParser], None]
    Link sequence length parameters between data and model
add_num_features_link : Callable[[LightningArgumentParser], None]
    Link feature count parameters between data and model

Examples
--------
Train a Temporal Fusion Transformer:

    $ transformertf fit --config sample_configs/tft_config.yml

Train with custom parameters:

    $ transformertf fit \\
        --model.class_path transformertf.models.temporal_fusion_transformer.TemporalFusionTransformer \\
        --data.class_path transformertf.data.EncoderDecoderDataModule \\
        --data.init_args.train_df_paths='["data.parquet"]'

Run prediction:

    $ transformertf predict --config config.yml --ckpt_path checkpoints/best.ckpt
"""

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
from transformertf.models.attention_lstm import (  # noqa: F401
    AttentionLSTM,
)
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
    """
    Configure logging with Rich formatting for enhanced output.

    Sets up a root logger with Rich formatting for colored and structured console output.
    The logging level is controlled by an integer parameter with multiple verbosity levels.
    Also configures matplotlib logging to reduce noise.

    Parameters
    ----------
    logging_level : int, default=0
        Verbosity level for logging:
        - 0: WARNING level (default, minimal output)
        - 1: INFO level (moderate verbosity)
        - 2+: DEBUG level (maximum verbosity)

    Examples
    --------
    Set up basic logging:

    >>> setup_logger()  # WARNING level

    Enable verbose logging:

    >>> setup_logger(2)  # DEBUG level

    Notes
    -----
    This function modifies the global logging configuration. It adds a Rich handler
    to the root logger and sets consistent formatting. The matplotlib logger is
    always set to WARNING to reduce noise from plot generation.
    """
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
    """
    Extended Lightning CLI with TransformerTF-specific features.

    This class extends Lightning's CLI to provide TransformerTF-specific functionality
    including automatic parameter linking between data modules and models, Neptune
    logger integration, transfer learning support, and specialized callback management.

    Attributes
    ----------
    model : LightningModuleBase
        The TransformerTF model instance
    datamodule : DataModuleBase
        The TransformerTF data module instance

    Parameters
    ----------
    *args : Any
        Positional arguments passed to parent LightningCLI
    **kwargs : Any
        Keyword arguments passed to parent LightningCLI

    Examples
    --------
    Create a CLI instance (typically called in main()):

    >>> cli = LightningCLI(
    ...     model_class=LightningModuleBase,
    ...     datamodule_class=DataModuleBase,
    ...     subclass_mode_model=True,
    ...     subclass_mode_data=True
    ... )

    Notes
    -----
    The CLI automatically configures:
    - OmegaConf parser mode for YAML configuration support
    - Parameter linking between data and model components
    - Default callbacks for checkpointing and monitoring
    - Neptune logger integration for experiment tracking
    - Transfer learning from checkpoint files
    """

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
    """
    Add default trainer configuration to the Lightning argument parser.

    Configures trainer defaults that are specific to TransformerTF workflows,
    including disabling distributed sampling by default.

    Parameters
    ----------
    parser : LightningArgumentParser
        The Lightning CLI argument parser to configure

    Notes
    -----
    Sets trainer.use_distributed_sampler to False as the default behavior
    for TransformerTF training workflows.
    """
    parser.set_defaults({"trainer.use_distributed_sampler": False})


def add_callback_defaults(parser: LightningArgumentParser) -> None:
    """
    Add default callback configuration to the Lightning argument parser.

    Configures standard callbacks for TransformerTF training including learning rate
    monitoring, model summary display, and checkpoint management. Sets up two
    checkpoint callbacks: one for periodic saving and one for best model selection.

    Parameters
    ----------
    parser : LightningArgumentParser
        The Lightning CLI argument parser to configure

    Notes
    -----
    Configures the following callbacks:
    - LearningRateMonitor: Logs learning rate at epoch intervals
    - RichModelSummary: Displays model architecture with max depth 2
    - ModelCheckpoint (periodic): Saves all checkpoints every 50 epochs
    - ModelCheckpoint (best): Saves only the best model based on validation loss

    All checkpoints are saved to the "checkpoints" directory with descriptive filenames
    including epoch number and validation loss.
    """
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
    """
    Link sequence length parameters between data module and model.

    Automatically propagates sequence length settings from the data module to the model
    to ensure consistency. This prevents configuration errors where the model expects
    different sequence lengths than what the data module provides.

    Parameters
    ----------
    parser : LightningArgumentParser
        The Lightning CLI argument parser to configure

    Notes
    -----
    Links the following parameter pairs:
    - data.seq_len -> model.init_args.seq_len (basic sequence length)
    - data.ctxt_seq_len -> model.init_args.ctxt_seq_len (context sequence length)
    - data.tgt_seq_len -> model.init_args.tgt_seq_len (target sequence length)

    The linking is applied at instantiation time to ensure the model receives
    the correct sequence length parameters from the data configuration.
    """
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
    """
    Link feature count parameters between data module and model.

    Automatically propagates feature dimensions from the data module to the model
    to ensure the model architecture matches the data. This supports both
    encoder-decoder and sequence-to-sequence model architectures.

    Parameters
    ----------
    parser : LightningArgumentParser
        The Lightning CLI argument parser to configure

    Notes
    -----
    Links the following parameter pairs:

    For encoder-decoder models:
    - data.num_past_known_covariates -> model.init_args.num_past_features
    - data.num_future_known_covariates -> model.init_args.num_future_features

    For sequence-to-sequence models:
    - data.num_past_known_covariates -> model.init_args.num_features

    The linking ensures that models receive the correct input feature dimensions
    based on the data configuration, preventing dimension mismatch errors.
    """
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
    """
    Main entry point for the TransformerTF command-line interface.

    Initializes the TransformerTF CLI with optimized PyTorch settings and configures
    the Lightning CLI for subclass mode operation. This allows dynamic selection
    of model and data module classes from the command line or configuration files.

    Examples
    --------
    This function is typically called when running the CLI:

    $ transformertf fit --config config.yml
    $ transformertf predict --ckpt_path model.ckpt

    Notes
    -----
    Sets PyTorch float32 matrix multiplication precision to "high" for improved
    performance on modern hardware. Enables subclass mode for both models and
    data modules, allowing any registered class to be used via configuration.
    """
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
