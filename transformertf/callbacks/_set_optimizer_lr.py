from __future__ import annotations

import logging
import os
import pathlib
import typing
from typing import Any

import lightning as L

from ..models import LightningModuleBase

log = logging.getLogger(__name__)


class SetOptimizerLRCallback(L.pytorch.callbacks.Callback):
    """
    Lightning callback for dynamic learning rate adjustment from external file.

    This callback enables external control of learning rates during training by
    reading values from a specified file. It's particularly useful for:
    - Interactive learning rate tuning during long training runs
    - Population-based training algorithms that need external LR control
    - Manual intervention when monitoring training progress
    - Automated hyperparameter optimization systems

    The callback monitors a file and updates optimizer learning rates when the
    file contains a valid float value. After reading, the file is automatically
    deleted to prevent repeated applications of the same value.

    Parameters
    ----------
    lr_file : str or os.PathLike, default="/tmp/lr.txt"
        Path to the file containing the new learning rate value. The file
        should contain a single float value that can be parsed.
    on : {"epoch", "step"}, default="epoch"
        When to check for learning rate updates:
        - "epoch": Check at the start of each training epoch
        - "step": Check at the start of each training step
        Use "epoch" for coarse-grained control, "step" for fine-grained control.
    to : list of int, optional
        List of optimizer indices to update. If None (default), all optimizers
        are updated. Use this to selectively update specific optimizers in
        multi-optimizer setups.

    Attributes
    ----------
    lr_file : pathlib.Path
        Path object for the learning rate file.
    on : str
        Timing of learning rate checks ("epoch" or "step").
    to : list of int or None
        Optimizer indices to update, or None for all optimizers.

    Methods
    -------
    on_train_epoch_start(trainer, pl_module)
        Check and update learning rate at epoch start (if on="epoch").
    on_train_batch_start(trainer, pl_module, batch, batch_idx, dataloader_idx)
        Check and update learning rate at batch start (if on="step").
    set_lr(trainer)
        Read learning rate from file and update optimizers.
    state_dict()
        Return callback state for checkpointing.
    load_state_dict(state_dict)
        Load callback state from checkpoint.

    Notes
    -----
    - File is automatically deleted after successful reading
    - Gracefully handles missing files and invalid file contents
    - Supports distributed training with proper synchronization
    - File operations are only performed on global rank 0
    - All optimizers are updated unless specific indices are provided

    Examples
    --------
    Basic usage with default settings:

    >>> callback = SetOptimizerLRCallback()
    >>> trainer = L.Trainer(callbacks=[callback])
    # During training, write "0.001" to /tmp/lr.txt to update LR

    Custom file path and step-wise updates:

    >>> callback = SetOptimizerLRCallback(
    ...     lr_file="/path/to/my_lr.txt",
    ...     on="step"
    ... )
    >>> trainer = L.Trainer(callbacks=[callback])

    Update only specific optimizers:

    >>> callback = SetOptimizerLRCallback(
    ...     lr_file="/tmp/lr.txt",
    ...     to=[0, 2]  # Update only first and third optimizers
    ... )
    >>> trainer = L.Trainer(callbacks=[callback])

    Integration with monitoring and external tools:

    >>> # In a separate monitoring script:
    >>> # if validation_loss > threshold:
    >>> #     with open("/tmp/lr.txt", "w") as f:
    >>> #         f.write(str(new_lr))

    See Also
    --------
    lightning.pytorch.callbacks.LearningRateMonitor : For logging LR changes
    SetOptimizerParamsCallback : For setting other optimizer parameters
    """

    def __init__(
        self,
        lr_file: str | os.PathLike = "/tmp/lr.txt",
        on: typing.Literal["epoch", "step"] = "epoch",
        to: list[int] | None = None,
    ):
        super().__init__()
        self.lr_file = pathlib.Path(lr_file)
        self.on = on
        self.to = to

    def on_train_epoch_start(
        self, trainer: L.pytorch.Trainer, pl_module: LightningModuleBase
    ) -> None:
        """
        Check for learning rate updates at the start of each training epoch.

        Called automatically by Lightning at the beginning of each epoch.
        Only performs learning rate update if on="epoch" was specified.

        Parameters
        ----------
        trainer : L.pytorch.Trainer
            Lightning trainer instance containing optimizers.
        pl_module : LightningModuleBase
            Lightning module (not used in this callback).
        """
        if self.on == "epoch":
            self.set_lr(trainer)

    def on_train_batch_start(
        self,
        trainer: L.pytorch.Trainer,
        pl_module: LightningModuleBase,
        batch: typing.Any,
        batch_idx: int = 0,
        dataloader_idx: int = 0,
    ) -> None:
        """
        Check for learning rate updates at the start of each training batch.

        Called automatically by Lightning before processing each batch.
        Only performs learning rate update if on="step" was specified.

        Parameters
        ----------
        trainer : L.pytorch.Trainer
            Lightning trainer instance containing optimizers.
        pl_module : LightningModuleBase
            Lightning module (not used in this callback).
        batch : Any
            Current training batch (not used in this callback).
        batch_idx : int, default=0
            Index of current batch (not used in this callback).
        dataloader_idx : int, default=0
            Index of current dataloader (not used in this callback).
        """
        if self.on == "step":
            self.set_lr(trainer)

    def set_lr(self, trainer: L.pytorch.Trainer) -> None:
        """
        Read learning rate from file and update optimizer learning rates.

        Attempts to read a float value from the specified file and update
        the learning rates of target optimizers. Handles file reading errors
        gracefully and ensures proper synchronization in distributed training.

        Parameters
        ----------
        trainer : L.pytorch.Trainer
            Lightning trainer containing optimizers to update.

        Notes
        -----
        - File is automatically deleted after successful reading
        - Gracefully handles missing files or invalid contents
        - Updates all optimizers unless specific indices were provided
        - Includes distributed training synchronization barrier
        - Only global rank 0 process performs file deletion
        """
        if not self.lr_file.exists():
            log.debug(f"File {self.lr_file} does not exist.")
            return

        with open(self.lr_file, encoding="utf-8") as f:
            try:
                contents = f.read().strip()
            except ValueError:
                log.exception(
                    f"Could not read LR from file {self.lr_file}, Contents: {contents}"
                )
                return

            try:
                lr = float(contents)
            except ValueError:
                log.exception(
                    f"Could not convert contents of file {self.lr_file} "
                    f"to float: {contents}"
                )
                return

        log.info(f"Setting LR to {lr}.")
        if self.to is None:
            for optimizer in trainer.optimizers:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr
        else:
            for i in self.to:
                if i < len(trainer.optimizers):
                    for param_group in trainer.optimizers[i].param_groups:
                        param_group["lr"] = lr
                else:
                    log.error(f"Optimizer index {i} out of range, skipping setting LR.")

        trainer.strategy.barrier()

        if trainer.is_global_zero:
            log.info(f"LR set to {lr} on all processes, removing {self.lr_file}")
            os.remove(self.lr_file)

    def state_dict(self) -> dict[str, Any]:
        """
        Return callback state for checkpointing.

        Returns
        -------
        dict[str, Any]
            Dictionary containing callback state with lr_file path.
        """
        return {"lr_file": str(self.lr_file)}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """
        Load callback state from checkpoint.

        Parameters
        ----------
        state_dict : dict[str, Any]
            Dictionary containing callback state to restore.
        """
        if "lr_file" in state_dict:
            self.lr_file = pathlib.Path(state_dict["lr_file"])

        state_dict.pop("lr_file", None)
