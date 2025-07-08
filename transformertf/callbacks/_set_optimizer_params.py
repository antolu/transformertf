from __future__ import annotations

import logging

import lightning as L

from ..models import LightningModuleBase

log = logging.getLogger(__name__)


class SetOptimizerParamsCallback(L.pytorch.callbacks.Callback):
    """
    Lightning callback for setting optimizer and scheduler parameters at training start.

    This callback addresses a limitation in Lightning's checkpoint loading mechanism
    where optimizer parameters are not automatically restored when resuming training.
    It's particularly useful for:
    - Population-based training algorithms that modify optimizer parameters
    - Hyperparameter optimization workflows that need parameter restoration
    - Resuming training with different optimizer settings than initially configured
    - Fine-tuning scenarios where optimizer parameters need adjustment

    The callback sets specified optimizer and learning rate scheduler parameters
    at the beginning of training, ensuring consistent parameter values regardless
    of checkpoint loading behavior.

    Parameters
    ----------
    lr : float, optional
        Learning rate to set for the optimizer. If None, learning rate is not modified.
    momentum : float, optional
        Momentum value to set for SGD-type optimizers. If None, momentum is not modified.
        Only applies if the optimizer supports momentum.
    weight_decay : float, optional
        Weight decay (L2 regularization) to set for the optimizer. If None, weight decay
        is not modified.
    **kwargs : float
        Additional optimizer or scheduler parameters to set. Parameter names should
        match those used by the optimizer or scheduler classes. Only float values
        are accepted.

    Attributes
    ----------
    lr : float or None
        Learning rate value to set.
    momentum : float or None
        Momentum value to set.
    weight_decay : float or None
        Weight decay value to set.
    params : dict[str, float]
        Additional parameters to set on optimizer and scheduler.

    Methods
    -------
    on_train_start(trainer, pl_module)
        Set optimizer and scheduler parameters at the start of training.

    Raises
    ------
    ValueError
        If unsupported parameter types are provided (non-float values).
    ValueError
        If multiple optimizers are configured (only single optimizer supported).
    ValueError
        If multiple learning rate schedulers are configured (only single scheduler supported).

    Notes
    -----
    - Only supports single optimizer configurations
    - Only supports single learning rate scheduler configurations
    - Parameters are only set if they exist in the optimizer/scheduler parameter groups
    - Gracefully handles missing parameters without raising errors
    - Provides debug logging for all parameter updates

    Examples
    --------
    Basic usage with common parameters:

    >>> callback = SetOptimizerParamsCallback(
    ...     lr=0.001,
    ...     momentum=0.9,
    ...     weight_decay=1e-4
    ... )
    >>> trainer = L.Trainer(callbacks=[callback])

    Setting additional optimizer-specific parameters:

    >>> callback = SetOptimizerParamsCallback(
    ...     lr=0.001,
    ...     betas=(0.9, 0.999),  # Note: this would fail as betas is not float
    ...     eps=1e-8,
    ...     amsgrad=True  # Note: this would fail as amsgrad is not float
    ... )

    Setting scheduler parameters:

    >>> callback = SetOptimizerParamsCallback(
    ...     lr=0.001,
    ...     gamma=0.95,  # For ExponentialLR
    ...     step_size=10  # For StepLR
    ... )
    >>> trainer = L.Trainer(callbacks=[callback])

    Integration with population-based training:

    >>> # In a population-based training loop:
    >>> best_params = {"lr": 0.002, "weight_decay": 1e-3}
    >>> callback = SetOptimizerParamsCallback(**best_params)
    >>> trainer = L.Trainer(callbacks=[callback])
    >>> trainer.fit(model, ckpt_path="population_member.ckpt")

    See Also
    --------
    SetOptimizerLRCallback : For dynamic learning rate updates during training
    lightning.pytorch.callbacks.LearningRateMonitor : For logging LR changes
    """

    def __init__(
        self,
        lr: float | None = None,
        momentum: float | None = None,
        weight_decay: float | None = None,
        **kwargs: float,
    ):
        super().__init__()
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay

        self.params = {
            key: value for key, value in kwargs.items() if isinstance(value, float)
        }

        remaining_keys = set(kwargs.keys()) - set(self.params.keys())
        if remaining_keys:
            msg = f"Unsupported optimizer parameters: {remaining_keys}"
            raise ValueError(msg)

    def on_train_start(
        self, trainer: L.Trainer, pl_module: LightningModuleBase
    ) -> None:
        """
        Set optimizer and learning rate scheduler parameters at training start.

        Called automatically by Lightning at the beginning of training.
        Updates optimizer parameter groups and scheduler attributes with
        the specified values, ensuring consistent parameter settings
        regardless of checkpoint loading.

        Parameters
        ----------
        trainer : L.Trainer
            Lightning trainer containing optimizers and schedulers to update.
        pl_module : LightningModuleBase
            Lightning module (not used in this callback).

        Raises
        ------
        ValueError
            If multiple optimizers are configured (only single optimizer supported).
        ValueError
            If multiple learning rate schedulers are configured (only single scheduler supported).

        Notes
        -----
        - Updates all parameter groups in the optimizer
        - Only sets parameters that exist in the parameter groups
        - Provides debug logging for all parameter updates
        - Gracefully handles missing scheduler parameters
        - Processes both standard parameters (lr, momentum, weight_decay) and custom parameters
        """
        optimizers = trainer.optimizers

        if len(optimizers) > 1:
            msg = f"The {self.__class__.__name__} callback only supports a single optimizer."
            raise ValueError(msg)

        optimizer = optimizers[0]

        for param_group in optimizer.param_groups:
            if self.lr is not None and "lr" in param_group:
                log.debug(f"Setting LR to {self.lr}.")
                param_group["lr"] = self.lr
            if self.momentum is not None and "momentum" in param_group:
                log.debug(f"Setting momentum to {self.momentum}.")
                param_group["momentum"] = self.momentum
            if self.weight_decay is not None and "weight_decay" in param_group:
                log.debug(f"Setting weight decay to {self.weight_decay}.")
                param_group["weight_decay"] = self.weight_decay
            for key, value in self.params.items():
                param_group[key] = value

            for param_name, param_value in self.params.items():
                if param_name not in param_group:
                    continue

                log.debug(f"Setting optimizer parameter {param_name} to {param_value}.")
                param_group[param_name] = param_value

        lr_schedulers = trainer.lr_scheduler_configs

        if len(lr_schedulers) == 0:
            log.debug(
                "No LR scheduler found, skipping setting LR scheduler parameters."
            )
            return

        if len(lr_schedulers) > 1:
            msg = f"The {self.__class__.__name__} callback only supports a single LR scheduler."
            raise ValueError(msg)

        lr_scheduler = lr_schedulers[0].scheduler

        for param_name, param_value in self.params.items():
            if not hasattr(lr_scheduler, param_name):
                continue

            log.debug(f"Setting LR scheduler parameter {param_name} to {param_value}.")
            setattr(lr_scheduler, param_name, param_value)
