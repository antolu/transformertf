from __future__ import annotations

import datetime
import logging
import typing
from dataclasses import dataclass, field
from functools import partial

import torch
from torch.optim.lr_scheduler import LRScheduler

__all__ = [
    "BaseConfig",
]

log = logging.getLogger(__name__)

PRECISION = typing.Literal[
    "16-mixed", "bf16-mixed", "32-true", "64-true", 16, 32, 64
]


@dataclass(init=True)
class BaseConfig:
    batch_size: int = 128
    shuffle: bool = True
    num_workers: int = 4
    lr: float = 0.001
    weight_decay: float = 0.0001
    momentum: float = 0.9
    num_epochs: int = 1000
    gradient_clip_val: float = 1.0
    log_grad_norm: bool = False
    patience: int | None = None

    precision: PRECISION = "32-true"

    device: torch.device | str | None = None

    # data processing parameters
    seq_len: int = 500
    min_seq_len: int | None = None
    randomize_seq_len: bool = False
    stride: int = 1
    normalize: bool = True
    downsample: int = 1

    remove_polynomial: bool = False
    polynomial_degree: int = 1
    polynomial_iterations: int = 1000

    # physics parameters
    use_derivative: bool = True
    lowpass_filter: bool = False
    mean_filter: bool = False

    optimizer: str = "sgd"
    lr_scheduler: str | typing.Type[LRScheduler] | partial | None = None
    lr_scheduler_interval: typing.Literal["epoch", "step"] = "epoch"
    optimizer_kwargs: dict = field(default_factory=dict)

    dataset: str | None = None  # do not use this one in the code
    train_dataset: str | typing.Sequence[str] | None = None
    val_dataset: str | typing.Sequence[str] | None = None
    test_dataset: str | None = None
    predict_dataset: str | None = None

    checkpoint_every: int = 50
    validate_every: int = 50

    timestamp: str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir: str = "logs"
    model_dir: str | None = None
    callbacks: list = field(default_factory=list)
