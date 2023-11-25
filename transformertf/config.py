from __future__ import annotations

import datetime
import logging
import typing
from dataclasses import dataclass, field
from functools import partial

import torch
from torch.optim.lr_scheduler import LRScheduler

from .data import BaseTransform

__all__ = [
    "BaseConfig",
    "TimeSeriesBaseConfig",
    "TransformerBaseConfig",
]

log = logging.getLogger(__name__)

PRECISION = typing.Literal[
    "16-mixed", "bf16-mixed", "32-true", "64-true", 16, 32, 64
]


@dataclass(init=True)
class BaseConfig:
    # Data loading
    batch_size: int = 128
    shuffle: bool = True
    num_workers: int = 4

    # Optimizer
    optimizer: str = "sgd"
    lr_scheduler: str | typing.Type[LRScheduler] | partial | None = None
    lr_scheduler_interval: typing.Literal["epoch", "step"] = "epoch"
    optimizer_kwargs: dict = field(default_factory=dict)

    lr: float = 0.001
    weight_decay: float = 0.0001
    momentum: float = 0.9

    num_epochs: int = 1000
    gradient_clip_val: float = 1.0
    log_grad_norm: bool = False
    patience: int | None = None

    # CUDA
    precision: PRECISION = "32-true"
    device: torch.device | str | None = None

    # bulk data processing parameters
    normalize: bool = True
    downsample: int = 1
    remove_polynomial: bool = False

    target_depends_on: str | None = None
    polynomial_degree: int = 1
    polynomial_iterations: int = 1000

    lowpass_filter: bool = False
    mean_filter: bool = False

    # extra transforms, map from column name to transform
    extra_transforms: dict[str, BaseTransform] = field(default_factory=dict)

    # Data source
    dataset: str | None = None  # do not use this one in the code
    train_dataset: str | typing.Sequence[str] | None = None
    val_dataset: str | typing.Sequence[str] | None = None

    input_columns: str | typing.Sequence[str] | None = None
    target_column: str | None = None

    # Checkpointing
    checkpoint_every: int = 50
    validate_every: int = 50

    callbacks: list = field(default_factory=list)
    """ PyTorch Lightning callbacks """

    # Misc. to not be used manually
    timestamp: str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir: str = "logs"
    model_dir: str | None = None


@dataclass
class TimeSeriesBaseConfig(BaseConfig):
    """Adds additional parameters for generating time series datasets"""

    seq_len: int = 500
    out_seq_len: int = 300
    min_seq_len: int | None = None
    randomize_seq_len: bool = False
    stride: int = 1


@dataclass
class TransformerBaseConfig(BaseConfig):
    ctxt_seq_len: int = 500
    tgt_seq_len: int = 300
    min_ctxt_seq_len: int | None = None
    min_tgt_seq_len: int | None = None
    randomize_seq_len: bool = False
    stride: int = 1
