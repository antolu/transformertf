from __future__ import annotations

import functools

import pytest
import pytorch_optimizer as py_optim
import torch

import transformertf.utils


@pytest.fixture
def model() -> torch.nn.Linear:
    return torch.nn.Linear(10, 10)


def test_configure_optimizer_adam(model: torch.nn.Linear) -> None:
    optimizer_fn = transformertf.utils.configure_optimizers("adam", lr=1e-3)
    optimizer = optimizer_fn(model.parameters())

    assert isinstance(optimizer, torch.optim.Adam)


def test_configure_optimizer_adamw(model: torch.nn.Linear) -> None:
    optimizer_fn = transformertf.utils.configure_optimizers("adamw", lr=1e-3)
    optimizer = optimizer_fn(model.parameters())

    assert isinstance(optimizer, torch.optim.AdamW)


def test_configure_optimizer_sgd(model: torch.nn.Linear) -> None:
    optimizer_fn = transformertf.utils.configure_optimizers("sgd", lr=1e-3)
    optimizer = optimizer_fn(model.parameters())

    assert isinstance(optimizer, torch.optim.SGD)


def test_configure_optimizer_ranger(model: torch.nn.Linear) -> None:
    optimizer_fn = transformertf.utils.configure_optimizers("ranger", lr=1e-3)
    optimizer = optimizer_fn(model.parameters())

    assert isinstance(optimizer, py_optim.Ranger)


def test_configure_optimizer_unknown() -> None:
    with pytest.raises(ValueError):  # noqa: PT011
        transformertf.utils.configure_optimizers("something", lr=1e-3)


def test_configure_optimizer_lr_auto(model: torch.nn.Linear) -> None:
    optimizer_fn = transformertf.utils.configure_optimizers("adam", lr="auto")
    optimizer = optimizer_fn(model.parameters())

    assert isinstance(optimizer, torch.optim.Adam)
    assert optimizer.defaults["lr"] == 1e-3


def test_configure_optimizer_partial(model: torch.nn.Linear) -> None:
    optimizer_fn = transformertf.utils.configure_optimizers(
        functools.partial(torch.optim.Adam, lr=1e-3)
    )
    optimizer = optimizer_fn(model.parameters())

    assert isinstance(optimizer, torch.optim.Adam)


def test_configure_lr_scheduler_plateau_patience_none(model: torch.nn.Linear) -> None:
    optimizer = torch.optim.Adam(model.parameters())
    lr_scheduler = transformertf.utils.configure_lr_scheduler(optimizer, "plateau")

    assert isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)


def test_configure_lr_scheduler_plateau_patience(model: torch.nn.Linear) -> None:
    optimizer = torch.optim.Adam(model.parameters())
    lr_scheduler = transformertf.utils.configure_lr_scheduler(
        optimizer, "plateau", reduce_on_plateau_patience=2
    )

    assert isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
    assert lr_scheduler.patience == 2


def test_configure_lr_scheduler_constant_then_cosine_max_epochs_none(
    model: torch.nn.Linear,
) -> None:
    optimizer = torch.optim.Adam(model.parameters())
    with pytest.raises(ValueError):  # noqa: PT011
        transformertf.utils.configure_lr_scheduler(optimizer, "constant_then_cosine")


def test_configure_lr_scheduler_constant_then_cosine_max_epochs(
    model: torch.nn.Linear,
) -> None:
    optimizer = torch.optim.Adam(model.parameters())
    lr_scheduler = transformertf.utils.configure_lr_scheduler(
        optimizer, "constant_then_cosine", max_epochs=10
    )

    assert isinstance(lr_scheduler, torch.optim.lr_scheduler.SequentialLR)


def test_configure_lr_scheduler_partial(model: torch.nn.Linear) -> None:
    optimizer = torch.optim.Adam(model.parameters())
    lr_scheduler = transformertf.utils.configure_lr_scheduler(
        optimizer, functools.partial(torch.optim.lr_scheduler.StepLR, step_size=1)
    )

    assert isinstance(lr_scheduler, torch.optim.lr_scheduler.StepLR)
    assert lr_scheduler.step_size == 1


def test_configure_lr_scheduler_class(model: torch.nn.Linear) -> None:
    optimizer = torch.optim.Adam(model.parameters())
    lr_scheduler = transformertf.utils.configure_lr_scheduler(
        optimizer, torch.optim.lr_scheduler.StepLR, step_size=1
    )

    assert isinstance(lr_scheduler, torch.optim.lr_scheduler.StepLR)


def test_configure_lr_scheduler_monitor(model: torch.nn.Linear) -> None:
    optimizer = torch.optim.Adam(model.parameters())
    lr_scheduler = transformertf.utils.configure_lr_scheduler(
        optimizer, "plateau", monitor="loss/validation"
    )

    assert isinstance(lr_scheduler, dict)
    assert lr_scheduler["monitor"] == "loss/validation"
    assert lr_scheduler["interval"] == "epoch"
