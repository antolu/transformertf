from __future__ import annotations

import functools
import sys
import typing

import torch

from ...data import EncoderDecoderTargetSample
from ...nn import MLP
from ..bwlstm import BoucWenLoss, StepOutput
from ..bwlstm.typing import BWState3
from ..sa_bwlstm import SABWLSTM
from ._model import PETEModel

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override


class PETE(SABWLSTM):
    def __init__(
        self,
        ctxt_seq_len: int = 100,
        num_layers: int | tuple[int, int, int] = 3,
        n_enc_heads: int = 4,
        n_dim_model: int | tuple[int, int, int] = 350,
        n_dim_selection: int = 32,
        n_dim_fc: int | tuple[int, int, int] | None = None,
        dropout: float | tuple[float, float, float] = 0.2,
        loss_weights: BoucWenLoss.LossWeights | None = None,
        optimizer: str
        | functools.partial
        | typing.Callable[
            [typing.Iterator[torch.nn.Parameter]], torch.optim.Optimizer
        ] = "adam",
        lr: float | typing.Literal["auto"] = 1e-3,
        weight_decay: float | None = None,
        momentum: float | None = None,
        optimizer_kwargs: dict[str, typing.Any] | None = None,
        lr_scheduler: str
        | type[torch.optim.lr_scheduler.LRScheduler]
        | functools.partial
        | typing.Callable[[torch.optim.Optimizer], torch.optim.lr_scheduler.LRScheduler]
        | None = None,
        monitor: str | None = None,
        scheduler_interval: typing.Literal["step", "epoch"] = "epoch",
        max_epochs: int | None = None,
        reduce_on_plateau_patience: int | None = None,
        lr_scheduler_kwargs: dict[str, typing.Any] | None = None,
        sa_optimizer: str
        | functools.partial
        | typing.Callable[
            [typing.Iterator[torch.nn.Parameter]], torch.optim.Optimizer
        ] = "adam",
        sa_lr: float | typing.Literal["auto"] = 1e-3,
        sa_weight_decay: float | None = None,
        sa_momentum: float | None = None,
        sa_optimizer_kwargs: dict[str, typing.Any] | None = None,
        sa_lr_scheduler: str
        | type[torch.optim.lr_scheduler.LRScheduler]
        | functools.partial
        | typing.Callable[[torch.optim.Optimizer], torch.optim.lr_scheduler.LRScheduler]
        | None = None,
        sa_monitor: str | None = None,
        sa_scheduler_interval: typing.Literal["step", "epoch"] = "epoch",
        sa_max_epochs: int | None = None,
        sa_reduce_on_plateau_patience: int | None = None,
        sa_lr_scheduler_kwargs: dict[str, typing.Any] | None = None,
        lbfgs_start: typing.Literal[False] | int = False,  # noqa: FBT002
        lbfgs_lr: float = 1.0,
        lbfgs_max_iter: int = 20,
        lbfgs_history_size: int = 5,
        *,
        log_grad_norm: bool = False,
        compile_model: bool = False,
    ):
        super().__init__(
            num_layers=num_layers,
            n_dim_model=n_dim_model,
            n_dim_fc=n_dim_fc,
            dropout=dropout,
            loss_weights=loss_weights,
            optimizer=optimizer,
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            optimizer_kwargs=optimizer_kwargs,
            lr_scheduler=lr_scheduler,
            monitor=monitor,
            scheduler_interval=scheduler_interval,
            max_epochs=max_epochs,
            reduce_on_plateau_patience=reduce_on_plateau_patience,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            sa_optimizer=sa_optimizer,
            sa_lr=sa_lr,
            sa_weight_decay=sa_weight_decay,
            sa_momentum=sa_momentum,
            sa_optimizer_kwargs=sa_optimizer_kwargs,
            sa_lr_scheduler=sa_lr_scheduler,
            sa_monitor=sa_monitor,
            sa_scheduler_interval=sa_scheduler_interval,
            sa_max_epochs=sa_max_epochs,
            sa_reduce_on_plateau_patience=sa_reduce_on_plateau_patience,
            sa_lr_scheduler_kwargs=sa_lr_scheduler_kwargs,
            lbfgs_start=lbfgs_start,
            lbfgs_lr=lbfgs_lr,
            lbfgs_max_iter=lbfgs_max_iter,
            lbfgs_history_size=lbfgs_history_size,
            log_grad_norm=log_grad_norm,
            compile_model=compile_model,
        )

        self.encoder = PETEModel(
            seq_len=ctxt_seq_len,
            num_features=4,
            n_dim_selection=n_dim_selection,
            n_dim_model=n_dim_model if isinstance(n_dim_model, int) else n_dim_model[0],
            n_heads=n_enc_heads,
            n_layers=num_layers if isinstance(num_layers, int) else num_layers[0],
            n_layers_encoded=num_layers
            if isinstance(num_layers, int)
            else num_layers[2],
            dropout=dropout if isinstance(dropout, float) else dropout[0],
        )

        self.b_lstm = torch.nn.LSTM(
            2,
            n_dim_model if isinstance(n_dim_model, int) else n_dim_model[0],
            num_layers=num_layers if isinstance(num_layers, int) else num_layers[0],
            batch_first=True,
        )
        self.b_mlp = MLP(
            input_dim=n_dim_model if isinstance(n_dim_model, int) else n_dim_model[0],
            hidden_dim=n_dim_fc,
            output_dim=2,
        )

    @override
    def training_step(
        self,
        batch: EncoderDecoderTargetSample,  # type: ignore[override]
        batch_idx: int,
    ) -> dict[str, torch.Tensor]:
        try:
            target = batch["target"]
        except KeyError as e:
            msg = (
                "The batch must contain a target key. "
                "This is probably due to using a dataset without targets "
                "(e.g. test or predict)."
            )
            raise ValueError(msg) from e

        states = self.encoder(batch["encoder_input"])
        output = self(
            batch["decoder_input"][..., 2, None],
            hx=states["hx"],
            hx2=states["hx2"],
            hx3=states["hx3"],
        )
        b, _ = self.b_lstm(batch["decoder_input"][..., :2])
        b = self.b_mlp(b)
        output["b"] = b

        loss, losses = self.criterion(output, target, return_all=True)

        for optimizer in self.optimizers()[:2]:
            optimizer.zero_grad()

        self.manual_backward(loss)

        self.optimizers()[0].step()

        self.criterion.invert_gradients()
        self.optimizers()[1].step()

        loss_weights = {
            "loss_weight/alpha": self.criterion.alpha.item(),
            "loss_weight/beta": self.criterion.beta.item(),
            "loss_weight/gamma": self.criterion.gamma.item(),
            "loss_weight/kappa": self.criterion.kappa.item(),
            "loss_weight/eta": self.criterion.eta.item(),
        }

        losses = self.rename_losses_dict(losses)
        self.common_log_step(losses | loss_weights, "train")

        return losses | {"output": output}

    @override
    def validation_step(
        self,
        batch: EncoderDecoderTargetSample,  # type: ignore[override]
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> StepOutput:
        prev_hidden = typing.cast(BWState3 | None, self._val_hidden[dataloader_idx])

        if prev_hidden is None:
            prev_hidden = self.encoder(batch["encoder_input"])

        output, states = self(
            batch["decoder_input"][..., 2, None],
            hx=prev_hidden["hx"],
            hx2=prev_hidden["hx2"],
            hx3=prev_hidden["hx3"],
            return_states=True,
        )
        b = self.b_lstm(batch["decoder_input"][..., :2])
        b = self.b_mlp(b[0])
        output["b"] = b

        _, losses = self.criterion(output, batch["target"], return_all=True)

        # remove batch dimension
        if output["z"].shape[0] == 1:
            for key in output:
                output[key] = output[key].squeeze(0)  # type: ignore[literal-required]

        losses = self.rename_losses_dict(losses)
        self.common_log_step(losses, "validation")

        return typing.cast(StepOutput, losses | {"output": output, "state": states})
