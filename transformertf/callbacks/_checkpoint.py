import typing

import lightning as L


def resolve_validation_metric(trainer: L.Trainer, *metric_name: str) -> float:
    for name in metric_name:
        if name in trainer.callback_metrics:
            return trainer.callback_metrics[name]
        elif name in trainer.logged_metrics:
            return trainer.logged_metrics[name]

    raise ValueError(
        f"Could not find metric {metric_name} in trainer metrics."
    )


class CheckpointEvery(L.pytorch.callbacks.checkpoint.Checkpoint):
    def __init__(
        self,
        checkpoint_every: int = 100,
        checkpoint_dir: str = "checkpoints",
        *args: typing.Any,
        **kwargs: typing.Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._checkpoint_every = checkpoint_every
        self._checkpoint_dir = checkpoint_dir

    def on_validation_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        super().on_validation_end(trainer, pl_module)

        if (trainer.current_epoch + 1) % self._checkpoint_every == 0:
            try:
                val_loss = resolve_validation_metric(
                    trainer,
                    "loss/validation",
                    "loss/validation/dataloader_idx_0",
                )
            except ValueError:
                val_loss = -1

            trainer.save_checkpoint(
                f"{self._checkpoint_dir}/"
                f"model_every_epoch={trainer.current_epoch + 1}_"
                f"val_loss={val_loss}.ckpt"
            )

    def state_dict(self) -> dict[str, typing.Any]:
        state_dict = super().state_dict()
        state_dict["checkpoint_every"] = self._checkpoint_every
        state_dict["checkpoint_dir"] = self._checkpoint_dir
        return state_dict

    def load_state_dict(self, state_dict: dict[str, typing.Any]) -> None:
        self._checkpoint_every = state_dict.pop("checkpoint_every")
        self._checkpoint_dir = state_dict.pop("checkpoint_dir")
        super().load_state_dict(state_dict)
