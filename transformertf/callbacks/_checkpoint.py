import typing
import lightning as L


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

        if trainer.current_epoch % self._checkpoint_every == 0:
            metric_name = "loss/validation"

            if metric_name in trainer.callback_metrics:
                val_loss = trainer.callback_metrics[metric_name]
            elif metric_name in trainer.logged_metrics:
                val_loss = trainer.logged_metrics[metric_name]
            else:
                val_loss = -1

            trainer.save_checkpoint(
                f"{self._checkpoint_dir}/model_every_epoch={trainer.current_epoch}_val_loss={val_loss}.pt"
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
