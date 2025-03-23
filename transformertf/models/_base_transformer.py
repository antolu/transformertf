from __future__ import annotations

import sys
import typing

import torch

from ..data import EncoderDecoderTargetSample
from ._base_module import LightningModuleBase

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override


class TransformerModuleBase(LightningModuleBase):
    criterion: torch.nn.Module

    @override
    def maybe_compile_model(self) -> None:
        """
        Compile the model if the "compile_model" key is present in the hyperparameters
        and is set to True. This is up to the subclass to implement. This also
        requires the model to be set to the "model" attribute.
        """
        if self.hparams.get("compile_model"):
            for name, mod in self.named_children():
                if "loss" in name.lower():
                    continue
                setattr(self, name, torch.compile(mod, dynamic=True))

    def on_train_batch_start(
        self, batch: EncoderDecoderTargetSample, batch_idx: int
    ) -> None:
        self._maybe_mark_dynamic(batch)

        super().on_train_batch_start(batch=batch, batch_idx=batch_idx)

    def on_validation_batch_start(
        self, batch: EncoderDecoderTargetSample, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        self._maybe_mark_dynamic(batch)

        super().on_validation_batch_start(
            batch=batch,
            batch_idx=batch_idx,
            dataloader_idx=dataloader_idx,
        )

    def _make_loss_weights(self, target: torch.Tensor) -> torch.Tensor:
        """
        Create loss weights for the quantile loss.
        The weight should be higher when the derivative of the target is close to zero.

        Parameters
        ----------
        target : torch.Tensor
            Target tensor

        Returns
        -------
        torch.Tensor
            Loss weights
        """
        derivative = torch.abs(torch.gradient(target, dim=1)[0])

        # normalize derivative
        derivative = derivative / derivative.max()

        # weight is 1 - derivative * 0.9
        return 1 - derivative * 0.8

    def calc_loss(
        self,
        model_output: torch.Tensor,
        batch: EncoderDecoderTargetSample,
    ) -> torch.Tensor:
        # weights = batch.get("decoder_lengths", None)
        # weights = 1.0 / weights if weights is not None else None
        #
        # # reshape to (bs, seq_len)
        # weights = (
        #     einops.repeat(weights, "b 1 -> b t", t=model_output.size(1))
        #     if weights is not None
        #     else None
        # )

        weights_dynamic = self._make_loss_weights(batch["target"].squeeze(-1))
        # expand to (bs, seq_len)

        # weights = weights * weights_dynamic if weights is not None else weights_dynamic
        # weights = weights.unsqueeze(-1) if weights is not None else None
        weights = weights_dynamic.unsqueeze(-1)
        if torch.any(torch.isnan(model_output)):
            msg = f"Model output contains {torch.sum(torch.isnan(model_output))} NaN values."
            raise ValueError(msg)

        if torch.any(torch.isinf(model_output)):
            msg = f"Model output contains {torch.sum(torch.isinf(model_output))} Inf values."
            raise ValueError(msg)

        if (
            "prediction_type" in self.hparams
            and self.hparams["prediction_type"] == "delta"
        ):
            with torch.no_grad():
                target = batch["target"].squeeze(-1)
                delta = torch.zeros_like(target)
                delta[:, 1:] = target[:, 1:] - target[:, :-1]

                past_target = batch["encoder_input"][:, -1, -1]
                delta[:, 0] = target[:, 0] - past_target

            return typing.cast(
                torch.Tensor, self.criterion(model_output, delta, weights=weights)
            )

        if (
            "prediction_type" not in self.hparams
            or self.hparams["prediction_type"] == "point"
        ):
            target = batch["target"].squeeze(-1)
            return typing.cast(
                torch.Tensor, self.criterion(model_output, target, weights=weights)
            )

        # This should never happen
        msg = f"Invalid prediction_type: {self.hparams['prediction_type']}"
        raise ValueError(msg)

    def _maybe_mark_dynamic(
        self, batch: EncoderDecoderTargetSample
    ) -> EncoderDecoderTargetSample:
        """
        Mark the input tensors as dynamic so that the torch compiler can optimize the
        computation graph, even when input shapes are changing.

        Needs PyTorch 2.7.0 to work when distributed training (DDP) is enabled.
        https://github.com/pytorch/pytorch/issues/140229
        """
        if not self.hparams["compile_model"]:
            return batch

        torch._dynamo.mark_dynamic(batch["encoder_input"], index=1)  # noqa: SLF001
        torch._dynamo.mark_dynamic(batch["decoder_input"], index=1)  # noqa: SLF001
        if "target" in batch:
            torch._dynamo.mark_dynamic(batch["target"], index=1)  # noqa: SLF001

        return batch
