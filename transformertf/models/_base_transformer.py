from __future__ import annotations

import sys
import typing

import einops
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
        return 1 - derivative * 0.9

    def calc_loss(
        self,
        model_output: torch.Tensor,
        batch: EncoderDecoderTargetSample,
    ) -> torch.Tensor:
        weights = batch.get("decoder_lengths", None)
        weights = 1.0 / weights if weights is not None else None

        # reshape to (bs, seq_len)
        weights = (
            einops.repeat(weights, "b 1 -> b t", t=model_output.size(1))
            if weights is not None
            else None
        )

        weights_dynamic = self._make_loss_weights(batch["target"].squeeze(-1))
        # expand to (bs, seq_len)

        weights = weights * weights_dynamic if weights is not None else weights_dynamic
        weights = weights.unsqueeze(-1) if weights is not None else None

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
