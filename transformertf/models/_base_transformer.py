from __future__ import annotations

import typing

import torch

from ..config import TransformerBaseConfig
from ..data import EncoderDecoderTargetSample
from ..nn import QuantileLoss, get_loss
from ._base_module import LightningModuleBase


class TransformerModuleBase(LightningModuleBase):
    criterion: torch.nn.Module

    @classmethod
    def parse_config_kwargs(  # type: ignore[override]
        cls, config: TransformerBaseConfig, **kwargs: typing.Any
    ) -> dict[str, typing.Any]:
        default_kwargs = super().parse_config_kwargs(config, **kwargs)

        default_kwargs |= {
            "ctxt_seq_len": config.ctxt_seq_len,
            "tgt_seq_len": config.tgt_seq_len,
            "prediction_type": config.prediction_type,
            "loss_fn": config.loss_fn,
            "quantiles": config.quantiles,
        }

        default_kwargs |= kwargs

        if "criterion" not in default_kwargs:
            if "loss_fn" in default_kwargs and isinstance(
                default_kwargs["loss_fn"], str
            ):
                loss_kwargs = (
                    {"quantiles": default_kwargs["quantiles"]}
                    if default_kwargs["loss_fn"] == "quantile"
                    else {}
                )
                default_kwargs["criterion"] = get_loss(
                    default_kwargs["loss_fn"],  # type: ignore[arg-type]
                    **loss_kwargs,
                )
                if default_kwargs["criterion"] is not None and isinstance(
                    default_kwargs["criterion"], QuantileLoss
                ):
                    default_kwargs["output_dim"] = len(
                        default_kwargs["criterion"].quantiles
                    )
            else:
                default_kwargs["criterion"] = None
        default_kwargs.pop("loss_fn", None)
        default_kwargs.pop("quantiles", None)

        return default_kwargs

    def calc_loss(
        self,
        model_output: torch.Tensor,
        batch: EncoderDecoderTargetSample,
    ) -> torch.Tensor:
        weights = batch.get("decoder_lengths", None)
        weights = 1.0 / weights if weights is not None else None
        if self.hparams["prediction_type"] == "delta":
            with torch.no_grad():
                target = batch["target"].squeeze(-1)
                delta = torch.zeros_like(target)
                delta[:, 1:] = target[:, 1:] - target[:, :-1]

                past_target = batch["encoder_input"][:, -1, -1]
                delta[:, 0] = target[:, 0] - past_target

            return typing.cast(
                torch.Tensor, self.criterion(model_output, delta, weights=weights)
            )

        if self.hparams["prediction_type"] == "point":
            target = batch["target"].squeeze(-1)
            return typing.cast(
                torch.Tensor, self.criterion(model_output, target, weights=weights)
            )

        # This should never happen
        msg = f"Invalid prediction_type: {self.hparams['prediction_type']}"
        raise ValueError(msg)
