from __future__ import annotations

import sys
import typing

import torch

from ..data import EncoderDecoderSample, EncoderDecoderTargetSample
from ..nn import QuantileLoss
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

    def training_step(
        self, batch: EncoderDecoderTargetSample, batch_idx: int
    ) -> dict[str, torch.Tensor]:
        model_output = self(batch)

        loss = self.calc_loss(model_output["output"], batch)

        loss_dict = {"loss": loss}
        point_prediction = model_output["output"]
        if isinstance(self.criterion, QuantileLoss) or (
            hasattr(self.criterion, "_orig_mod")
            and isinstance(self.criterion._orig_mod, QuantileLoss)  # noqa: SLF001
        ):
            point_prediction = self.criterion.point_prediction(point_prediction)

        self.common_log_step(loss_dict, "train")

        return {
            **loss_dict,
            "output": model_output.pop("output"),
            **{k: v.detach() for k, v in model_output.items()},
            "point_prediction": point_prediction,
        }

    def validation_step(
        self,
        batch: EncoderDecoderTargetSample,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> dict[str, torch.Tensor]:
        model_output = self(batch)

        loss = self.calc_loss(model_output["output"], batch)

        loss_dict = {"loss": loss}
        point_prediction = model_output["output"]
        if isinstance(self.criterion, QuantileLoss) or (
            hasattr(self.criterion, "_orig_mod")
            and isinstance(self.criterion._orig_mod, QuantileLoss)  # noqa: SLF001
        ):
            point_prediction = self.criterion.point_prediction(point_prediction)

        self.common_log_step(loss_dict, "validation")

        return {
            **loss_dict,
            "output": model_output.pop("output"),
            **{k: v.detach() for k, v in model_output.items()},
            "point_prediction": point_prediction,
        }

    def predict_step(
        self, batch: EncoderDecoderSample, batch_idx: int
    ) -> dict[str, torch.Tensor]:
        model_output = self(batch)

        point_prediction = model_output["output"]
        if isinstance(self.criterion, QuantileLoss) or (
            hasattr(self.criterion, "_orig_mod")
            and isinstance(self.criterion._orig_mod, QuantileLoss)  # noqa: SLF001
        ):
            point_prediction = self.criterion.point_prediction(model_output["output"])

        model_output["point_prediction"] = point_prediction
        return model_output

    def on_train_epoch_start(self) -> None:
        """
        Set normalizing layers not in trainable_parameters to eval mode.
        """
        if (
            "trainable_parameters" not in self.hparams
            or self.hparams["trainable_parameters"] is None
        ):
            return

        trainable_params = set(self.hparams["trainable_parameters"])
        for name, module in self.named_modules():
            if not isinstance(module, torch.nn.LayerNorm):
                continue

            param_name = name.split(".")[1]  # model.[name].xxx
            if param_name not in trainable_params:
                module.eval()

    def parameters(self, recurse: bool = True) -> typing.Iterator[torch.nn.Parameter]:  # noqa: FBT001, FBT002
        """
        Override the parameters method to only return the trainable parameters, for
        use with LightningCLI where we cannot easily specify the trainable parameters.

        Parameters
        ----------
        recurse : bool, optional
            Whether to return parameters of this module and all submodules,
            by default True

        Returns
        -------
        Iterator[torch.nn.Parameter]
        """
        if self.hparams["trainable_parameters"] is None:
            yield from super().parameters(recurse=recurse)
            return

        trainable_params = set(self.hparams["trainable_parameters"])
        for name, param in self.named_parameters(recurse=recurse):
            param_name = name.split(".")[1]  # model.[name].xxx
            if param_name in trainable_params:
                yield param

    def calc_loss(
        self,
        model_output: torch.Tensor,
        batch: EncoderDecoderTargetSample,
    ) -> torch.Tensor:
        weights = None

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


def get_attention_mask(
    encoder_lengths: torch.LongTensor,
    decoder_lengths: torch.LongTensor,
    max_encoder_length: int,
    max_decoder_length: int,
    *,
    causal_attention: bool = True,
) -> torch.Tensor:
    """
    Returns causal mask to apply for self-attention layer.
    """
    if causal_attention:
        # indices to which is attended
        attend_step = torch.arange(max_decoder_length, device=encoder_lengths.device)
        # indices for which is predicted
        predict_step = torch.arange(
            0, max_decoder_length, device=encoder_lengths.device
        )[:, None]
        # do not attend to steps to self or after prediction
        decoder_mask = (
            (attend_step >= predict_step)
            .unsqueeze(0)
            .expand(encoder_lengths.size(0), -1, -1)
        )
    else:
        # there is value in attending to future forecasts if
        # they are made with knowledge currently available
        #   one possibility is here to use a second attention layer
        # for future attention
        # (assuming different effects matter in the future than the past)
        #  or alternatively using the same layer but
        # allowing forward attention - i.e. only
        #  masking out non-available data and self
        decoder_mask = (
            create_mask(max_decoder_length, decoder_lengths)
            .unsqueeze(1)
            .expand(-1, max_decoder_length, -1)
        )
    # do not attend to steps where data is padded
    encoder_mask = (
        create_mask(max_encoder_length, encoder_lengths)
        .unsqueeze(1)
        .expand(-1, max_decoder_length, -1)
    )
    # combine masks along attended time - first encoder and then decoder
    return torch.cat(
        (
            encoder_mask,
            decoder_mask,
        ),
        dim=2,
    )


def create_mask(
    size: int, lengths: torch.LongTensor, *, inverse: bool = False
) -> torch.BoolTensor:
    """
    Create boolean masks of shape len(lenghts) x size.

    An entry at (i, j) is True if lengths[i] > j.

    Args:
        size (int): size of second dimension
        lengths (torch.LongTensor): tensor of lengths
        inverse (bool, optional): If true, boolean mask is inverted. Defaults to False.

    Returns:
        torch.BoolTensor: mask
    """

    if inverse:  # return where values are
        return torch.arange(size, device=lengths.device).unsqueeze(
            0
        ) < lengths.unsqueeze(-1)
    # return where no values are
    return torch.arange(size, device=lengths.device).unsqueeze(0) >= lengths.unsqueeze(
        -1
    )
