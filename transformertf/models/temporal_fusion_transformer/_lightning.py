from __future__ import annotations

import typing

import torch

from ...data import EncoderDecoderSample, EncoderDecoderTargetSample
from ...nn import QuantileLoss
from .._base_transformer import TransformerModuleBase
from ._model import TemporalFusionTransformerModel


class TemporalFusionTransformer(TransformerModuleBase):
    def __init__(
        self,
        num_past_features: int,
        num_future_features: int,
        ctxt_seq_len: int,
        tgt_seq_len: int,
        n_dim_model: int = 300,
        hidden_continuous_dim: int = 8,
        num_heads: int = 4,
        num_lstm_layers: int = 2,
        dropout: float = 0.1,
        output_dim: int = 7,
        criterion: QuantileLoss | torch.nn.Module | None = None,
        *,
        prediction_type: typing.Literal["delta", "point"] = "point",
        log_grad_norm: bool = False,
        compile_model: bool = False,
        trainable_parameters: list[str] | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["lr_scheduler", "criterion"])

        if criterion is None:
            criterion = QuantileLoss()
            self.hparams["output_dim"] = len(criterion.quantiles)
            output_dim = self.hparams["output_dim"]
        self.criterion = criterion

        self.model = TemporalFusionTransformerModel(
            num_past_features=num_past_features,
            num_future_features=num_future_features,
            ctxt_seq_len=ctxt_seq_len,
            tgt_seq_len=tgt_seq_len,
            n_dim_model=n_dim_model,
            num_static_features=1,
            hidden_continuous_dim=hidden_continuous_dim,
            num_heads=num_heads,
            num_lstm_layers=num_lstm_layers,
            dropout=dropout,
            output_dim=output_dim,
        )

    def forward(self, x: EncoderDecoderTargetSample) -> torch.Tensor:
        return self.model(
            past_covariates=x["encoder_input"],
            future_covariates=x["decoder_input"][
                ...,
                -self.hparams["num_future_features"] :,
            ],
            static_covariates=x["encoder_lengths"],  # type: ignore[typeddict-item]
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
            "output": model_output["output"],
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
            "output": model_output["output"],
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
        if self.hparams["trainable_parameters"] is None:
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
