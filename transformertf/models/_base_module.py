from __future__ import annotations

import collections
import logging
import typing

import lightning as L
import lightning.pytorch.utilities
import torch

from ..data import EncoderDecoderTargetSample, TimeSeriesSample
from ..nn import functional as F
from ..utils import ops

if typing.TYPE_CHECKING:
    SameType = typing.TypeVar("SameType", bound="LightningModuleBase")
    from .typing import MODEL_OUTPUT, STEP_OUTPUT


log = logging.getLogger(__name__)


class LightningModuleBase(L.LightningModule):
    model: torch.nn.Module
    criterion: torch.nn.Module

    def __init__(self) -> None:
        super().__init__()
        self.save_hyperparameters()

        self._train_outputs: dict[int, list[MODEL_OUTPUT]] = {}
        self._val_outputs: dict[int, list[MODEL_OUTPUT]] = {}
        self._test_outputs: dict[int, list[MODEL_OUTPUT]] = {}
        self._inference_outputs: dict[int, list[MODEL_OUTPUT]] = {}

    def on_fit_start(self) -> None:
        self.maybe_compile_model()

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
                setattr(self, name, torch.compile(mod))

    def on_train_start(self) -> None:
        self._train_outputs = {}

    def on_validation_epoch_start(self) -> None:
        self._val_outputs = {}

        super().on_validation_epoch_start()

    def on_train_batch_end(
        self,
        outputs: torch.Tensor | typing.Mapping[str, typing.Any] | None,
        batch: TimeSeriesSample | EncoderDecoderTargetSample,
        batch_idx: int,
    ) -> None:
        assert outputs is not None
        assert "target" in batch
        assert isinstance(outputs, dict)
        other_metrics = self.calc_other_metrics(outputs, batch["target"])
        self.common_log_step(other_metrics, "train")  # type: ignore[attr-defined]

        return super().on_train_batch_end(outputs, batch, batch_idx)  # type: ignore[misc]

    def on_validation_batch_end(
        self,
        outputs: STEP_OUTPUT,  # type: ignore[override]
        batch: TimeSeriesSample,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if dataloader_idx not in self._val_outputs:
            self._val_outputs[dataloader_idx] = []
        self._val_outputs[dataloader_idx].append(ops.to_cpu(ops.detach(outputs)))  # type: ignore[arg-type,type-var]

        assert "target" in batch
        other_metrics = self.calc_other_metrics(outputs, batch["target"])
        self.common_log_step(other_metrics, "validation")  # type: ignore[attr-defined]

    def on_test_epoch_start(self) -> None:
        self._test_outputs = {}

        super().on_test_epoch_start()

    def on_test_batch_end(
        self,
        outputs: STEP_OUTPUT,  # type: ignore[override]
        batch: TimeSeriesSample,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if dataloader_idx not in self._test_outputs:
            self._test_outputs[dataloader_idx] = []
        self._test_outputs[dataloader_idx].append(ops.to_cpu(ops.detach(outputs)))  # type: ignore[type-var, arg-type]

    def on_predict_epoch_start(self) -> None:
        self._inference_outputs = {}

        super().on_predict_epoch_start()

    def on_predict_batch_end(
        self,
        outputs: STEP_OUTPUT,  # type: ignore[override]
        batch: torch.Tensor,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if dataloader_idx not in self._inference_outputs:
            self._inference_outputs[dataloader_idx] = []
        self._inference_outputs[dataloader_idx].append(ops.to_cpu(ops.detach(outputs)))  # type: ignore[type-var, arg-type]

    def common_log_step(
        self,
        loss: dict[str, torch.Tensor],
        stage: typing.Literal["train", "validation", "test", "predict"],
    ) -> dict[str, torch.Tensor]:
        log_dict = {k + f"/{stage}": v for k, v in loss.items()}

        if self.logger is not None:
            self.log_dict(
                log_dict,
                on_step=stage == "train",
                prog_bar=stage == "train",
                sync_dist=True,
            )

        return log_dict

    def calc_other_metrics(
        self,
        outputs: STEP_OUTPUT,
        target: torch.Tensor,  # type: ignore[override]
    ) -> dict[str, torch.Tensor]:
        """
        Calculate other metrics from the outputs of the model.

        Parameters
        ----------
        outputs : STEP_OUTPUT
            The outputs of the model. Should contain keys "output" and "loss", and optionally "point_prediction".
            If the "point_prediction" key is present, it should be a tensor of shape (batch_size, prediction_horizon, 1),
            otherwise it will be calculated from the "output" tensor.
        target : torch.Tensor
            The target tensor. Should be a tensor of shape (batch_size, prediction_horizon, 1).

        Returns
        -------
        dict[str, torch.Tensor]
            A dictionary containing the calculated metrics.
        """
        loss_dict: dict[str, torch.Tensor] = {}

        assert isinstance(outputs, dict)
        if "point_prediction" in outputs:
            prediction = outputs["point_prediction"]
        else:
            prediction = outputs["output"]

        assert isinstance(prediction, torch.Tensor)

        target = target.squeeze()
        prediction = prediction.squeeze()

        loss_dict["MSE"] = torch.nn.functional.mse_loss(prediction, target)
        loss_dict["MAE"] = torch.nn.functional.l1_loss(prediction, target)
        loss_dict["MAPE"] = F.mape_loss(prediction, target)
        loss_dict["SMAPE"] = F.smape_loss(prediction, target)
        loss_dict["RMSE"] = torch.sqrt(loss_dict["MSE"])

        return loss_dict

    def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        if self.hparams.get("log_grad_norm") and self.global_rank == 0:
            self.log_dict(lightning.pytorch.utilities.grad_norm(self, norm_type=2))

    @property
    def validation_outputs(self) -> dict[int, list[MODEL_OUTPUT]]:
        """
        Returns the validation outputs, which are stored in a dictionary where the keys are the dataloader indices and
        the values are lists of model outputs at each validation step.

        Returns
        -------
        dict[int, list[MODEL_OUTPUT]]
            Validation outputs, where the keys are the dataloader indices and the values are lists of model outputs.

        """
        return self._val_outputs

    @property
    def test_outputs(self) -> dict[int, list[MODEL_OUTPUT]]:
        """
        Returns the test outputs, which are stored in a dictionary where the keys are the dataloader indices and
        the values are lists of model outputs at each test step.

        Returns
        -------
        dict[int, list[MODEL_OUTPUT]]
            Test outputs, where the keys are the dataloader indices and the values are lists of model outputs.
        """
        return self._test_outputs

    @property
    def inference_outputs(self) -> dict[int, list[MODEL_OUTPUT]]:
        """
        Returns the inference outputs, which are stored in a dictionary where the keys are the dataloader indices and
        the values are lists of model outputs at each inference step.

        Returns
        -------
        dict[int, list[MODEL_OUTPUT]]
            Inference outputs, where the keys are the dataloader indices and the values are lists of model outputs.
        """
        return self._inference_outputs

    def state_dict(
        self,
        *args: typing.Any,
        destination: dict[str, torch.Tensor] | None = None,
        prefix: str = "",
        keep_vars: bool = False,
    ) -> dict[str, torch.Tensor]:
        state_dict = super().state_dict(
            *args, destination=destination, prefix=prefix, keep_vars=keep_vars
        )

        # hack to save the original model state dict and not the compiled one
        # this assumes that internally the model is stored in the `model` attribute
        # and that the model is not compiled when the LightningModule is instantiated

        # keys are xxx._orig_mod.xxx, remove _orig_mod
        if self.hparams.get("compile_model"):
            odict = collections.OrderedDict()
            for k in list(state_dict.keys()):
                if "_orig_mod" in k:
                    new_key = k.replace("_orig_mod.", "")
                    odict[new_key] = state_dict[k]
                else:
                    odict[k] = state_dict[k]

            state_dict = odict

        return state_dict
