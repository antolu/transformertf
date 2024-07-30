from __future__ import annotations

import typing

import numpy as np
import pandas as pd
import torch

from ..data import DataModuleBase, EncoderDecoderDataModule, TimeSeriesDataModule
from ..data.dataset import EncoderDecoderPredictDataset
from ..models import LightningModuleBase
from ..models.bwlstm import BWLSTM3
from ..nn import QuantileLoss
from ..utils import ops

__all__ = [
    "predict",
    "predict_encoder_decoder",
    "predict_phylstm",
    "predict_timeseries",
]

DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict(
    module: LightningModuleBase,
    datamodule: DataModuleBase,
    past_covariates: pd.DataFrame,
    future_covariates: pd.DataFrame,
    past_target: pd.DataFrame | np.ndarray | pd.Series | None = None,
    device: torch.device = DEFAULT_DEVICE,
) -> np.ndarray:
    """
    Predict the target variable using the given module and datamodule.

    Parameters
    ----------
    module
    datamodule
    past_covariates
    future_covariates
    past_target
    device

    Returns
    -------

    """
    if isinstance(datamodule, TimeSeriesDataModule) and isinstance(module, BWLSTM3):
        return predict_phylstm(
            module,
            datamodule,
            past_covariates,
            future_covariates,
            device,
        )
    if isinstance(datamodule, TimeSeriesDataModule):
        return predict_timeseries(
            module,
            datamodule,
            past_covariates,
            future_covariates,
            device,
        )
    if isinstance(datamodule, EncoderDecoderDataModule):
        assert past_target is not None, "past_target must be provided"
        return predict_encoder_decoder(
            module,
            datamodule,
            past_covariates,
            future_covariates,
            past_target,  # type: ignore
            device=device,
        )
    msg = f"Predicting with datamodule of type {type(datamodule)} is not implemented"
    raise NotImplementedError(msg)


def predict_timeseries(
    module: LightningModuleBase,
    datamodule: TimeSeriesDataModule,
    past_covariates: pd.DataFrame,
    future_covariates: pd.DataFrame,
    device: torch.device = DEFAULT_DEVICE,
) -> np.ndarray:
    """
    UNTESTED

    Parameters
    ----------
    module
    datamodule
    past_covariates
    future_covariates
    device

    Returns
    -------

    """
    covariates = pd.concat((past_covariates, future_covariates))

    dataset = datamodule.make_dataset(covariates, predict=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0)

    outputs = []
    for _idx, batch in enumerate(dataloader):
        batch = ops.to(batch, device)

        model_output = module(batch)
        model_output = ops.to_cpu(model_output)
        model_output = ops.detach(model_output)

        outputs.append(model_output)

    outputs_t = torch.cat([o.squeeze(0) for o in outputs], dim=0)

    if datamodule.target_transform is not None:
        outputs_t = datamodule.target_transform.inverse_transform(
            covariates[datamodule.hparams["input_columns"][0]].to_numpy(),
            outputs_t,
        )

    # truncate the outputs to the length of the future covariates
    outputs_t = outputs_t[: dataset.num_points]
    outputs_t = outputs_t[len(past_covariates) // datamodule.hparams["downsample"] :]
    return typing.cast(torch.Tensor, outputs_t).squeeze().numpy()


def predict_phylstm(
    module: BWLSTM3,
    datamodule: TimeSeriesDataModule,
    past_covariates: pd.DataFrame,
    future_covariates: pd.DataFrame,
    device: torch.device = DEFAULT_DEVICE,
) -> np.ndarray:
    """

    Parameters
    ----------
    module
    datamodule
    past_covariates
    future_covariates
    device

    Returns
    -------

    """
    covariates = pd.concat((past_covariates, future_covariates))

    dataset = datamodule.make_dataset(covariates, predict=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0)

    outputs = []
    inputs = []

    module.on_predict_start()
    module.on_predict_epoch_start()
    for idx, batch in enumerate(dataloader):
        batch = ops.to(batch, device)

        model_output = module.predict_step(batch, idx)
        module.on_predict_batch_end(model_output, batch, idx)  # type: ignore[arg-type]

        model_output = ops.to_cpu(model_output)  # type: ignore[type-var]
        model_output = ops.detach(model_output)  # type: ignore[type-var]

        inputs.append(batch["input"])
        outputs.append(model_output)

    module.on_predict_epoch_end()
    module.on_predict_end()

    predictions = torch.cat([o["output"]["z"].squeeze(0) for o in outputs], dim=0)
    predictions = predictions[..., 0]  # get B

    inputs_t = torch.cat([i.squeeze(0) for i in inputs], dim=0).squeeze()

    if datamodule.target_transform is not None:
        input_transform = datamodule.input_transforms[
            datamodule.hparams["known_covariates"][0]
        ]

        inputs_t = input_transform.inverse_transform(inputs_t)

        predictions = datamodule.target_transform.inverse_transform(
            inputs_t,
            predictions,
        )

    # truncate the outputs to the length of the future covariates
    predictions = predictions[: dataset.num_points]
    predictions = predictions[
        len(past_covariates) // datamodule.hparams["downsample"] :
    ]
    return typing.cast(torch.Tensor, predictions).squeeze().numpy()


@typing.overload
def predict_encoder_decoder(
    module: LightningModuleBase,
    datamodule: EncoderDecoderDataModule,
    past_covariates: pd.DataFrame,
    future_covariates: pd.DataFrame,
    past_target: pd.DataFrame | np.ndarray | pd.Series,
    device: torch.device = DEFAULT_DEVICE,
    *,
    raw_output: typing.Literal[False] = False,
) -> np.ndarray: ...


@typing.overload
def predict_encoder_decoder(
    module: LightningModuleBase,
    datamodule: EncoderDecoderDataModule,
    past_covariates: pd.DataFrame,
    future_covariates: pd.DataFrame,
    past_target: pd.DataFrame | np.ndarray | pd.Series,
    device: torch.device = DEFAULT_DEVICE,
    *,
    raw_output: typing.Literal[True],
) -> torch.Tensor: ...


def predict_encoder_decoder(
    module: LightningModuleBase,
    datamodule: EncoderDecoderDataModule,
    past_covariates: pd.DataFrame,
    future_covariates: pd.DataFrame,
    past_target: pd.DataFrame | np.ndarray | pd.Series,
    device: torch.device = DEFAULT_DEVICE,
    *,
    raw_output: bool = False,
) -> np.ndarray | torch.Tensor:
    if isinstance(past_target, pd.DataFrame):
        past_target = past_target[datamodule.hparams["target_covariate"]].to_numpy()
    elif isinstance(past_target, pd.Series):
        past_target = past_target.to_numpy()

    known_past_columns = datamodule.hparams.get("known_past_covariates")

    past_covariates["target"] = past_target
    past_df = datamodule.preprocess_dataframe(past_covariates)

    past_covariates = past_df[past_covariates.columns]
    (past_df[known_past_columns] if known_past_columns is not None else None)
    past_target = past_df["target"].to_numpy()
    future_covariates = datamodule.preprocess_dataframe(future_covariates)

    dataset = EncoderDecoderPredictDataset(
        past_covariates=past_covariates,
        future_covariates=future_covariates,
        past_target=past_target,
        context_length=datamodule.hparams["ctxt_seq_len"],
        prediction_length=datamodule.hparams["tgt_seq_len"],
        input_transforms=datamodule.input_transforms,
        target_transform=datamodule.target_transform,
        input_columns=datamodule.hparams["known_covariates"],
        target_column=datamodule.hparams["target_covariate"],
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0)

    outputs = []
    for idx, batch in enumerate(dataloader):
        batch["encoder_lengths"] = torch.ones(
            [batch["encoder_input"].shape[0], 1],
            dtype=batch["encoder_input"].dtype,
        )
        batch = ops.to(batch, device)

        model_output = module(batch)
        if isinstance(model_output, dict) and "output" in model_output:
            model_output = model_output["output"]
        model_output = ops.to_cpu(model_output)
        model_output = ops.detach(model_output)

        outputs.append(model_output)

        if idx < len(dataloader) - 1:
            point_prediction = to_point_prediction(
                model_output, module.criterion
            ).squeeze()
            dataset.append_past_target(point_prediction)

    outputs_t = torch.cat([o.squeeze(0) for o in outputs], dim=0)

    if raw_output:
        return outputs_t.numpy()  # type: ignore[attr-defined]

    outputs_t = to_point_prediction(outputs_t, module.criterion)

    # truncate the outputs to the length of the future covariates
    outputs_t = outputs_t[: len(future_covariates)]

    if datamodule.target_transform is not None:
        future_x = future_covariates[datamodule.hparams["known_covariates"][0]]
        outputs_t = datamodule.target_transform.inverse_transform(future_x, outputs_t)

    # truncate the outputs to the length of the future covariates
    return outputs_t[: len(future_covariates)].numpy()  # type: ignore[attr-defined]


def to_point_prediction(
    model_output: torch.Tensor, criterion: torch.nn.Module
) -> torch.Tensor:
    if model_output.ndim < 2:
        msg = f"Model output must have at least 2 dimensions, got {model_output.ndim}"
        raise ValueError(msg)
    if isinstance(criterion, QuantileLoss):
        if model_output.ndim == 2:
            return criterion.point_prediction(model_output.unsqueeze(0)).squeeze(0)
        return criterion.point_prediction(model_output)

    return model_output
