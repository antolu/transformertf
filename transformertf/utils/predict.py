from __future__ import annotations

import typing

import numpy as np
import pandas as pd
import torch

from ..data import (
    EncoderDecoderDataModule,
    TimeSeriesDataModule,
)
from ..nn import QuantileLoss
from ..data.dataset import EncoderDecoderPredictDataset
from ..models import LightningModuleBase
from ..models.phylstm import PhyLSTMModule
from ..utils import ops


def predict_timeseries(
    module: LightningModuleBase,
    datamodule: TimeSeriesDataModule,
    past_covariates: pd.DataFrame,
    future_covariates: pd.DataFrame,
    device: torch.device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    ),
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
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, num_workers=0
    )

    outputs = []
    for idx, batch in enumerate(dataloader):
        batch = ops.to(batch, device)

        model_output = module(batch)
        model_output = ops.to_cpu(model_output)
        model_output = ops.detach(model_output)

        outputs.append(model_output)

    outputs = torch.cat([o.squeeze(0) for o in outputs], dim=0)

    if datamodule.target_transform is not None:
        outputs = datamodule.target_transform.inverse_transform(
            covariates[datamodule.hparams["input_columns"][0]].to_numpy(),
            outputs,
        )

    # truncate the outputs to the length of the future covariates
    outputs = outputs[: dataset.num_points]
    outputs = outputs[
        len(past_covariates) // datamodule.hparams["downsample"] :
    ]
    return typing.cast(torch.Tensor, outputs).squeeze().numpy()


def predict_phylstm(
    module: PhyLSTMModule,
    datamodule: TimeSeriesDataModule,
    past_covariates: pd.DataFrame,
    future_covariates: pd.DataFrame,
    device: torch.device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    ),
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
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, num_workers=0
    )

    outputs = []
    inputs = []

    module.on_predict_start()
    module.on_predict_epoch_start()
    for idx, batch in enumerate(dataloader):
        batch = ops.to(batch, device)

        model_output = module.predict_step(batch, idx)
        module.on_predict_batch_end(model_output, batch, idx)

        model_output = ops.to_cpu(model_output)
        model_output = ops.detach(model_output)

        inputs.append(batch["input"])
        outputs.append(model_output)

    module.on_predict_epoch_end()
    module.on_predict_end()

    predictions = torch.cat(
        [o["output"]["z"].squeeze(0) for o in outputs], dim=0
    )
    predictions = predictions[..., 0]  # get B

    inputs = torch.cat([i.squeeze(0) for i in inputs], dim=0).squeeze()

    if datamodule.target_transform is not None:
        input_transform = datamodule.input_transforms[
            datamodule.hparams["input_columns"][0]
        ]

        inputs = input_transform.inverse_transform(inputs)

        predictions = datamodule.target_transform.inverse_transform(
            inputs,
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
    past_target: pd.DataFrame,
    raw_output: typing.Literal[False] = False,
    device: torch.device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    ),
) -> np.ndarray: ...


@typing.overload
def predict_encoder_decoder(
    module: LightningModuleBase,
    datamodule: EncoderDecoderDataModule,
    past_covariates: pd.DataFrame,
    future_covariates: pd.DataFrame,
    past_target: np.ndarray,
    raw_output: typing.Literal[True],
    device: torch.device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    ),
) -> torch.Tensor: ...


def predict_encoder_decoder(
    module: LightningModuleBase,
    datamodule: EncoderDecoderDataModule,
    past_covariates: pd.DataFrame,
    future_covariates: pd.DataFrame,
    past_target: pd.DataFrame | np.ndarray | pd.Series,
    raw_output: bool = False,
    device: torch.device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    ),
) -> np.ndarray | torch.Tensor:
    if isinstance(past_target, pd.DataFrame):
        past_target = past_target[
            datamodule.hparams["target_column"]
        ].to_numpy()
    elif isinstance(past_target, pd.Series):
        past_target = past_target.to_numpy()

    past_df = pd.concat(
        [past_covariates, pd.DataFrame({"target": past_target})], axis=1
    ).reset_index(drop=True)
    past_df = datamodule.preprocess_dataframe(past_df)

    past_covariates = past_df[past_covariates.columns]
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
        input_columns=datamodule.hparams["input_columns"],
        target_column=datamodule.hparams["target_column"],
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, num_workers=0
    )

    outputs = []
    for idx, batch in enumerate(dataloader):
        batch = ops.to(batch, device)

        model_output = module(batch)
        model_output = ops.to_cpu(model_output)
        model_output = ops.detach(model_output)

        outputs.append(model_output)
        point_prediction = to_point_prediction(
            model_output, module.criterion
        ).squeeze()
        dataset.append_past_target(point_prediction)

    outputs = torch.cat([o.squeeze(0) for o in outputs], dim=0)

    if raw_output:
        return outputs

    outputs = to_point_prediction(outputs, module.criterion)

    if datamodule.target_transform is not None:
        future_x = future_covariates[
            datamodule.hparams["input_columns"][0]
        ].to_numpy()
        outputs = datamodule.target_transform.inverse_transform(
            future_x, outputs
        )

    return outputs


def to_point_prediction(
    model_output: torch.Tensor, criterion: torch.nn.Module
) -> torch.Tensor:
    if model_output.ndim < 2:
        raise ValueError(
            f"Model output must have at least 2 dimensions, got {model_output.ndim}"
        )
    if isinstance(criterion, QuantileLoss):
        if model_output.ndim == 2:
            return criterion.point_prediction(
                model_output.unsqueeze(0)
            ).squeeze(0)
        return criterion.point_prediction(model_output)

    return model_output
