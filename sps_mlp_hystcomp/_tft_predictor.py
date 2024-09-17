from __future__ import annotations

import os
import sys
import typing

import numpy
import numpy as np
import pandas as pd
import torch
from hystcomp_utils.cycle_data import CycleData
from numpy import typing as npt
from transformertf.data import EncoderDecoderDataModule
from transformertf.data.dataset import EncoderDecoderPredictDataset
from transformertf.models.temporal_fusion_transformer import TemporalFusionTransformer
from transformertf.utils import ops

from ._base_predictor import (
    TARGET_COLNAME,
    TARGET_PAST_COLNAME,
    TIME_COLNAME,
    NoInitialStateError,
    Predictor,
)

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override


class TFTPredictor(Predictor):
    _module: TemporalFusionTransformer
    _datamodule: EncoderDecoderDataModule
    state: pd.DataFrame | None

    def __init__(
        self,
        device: typing.Literal["cpu", "cuda", "auto"] = "auto",
        *,
        compile: bool = False,  # noqa: A002
    ) -> None:
        super().__init__(device=device, compile=compile)

        self.state = None
        self._rdp_eps = 0.0

    @property
    def rdp_eps(self) -> float:
        return self._rdp_eps

    @rdp_eps.setter
    def rdp_eps(self, value: float) -> None:
        self._rdp_eps = value

    def set_rdp_eps(self, value: float) -> None:
        self.rdp_eps = value

    @override
    def _set_initial_state_impl(
        self,
        past_covariates: pd.DataFrame,
        past_targets: pd.DataFrame | np.ndarray | None = None,
    ) -> None:
        df = past_covariates
        if past_targets is not None:
            df = pd.concat([df, pd.DataFrame(past_targets)], axis=1)

        if "downsample" in self.hparams and self.hparams["downsample"] > 1:
            df = df.iloc[:: self.hparams["downsample"]].reset_index(drop=True)

        # keep ctxt_seq_len number of points for context
        self.state = self._keep_ctxt(df).reset_index(drop=True)

    @override
    def set_cycled_initial_state(
        self,
        cycles: list[CycleData],
        *args: typing.Any,
        use_programmed_current: bool = True,
        **kwargs: typing.Any,
    ) -> None:
        past_covariates = self.buffer_to_covariates(
            cycles,
            use_programmed_current=use_programmed_current,
            interpolate=self.rdp_eps == 0.0,
            rdp=self.rdp_eps,
            prog_t_phase=self._prog_t_phase,
        )

        self.set_initial_state(
            *args,
            past_covariates=past_covariates,
            **kwargs,
        )

    @override
    def _predict_impl(
        self,
        future_covariates: pd.DataFrame,
        *args: typing.Any,
        save_state: bool = True,
        **kwargs: typing.Any,
    ) -> npt.NDArray[np.float64]:
        if self.state is None:
            msg = "Initial state not set. Call set_initial_state() first."
            raise NoInitialStateError(msg)

        past_covariates = self.state.copy()
        past_targets = past_covariates.pop(TARGET_COLNAME).to_numpy().flatten()
        future_covariates[TIME_COLNAME] += past_covariates[TIME_COLNAME].iloc[-1]
        future_covariates = future_covariates.reset_index(drop=True)
        if "downsample" in self.hparams and self.hparams["downsample"] > 1:
            future_covariates = future_covariates.iloc[
                :: self.hparams["downsample"]
            ].reset_index(drop=True)

        dataset = EncoderDecoderPredictDataset(
            past_covariates=past_covariates,
            past_target=past_targets,
            future_covariates=future_covariates,
            context_length=self.hparams["ctxt_seq_len"],
            prediction_length=self.hparams["tgt_seq_len"],
            transforms=self._datamodule.transforms,
            input_columns=[TIME_COLNAME]
            + [col.name for col in self._datamodule.known_covariates],
            known_past_columns=[
                col.name for col in self._datamodule.known_past_covariates
            ],
            target_column=TARGET_COLNAME,
            apply_transforms=True,
            time_format=self.hparams["time_format"],
        )

        with torch.no_grad():
            predictions = self._predict_dataset(dataset, future_covariates)

        if save_state:
            future_covariates = future_covariates.copy()
            future_covariates[TARGET_PAST_COLNAME] = predictions
            future_covariates[TARGET_COLNAME] = predictions

            new_state = pd.concat([self.state, future_covariates], axis=0).reset_index(
                drop=True
            )
            self.state = self._keep_ctxt(new_state).reset_index(drop=True)

        return predictions

    def _predict_dataset(
        self, dataset: EncoderDecoderPredictDataset, future_covariates: pd.DataFrame
    ) -> npt.NDArray[np.float64]:
        dataloader = self._fabric.setup_dataloaders(
            torch.utils.data.DataLoader(dataset, batch_size=1)
        )

        input_slices = [
            slice(start, start + self.hparams["tgt_seq_len"])
            for start in range(0, len(future_covariates), self.hparams["tgt_seq_len"])
        ]

        self._module.on_predict_start()
        self._module.on_predict_epoch_start()

        outputs: list[npt.NDArray[np.float64]] = []
        for idx, batch in enumerate(dataloader):
            self._module.on_predict_batch_start(batch, idx)
            output = self._module.predict_step(batch, idx)
            output = ops.to_cpu(ops.detach(output))
            self._module.on_predict_batch_end(output, batch, idx)

            if idx < len(dataloader) - 1:  # only append if not the last batch
                dataset.append_past_target(
                    output["point_prediction"].squeeze().numpy(),
                    transform=False,
                )

                if self.hparams["target_depends_on"] is not None:
                    input_ = torch.tensor(
                        future_covariates.iloc[input_slices[idx]][
                            self.hparams["target_depends_on"]
                        ].to_numpy(),
                        dtype=torch.float32,
                    )
                    target_inverse = (
                        (
                            self._datamodule.target_transform.inverse_transform(
                                input_, output["point_prediction"].squeeze()
                            )
                        )
                        .squeeze()
                        .to(torch.float32)
                    )
                else:
                    target_inverse = (
                        (
                            self._datamodule.target_transform.inverse_transform(
                                output["point_prediction"].squeeze()
                            )
                        )
                        .squeeze()
                        .to(torch.float32)
                    )

                dataset.append_past_covariates(
                    pd.DataFrame({TARGET_PAST_COLNAME: target_inverse}),
                    transform=True,
                )

            outputs.append(output["point_prediction"].numpy().astype(np.float64))

        self._module.on_predict_epoch_end()
        self._module.on_predict_end()

        outputs_arr = np.concatenate([o.flatten() for o in outputs], axis=0)
        outputs_arr = outputs_arr[: len(future_covariates)]

        if self.hparams["target_depends_on"] is not None:
            return self._datamodule.target_transform.inverse_transform(
                future_covariates[self.hparams["target_depends_on"]].to_numpy(),
                outputs_arr,
            ).numpy()

        return self._datamodule.target_transform.inverse_transform(outputs_arr).numpy()

    @override
    def _predict_cycle_impl(
        self,
        cycle: CycleData,
        *args: typing.Any,
        save_state: bool = True,
        use_programmed_current: bool = True,
        **kwargs: typing.Any,
    ) -> npt.NDArray[np.float64]:
        future_covariates = self.buffer_to_covariates(
            [cycle],
            use_programmed_current=use_programmed_current,
            interpolate=self.rdp_eps == 0.0,
            rdp=self.rdp_eps,
            add_target=False,
            prog_t_phase=self._prog_t_phase,
        )

        prediction = self.predict(
            future_covariates,
            *args,
            save_state=save_state,
            **kwargs,
        )

        if "__time__" in future_covariates.columns:
            time = future_covariates["__time__"].to_numpy()
            time -= time[0]
            time = time[:: self.hparams["downsample"]]
        else:
            time = np.arange(0, cycle.num_samples) / 1e3
            time = time[:: self.hparams["downsample"]]

        time = np.round(time, 3)  # round to ms

        return numpy.vstack((time, prediction))

    @override
    def _load_checkpoint_impl(self, checkpoint_path: str | os.PathLike) -> None:
        self._module = TemporalFusionTransformer.load_from_checkpoint(checkpoint_path)
        self._datamodule = EncoderDecoderDataModule.load_from_checkpoint(
            checkpoint_path
        )

    def _keep_ctxt(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[-self.hparams["ctxt_seq_len"] :]
