from __future__ import annotations

import logging
import os
import typing

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from hystcomp_utils.cycle_data import CycleData
from transformertf.data import EncoderDecoderDataModule
from transformertf.data.dataset import EncoderDecoderDataset
from transformertf.models.pete import PETE
from transformertf.utils import ops

from ._base_predictor import BasePredictor, T_DataModule_co, T_Module_co

HiddenState = tuple[torch.Tensor, torch.Tensor]

log = logging.getLogger(__name__)


class PETEPredictor(BasePredictor):
    _module: PETE
    _datamodule: EncoderDecoderDataModule

    def __init__(self, device: typing.Literal["cpu", "cuda", "auto"] = "auto") -> None:
        super().__init__(device=device)

        self.state: HiddenState | None = None

    def set_initial_state(
        self,
        initial_state: tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]
        | tuple[torch.Tensor, torch.Tensor]
        | None = None,
        initial_state_path: str | os.PathLike | None = None,
        past_covariates: pd.DataFrame | None = None,
        past_targets: pd.DataFrame | np.ndarray | None = None,
    ) -> None:
        if initial_state is not None:
            # convert to torch tensor
            initial_state = (
                torch.tensor(initial_state[0]),
                torch.tensor(initial_state[1]),
            )

            self._initial_state = initial_state
        elif initial_state_path is not None:
            # load from file
            initial_state_ = torch.load(initial_state_path)
            self.state = tuple(o[0] for o in torch.split(initial_state_, 2))
        elif past_covariates is not None:
            if past_targets is None:
                if "__target__" in past_covariates:
                    past_targets = past_covariates.pop("__target__").to_numpy()
                elif self._datamodule.hparams["target_covariate"] in past_covariates:
                    past_targets = past_covariates.pop(
                        self._datamodule.hparams["target_covariate"]
                    ).to_numpy()
                else:
                    msg = (
                        "past_targets must be provided if '__target__' is not "
                        "in past_covariates."
                    )
                    raise ValueError(msg)

            self._fit_initial_state(
                past_covariates,
                past_targets,
            )
        else:
            msg = "Either initial_state or initial_state_path must be provided."
            raise ValueError(msg)

    def reset_state(self) -> None:
        self.state = None

    def set_cycled_initial_state(
        self,
        cycles: list[CycleData],
        *args: typing.Any,
        use_programmed_current: bool = True,
        **kwargs: typing.Any,
    ) -> None:
        past_covariates = BasePredictor.buffer_to_covariates(
            cycles[:-1],
            use_programmed_current=use_programmed_current,
        )
        past_targets = past_covariates.pop("__target__").to_numpy()

        self.set_initial_state(
            *args,
            past_covariates=past_covariates,  # type: ignore[misc]
            past_targets=past_targets,  # type: ignore[misc]
            **kwargs,
        )

    def _fit_initial_state(
        self,
        past_covariates: pd.DataFrame,
        past_targets: pd.DataFrame | np.ndarray,
    ) -> None:
        if isinstance(past_targets, np.ndarray):
            past_targets = pd.DataFrame(
                {self._datamodule.hparams["target_covariate"]: past_targets}
            )

        df = pd.concat([past_covariates, past_targets], axis=1)

        transformed_covariates = self.preprocess_df(
            df, seq_len=self._datamodule.hparams["ctxt_seq_len"]
        )

        sample = EncoderDecoderDataset.make_encoder_input(
            transformed_covariates,
            seq_len=self._datamodule.hparams["ctxt_seq_len"],
            time_format=self._datamodule.hparams["time_format"],
        )
        sample = ops.to(sample, self._module.device)

        with torch.no_grad():
            self.state = self._module.encoder(sample["encoder_input"][None, ...])["hx"]

    def preprocess_df(
        self, df: pd.DataFrame, seq_len: int | None = None
    ) -> pd.DataFrame:
        df = df.rename(
            {"__target__": self._datamodule.hparams["target_covariate"]},
            axis=1,
        )
        df = self._datamodule.transform_input(df, timestamp=df["__time__"].to_numpy())

        if seq_len is not None:
            if len(df) < seq_len:
                msg = "df must have at least seq_len rows."
                raise ValueError(msg)

            df = df.iloc[-seq_len:]

        if self._datamodule.hparams["time_column"] is None:
            df = df.drop("__time__", axis=1)

        return df

    def predict(
        self,
        future_covariates: pd.DataFrame,
        *args: typing.Any,
        save_state: bool = True,
        **kwargs: typing.Any,
    ) -> npt.NDArray[np.float64]:
        self._check_state()

        preprocessed_df = self.preprocess_df(future_covariates, seq_len=None)

        sample = EncoderDecoderDataset.make_decoder_input(
            preprocessed_df,
            seq_len=None,
            time_format=self._datamodule.hparams["time_format"],
            transforms=self._datamodule.transforms,
        )
        x = sample["decoder_input"][
            None, ..., : self._datamodule.num_future_known_covariates - 1
        ]
        x = ops.to(x, self._module.device)

        with torch.no_grad():
            prediction, state = self._module.bwlstm1(
                x,
                hx=self.state,
                return_states=True,
            )
            if save_state:
                self.state = state

            prediction = prediction["z"][..., 0].squeeze().detach().cpu()

        target_transform = self._datamodule.target_transform

        # inverse transform
        if self._datamodule.hparams["target_depends_on"] is None:
            prediction = target_transform.inverse_transform(prediction)
        else:
            # inverse transform the target dependent covariate
            transformed_name = next(
                filter(
                    lambda x: x.endswith(self._datamodule.hparams["target_depends_on"]),
                    preprocessed_df.columns,
                )
            )
            transform = self._datamodule.transforms[
                self._datamodule.hparams["target_depends_on"]
            ]
            target_depends_on_val = transform.inverse_transform(
                preprocessed_df[transformed_name].to_numpy()
            )

            prediction = target_transform.inverse_transform(
                target_depends_on_val, prediction
            )

        return prediction.numpy().astype(np.float64)

    def predict_cycle(
        self,
        cycle: CycleData,
        *args: typing.Any,
        save_state: bool = True,
        use_programmed_current: bool = True,
        **kwargs: typing.Any,
    ) -> npt.NDArray[np.float64]:
        future_covariates = BasePredictor.buffer_to_covariates(
            [cycle],
            use_programmed_current=use_programmed_current,
            add_target=False,
        )

        prediction = self.predict(future_covariates, save_state=save_state, **kwargs)

        # interpolate to match the length of the cycle at 1kHz
        time = np.arange(0, cycle.num_samples) / 1e3
        time = time[:: self._datamodule.hparams["downsample"]]

        return np.vstack((time, prediction))

    def predict_last_cycle(
        self,
        cycle_data: list[CycleData],
        *args: typing.Any,
        save_state: bool = True,
        autoregressive: bool = False,
        use_programmed_current: bool = True,
        **kwargs: typing.Any,
    ) -> npt.NDArray[np.float64]:
        if autoregressive or self.state is None:
            self.set_initial_state(
                past_covariates=BasePredictor.buffer_to_covariates(
                    cycle_data[:-1],
                    use_programmed_current=use_programmed_current,
                ),
            )

        return self.predict_cycle(
            cycle_data[-1],
            save_state=save_state,
            use_programmed_current=use_programmed_current,
            **kwargs,
        )

    def _check_state(self) -> None:
        if self.state is None:
            msg = (
                "Initial state must be set before making predictions. "
                "Use the set_initial_state method."
            )
            raise ValueError(msg)

    @classmethod
    def load_from_checkpoint(
        cls: type[BasePredictor[T_Module_co, T_DataModule_co]],
        checkpoint_path: str | os.PathLike,
        device: typing.Literal["cpu", "cuda", "auto"] = "auto",
    ) -> BasePredictor[T_Module_co, T_DataModule_co]:
        predictor = cls(device=device)
        predictor.load_checkpoint(checkpoint_path)

        return predictor

    def load_checkpoint(self, checkpoint_path: str | os.PathLike) -> None:
        self._module = PETE.load_from_checkpoint(checkpoint_path)
        self._datamodule = EncoderDecoderDataModule.load_from_checkpoint(
            checkpoint_path
        )

        self._reconfigure_module()
