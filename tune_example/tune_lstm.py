"""
Do a hyperparameter scan for the LSTM model.
"""

from __future__ import annotations

import math
import os
import pathlib

import ray
import ray.train
import ray.train.lightning
import ray.tune
import ray.tune.integration.pytorch_lightning
import ray.tune.schedulers
import ray.tune.search
import ray.tune.search.hyperopt
import torch

from transformertf.main import setup_logger
from transformertf.utils.tune import ASHATuneConfig, tune

torch.set_float32_matmul_precision("high")

HERE = pathlib.Path(__file__).resolve().parent


# ============= Configuration =============

CONFIG_PATH = HERE / "lstm_config.yml"

GRID = {
    "num_layers": ray.tune.choice([1, 2, 3]),
    "n_dim_model": ray.tune.choice([100, 200, 300, 400, 500]),
    "seq_len": ray.tune.choice([300, 600, 900, 1200]),
    "quantiles": ray.tune.choice([
        [0.1, 0.5, 0.9],
        [0.25, 0.5, 0.75],
        [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98],
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        [
            0.067,
            0.133,
            0.2,
            0.267,
            0.333,
            0.4,
            0.467,
            0.5,
            0.533,
            0.6,
            0.667,
            0.733,
            0.8,
            0.867,
            0.933,
        ],
        # 15 quantiles
        [0.05 * i for i in range(1, 21)],
    ]),
    "downsample": ray.tune.choice([10, 20, 40, 50, 60, 80, 100]),
}

PARAM2KEY = {
    "num_layers": "model.init_args.num_layers",
    "n_dim_model": "model.init_args.n_dim_model",
    "seq_len": "data.init_args.seq_len",
    "quantiles": "model.init_args.criterion.init_args.quantiles",
    "downsample": "data.init_args.downsample",
}

metrics = [
    "RMSE/validation/dataloader_idx_0",
    "RMSE/validation/dataloader_idx_1",
    "loss/train",
    "loss/validation/dataloader_idx_0",
    "loss/validation/dataloader_idx_1",
    "MSE/validation/dataloader_idx_0",
    "MSE/validation/dataloader_idx_1",
    "SMAPE/validation/dataloader_idx_0",
    "SMAPE/validation/dataloader_idx_1",
]

MONITOR = "RMSE/validation/dataloader_idx_1"
LOG_DIR = HERE / "lstm_results"
RUN_NAME = "ASHA-LSTM"
NUM_EPOCHS = 500
PATIENCE = 10
NUM_SAMPLES = 100


# ========= End of Configuration =========


def stop_fn(trial_id: str, result: dict[str, float]) -> bool:
    if math.isnan(result["loss/train"]):
        return True

    return result["SMAPE/validation/dataloader_idx_1"] < 0.01


def main() -> None:
    setup_logger(logging_level=1)
    tune_config = ASHATuneConfig(
        grid=GRID,
        param2key=PARAM2KEY,
        cli_config_path=os.fspath(CONFIG_PATH),
        num_samples=NUM_SAMPLES,
        num_epochs_per_trial=NUM_EPOCHS,
        stop_condition=stop_fn,
        log_dir=os.fspath(LOG_DIR),
        run_name=RUN_NAME,
        metrics=metrics,
        monitor=MONITOR,
        patience=PATIENCE,
    )

    results_grid = tune(tune_config)
    results_grid.get_dataframe().to_parquet(LOG_DIR / "results.parquet")

    best_trial = results_grid.get_best_result(MONITOR, "min", "last")

    print(f"Best trial: {best_trial}")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial checkpoint: {best_trial.checkpoint}")


if __name__ == "__main__":
    main()
