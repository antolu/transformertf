"""
Do a hyperparameter scan for the LSTM model.

DEPRECATED: This file uses the old Python-based tuning API which has been replaced
with the YAML-based configuration system. See lstm_tune_config.yml for the new approach.
This file is kept for reference but is no longer functional.
"""

from __future__ import annotations

import math
import os
import pathlib

import ray.tune
import ray.tune.search.hyperopt
import torch

from transformertf.main import setup_logger
from transformertf.utils.tune import PBTTuneConfig, tune

torch.set_float32_matmul_precision("high")

HERE = pathlib.Path(__file__).resolve().parent


# ============= Configuration =============

CONFIG_PATH = HERE / "lstm_config.yml"

GRID = {
    "dropout": ray.tune.uniform(0.1, 0.5),
    "lr": ray.tune.loguniform(1e-5, 1e-2),
    "momentum": ray.tune.uniform(0.1, 0.95),
    "weight_decay": ray.tune.loguniform(1e-5, 1e-2),
    "batch_size": ray.tune.choice([256, 512, 1024]),
}

PARAM2KEY = {
    "dropout": "model.init_args.dropout",
    "lr": "optimizer.init_args.lr",
    "momentum": "optimizer.init_args.momentum",
    "weight_decay": "optimizer.init_args.weight_decay",
    "batch_size": "data.init_args.batch_size",
}

metrics = [
    "loss/train",
    "loss/validation/dataloader_idx_0",
    "loss/validation/dataloader_idx_1",
    "MSE/validation/dataloader_idx_0",
    "MSE/validation/dataloader_idx_1",
    "RMSE/validation/dataloader_idx_0",
    "RMSE/validation/dataloader_idx_1",
    "SMAPE/validation/dataloader_idx_0",
    "SMAPE/validation/dataloader_idx_1",
]

MONITOR = "RMSE/validation/dataloader_idx_1"
LOG_DIR = HERE / "pbt_results"
RUN_NAME = "PBT-LSTM"
NUM_EPOCHS = 500
PERTURBATION_INTERVAL = 10


# ========= End of Configuration =========


def stop_fn(trial_id: str, result: dict[str, float]) -> bool:
    if math.isnan(result["loss/train"]):
        return True

    return result["SMAPE/validation/dataloader_idx_1"] < 0.01


def main() -> None:
    setup_logger(logging_level=1)
    tune_config = PBTTuneConfig(
        grid=GRID,
        param2key=PARAM2KEY,
        cli_config_path=os.fspath(CONFIG_PATH),
        metrics=metrics,
        monitor=MONITOR,
        log_dir=os.fspath(LOG_DIR),
        run_name=RUN_NAME,
        num_samples=100,
        stop_condition=stop_fn,
        num_epochs_per_trial=NUM_EPOCHS,
        perturbation_interval=PERTURBATION_INTERVAL,
    )
    results_grid = tune(tune_config)  # type: ignore[arg-type]  # Deprecated API
    results_grid.get_dataframe().to_parquet(LOG_DIR / "results.parquet")

    best_trial = results_grid.get_best_result(MONITOR, "min", "last")

    print(f"Best trial: {best_trial}")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial checkpoint: {best_trial.checkpoint}")


if __name__ == "__main__":
    main()
