"""
Do a hyperparameter scan for the LSTM model.
"""

from __future__ import annotations

import math
import os
import pathlib
import typing

import ray
import ray.train
import ray.train.lightning
import ray.tune
import ray.tune.integration.pytorch_lightning
import ray.tune.schedulers
import ray.tune.search
import ray.tune.search.hyperopt
import torch
import yaml

from transformertf.main import LightningCLI

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


def read_from_dot_key(config: dict[str, typing.Any], key: str) -> typing.Any:
    """
    Read a value from a dot-separated key.

    Parameters
    ----------
    config : dict
        The config to read from.
    key : str
        The key to read from.

    Returns
    -------
    Any
        The value at the key.
    """
    key_split = key.split(".")
    value = config
    for k in key_split:
        value = value[k]

    return value


def apply_key(
    config: dict[str, typing.Any], key: list[str] | str, value: typing.Any
) -> None:
    """Recursively apply the key to the config."""
    if isinstance(key, str):
        key = key.split(".")

    if len(key) == 1:
        config[key[0]] = value
    else:
        if key[0] not in config:
            config[key[0]] = {}
        apply_key(config[key[0]], key[1:], value)


def apply_config(
    config: dict[str, typing.Any], cli_config: dict[str, typing.Any]
) -> dict[str, typing.Any]:
    """
    Apply the hyperparameters to the config using the PARAM2KEY mapping.

    Parameters
    ----------
    config : dict
        The hyperparameters to apply.
    cli_config : dict
        The config to apply the hyperparameters to, which is a config for
        LightningCLI.

    Returns
    -------
    dict
        The new config with the hyperparameters applied.
    """
    new_config = cli_config.copy()

    for key, value in config.items():
        new_key = PARAM2KEY[key]
        key_split = new_key.split(".")
        apply_key(new_config, key_split, value)

    return new_config


def make_tune_callback(metrics: list[str] | None = None) -> dict[str, typing.Any]:
    """Defines the TuneReportCallback."""
    metrics = metrics or []

    return {
        "class_path": "transformertf.tune.TuneReportCallback",
        "init_args": {
            "on": "validation_end",
            "metrics": metrics,
        },
    }


def train_fn(
    config: dict[str, typing.Any], cli_config: dict[str, typing.Any]
) -> dict[str, float]:
    torch.set_float32_matmul_precision("high")
    new_config = apply_config(config, cli_config)

    if "callbacks" not in new_config["trainer"]:
        new_config["trainer"]["callbacks"] = []
    tune_callback = make_tune_callback(metrics)
    new_config["trainer"]["callbacks"].append(tune_callback)

    cli = LightningCLI(args=new_config, run=False)

    trainer = cli.trainer
    model = cli.model
    dm = cli.datamodule

    # trainer = ray.train.lightning.prepare_trainer(trainer)
    trainer.fit(model, dm)

    return {}


def stop_fn(trial_id: str, result: dict[str, float]) -> bool:
    if math.isnan(result["loss/train"]):
        return True

    return result["SMAPE/validation/dataloader_idx_1"] < 0.01


def main() -> None:
    with open(CONFIG_PATH, encoding="locale") as f:
        cli_config = yaml.safe_load(f)
    cli_config.pop("ckpt_path")

    apply_key(cli_config, "trainer.max_epochs", NUM_EPOCHS)
    apply_key(
        cli_config,
        "trainer.callbacks",
        [
            {
                "class_path": "ray.train.lightning.RayTrainReportCallback",
            }
        ],
    )
    apply_key(cli_config, "trainer.enable_progress_bar", value=False)
    apply_key(
        cli_config,
        "trainer.plugins",
        [{"class_path": "ray.train.lightning.RayLightningEnvironment"}],
    )

    reporter = ray.tune.CLIReporter(
        parameter_columns=list(GRID.keys()),
        metric_columns=metrics,
    )
    search_alg = ray.tune.search.hyperopt.HyperOptSearch(
        metric=MONITOR,
        mode="min",
    )
    scheduler = ray.tune.schedulers.ASHAScheduler(
        max_t=NUM_EPOCHS, grace_period=PATIENCE, reduction_factor=2
    )
    tune_config = ray.tune.TuneConfig(
        metric=MONITOR,
        mode="min",
        scheduler=scheduler,
        num_samples=NUM_SAMPLES,
        max_concurrent_trials=4,
        search_alg=search_alg,
    )
    run_config = ray.train.RunConfig(
        name=RUN_NAME,
        progress_reporter=reporter,
        storage_path=os.fspath(LOG_DIR),
        local_dir=os.fspath(LOG_DIR),
        stop=stop_fn,
        checkpoint_config=ray.train.CheckpointConfig(
            checkpoint_score_attribute=MONITOR,
            checkpoint_score_order="min",
            num_to_keep=3,
        ),
    )

    tuner = ray.tune.Tuner(
        ray.tune.with_resources(
            ray.tune.with_parameters(train_fn, cli_config=cli_config),
            resources={"cpu": 4, "gpu": 1},
        ),
        tune_config=tune_config,
        run_config=run_config,
        param_space=GRID,
    )

    results_grid = tuner.fit()
    results_grid.get_dataframe().to_parquet(LOG_DIR / "results.parquet")

    best_trial = results_grid.get_best_result(MONITOR, "min", "last")

    print(f"Best trial: {best_trial}")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial checkpoint: {best_trial.checkpoint}")


if __name__ == "__main__":
    main()
