"""
Do a hyperparameter scan for the LSTM model.
"""

from __future__ import annotations

import math
import os
import pathlib
import typing

import ray.train
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
    "hidden_dim": ray.tune.choice([100, 200, 300, 400, 500]),
    "hidden_dim_fc": ray.tune.choice([512, 1024, 2048]),
    "seq_len": ray.tune.choice([300, 600, 900, 1200]),
}

PARAM2KEY = {
    "num_layers": "model.init_args.num_layers",
    "hidden_dim": "model.init_args.hidden_dim",
    "hidden_dim_fc": "model.init_args.hidden_dim_fc",
    "seq_len": "data.init_args.seq_len",
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

MONITOR = "MSE/validation/dataloader_idx_1"
LOG_DIR = HERE / "lstm_results"
RUN_NAME = "ASHA-LSTM"
NUM_EPOCHS = 500
PATIENCE = 50
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

    def apply_key(
        config: dict[str, typing.Any], key: list[str], value: typing.Any
    ) -> None:
        """Recursively apply the key to the config."""
        if len(key) == 1:
            config[key[0]] = value
        else:
            apply_key(config[key[0]], key[1:], value)

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

    trainer.fit(model, dm)

    return {}


def stop_fn(trial_id: str, result: dict[str, float]) -> bool:
    if math.isnan(result["loss/train"]):
        return True

    return result["SMAP/validation/dataloader_idx_1"] < 0.01


def main() -> None:
    with open(CONFIG_PATH, encoding="locale") as f:
        cli_config = yaml.safe_load(f)
    cli_config.pop("ckpt_path")
    reporter = ray.tune.CLIReporter(
        parameter_columns=list(GRID.keys()),
        metric_columns=metrics,
    )
    search_alg = ray.tune.search.hyperopt.HyperOptSearch(
        metric=MONITOR,
        mode="min",
        points_to_evaluate=[
            {key: read_from_dot_key(cli_config, PARAM2KEY[key]) for key in GRID}
        ],
    )
    scheduler = ray.tune.schedulers.ASHAScheduler(
        max_t=NUM_EPOCHS, grace_period=PATIENCE, reduction_factor=2
    )
    tune_config = ray.tune.TuneConfig(
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
        # checkpoint_config=ray.train.CheckpointConfig(
        #     checkpoint_score_attribute="SMAPE/validation/dataloader_idx_1",
        #     checkpoint_score_order="min",
        #     num_to_keep=3,
        # ),
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


if __name__ == "__main__":
    main()
