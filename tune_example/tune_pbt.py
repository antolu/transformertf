"""
Do a hyperparameter scan for the LSTM model.
"""

from __future__ import annotations

import math
import os
import pathlib
import typing
from pprint import pprint

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


CONFIG_PATH = HERE / "lstm_config.yml"


GRID = {
    "dropout": ray.tune.uniform(0.1, 0.5),
    "lr": ray.tune.loguniform(1e-5, 1e-2),
    "batch_size": ray.tune.choice([32, 64, 128, 256, 512]),
}

PARAM2KEY = {
    "dropout": "model.init_args.dropout",
    "lr": "optimizer.init_args.lr",
    "batch_size": "data.init_args.batch_size",
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
    new_config = apply_config(config, cli_config)
    # train the model
    pprint(new_config)

    if "callbacks" not in new_config["trainer"]:
        new_config["trainer"]["callbacks"] = []
    tune_callback = make_tune_callback(metrics)
    new_config["trainer"]["callbacks"].append(tune_callback)

    cli = LightningCLI(args=new_config, run=False)

    trainer = cli.trainer
    model = cli.model
    dm = cli.datamodule

    # If `train.get_checkpoint()` is populated, then we are resuming from a checkpoint.
    checkpoint = ray.train.get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            checkpoint_path = os.path.join(checkpoint_dir, "checkpoint")

            state_dict = torch.load(checkpoint_path)
            model.load_state_dict(state_dict["state_dict"])
            dm = dm.load_from_checkpoint(checkpoint_path)
    else:
        checkpoint_path = None

    trainer.fit(model, dm, ckpt_path=checkpoint_path)

    return {}


def main() -> None:
    with open(CONFIG_PATH, encoding="locale") as f:
        cli_config = yaml.safe_load(f)
    cli_config.pop("ckpt_path")
    reporter = ray.tune.CLIReporter(
        parameter_columns=list(GRID.keys()),
        metric_columns=metrics,
    )
    scheduler = ray.tune.schedulers.PopulationBasedTraining(
        time_attr="training_iteration",
        metric="MSE/validation/dataloader_idx_1",
        mode="min",
        perturbation_interval=10,
        hyperparam_mutations=GRID,
    )
    tune_config = ray.tune.TuneConfig(
        scheduler=scheduler,
        num_samples=100,
        max_concurrent_trials=4,
    )
    run_config = ray.train.RunConfig(
        name="PBT-LSTM",
        progress_reporter=reporter,
        storage_path=os.fspath(HERE / "pbt_results"),
        local_dir=os.fspath(HERE / "pbt_results"),
        stop=lambda trial_id, result: math.isnan(result["loss/train"]),
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
    )

    results_grid = tuner.fit()
    results_grid.get_dataframe().to_parquet(HERE / "pbt_results/results.parquet")


if __name__ == "__main__":
    main()
