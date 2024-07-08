from pathlib import Path

import numpy as np
from optuna import Trial, create_study
from rich.pretty import pprint

from rolf.io.data import ReadHDF5
from rolf.tools.toml_reader import ReadConfig
from rolf.training.training import train_model


def map_suggestions(trial, key):
    trial_suggest = {
        "hidden_channels": trial.suggest_categorical,
        "block_groups": trial.suggest_int,
        "block_name": trial.suggest_categorical,
        "activation_name": trial.suggest_categorical,
        "optimizer": trial.suggest_categorical,
        "lr": trial.suggest_float,
        "momentum": trial.suggest_float,
        "weight_decay": trial.suggest_float,
    }

    return trial_suggest[key]


class ParameterOptimization:
    def __init__(
        self,
        optim_conf_path: str | Path,
        optuna_path: str | Path,
        data_path: str | Path,
        random_state: int | np.random.RandomState = None,
    ) -> None:
        self.optim_conf_path = Path(optim_conf_path)
        reader = ReadConfig(self.optim_conf_path)
        self.model_config = reader.training()
        self.tuning_config = reader.tuning()
        self.optuna_path = optuna_path
        self.data_path = Path(data_path)

        if random_state is None:
            self.random_state = np.random.mtrand.RandomState()
        elif isinstance(random_state, int):
            self.random_state = np.random.mtrand.RandomState(random_state)
        elif type(random_state) == np.random.mtrand.RandomState:
            self.random_state = random_state
        else:
            raise TypeError(
                "random_state has to be None, int or "
                "numpy.random.mtrand.RandomState instance!"
            )

    def load_data(self, image_path: str | Path) -> None:
        self.data = ReadHDF5(
            self.data_path,
            validation_ratio=0.1,
            test_ratio=0.05,
            random_state=self.random_state,
        )
        (
            self.train_loader,
            self.val_loader,
            self.test_loader,
        ) = self.data.create_data_loaders(
            batch_size=self.model_config["batch_size"], img_dir=image_path
        )

    def make_network(self, trial: Trial) -> None:
        use_tuning = {}

        for key in self.tuning_config:
            if not isinstance(self.tuning_config[key], list):
                use_tuning[key] = map_suggestions(trial, key)(
                    key, **self.tuning_config[key]
                )
            else:
                use_tuning[key] = map_suggestions(trial, key)(
                    key, self.tuning_config[key]
                )

        use_tuning["hidden_channels"] *= np.geomspace(1, 8, num=4, dtype=int)
        use_tuning["block_groups"] = np.full(4, use_tuning["block_groups"])

        optimizer_hparams = {
            "lr": use_tuning["lr"],
            "weight_decay": use_tuning["weight_decay"],
        }
        if use_tuning["optimizer"] == "SGD":
            optimizer_hparams["momentum"] = use_tuning["momentum"]

        model_hparams = {}
        for key in ["hidden_channels", "block_groups", "block_name", "activation_name"]:
            model_hparams[key] = use_tuning[key]

        model_hparams["num_classes"] = self.model_config["net_hyperparams"][
            "num_classes"
        ]

        pprint("Model parameters:")
        pprint(model_hparams)
        pprint(f"Optimizer '{use_tuning['optimizer']}' parameters:")
        pprint(optimizer_hparams)

        self.model, self.result, _ = train_model(
            model_name=self.model_config["model_name"],
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            test_loader=self.test_loader,
            checkpoint_path=self.model_config["paths"]["model"],
            save_name=self.model_config["save_name"],
            model_hparams=model_hparams,
            optimizer_name=use_tuning["optimizer"],
            optimizer_hparams=optimizer_hparams,
            epochs=self.model_config["epochs"],
        )

        self.score = self.result["val"]

    def objective(self, trial: Trial) -> float:
        self.make_network(trial)
        return self.score

    def optimize(
        self, study_name: str, direction: str, n_trials: int, n_jobs: int
    ) -> None:
        self.study = create_study(
            study_name=study_name,
            direction=direction,
            storage=self.optuna_path,
            load_if_exists=True,
        )
        self.study.optimize(
            lambda trial: self.objective(trial),
            n_trials=n_trials,
            n_jobs=n_jobs,
            show_progress_bar=True,
            gc_after_trial=True,
        )
