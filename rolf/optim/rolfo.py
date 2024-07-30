from pathlib import Path

import numpy as np
from optuna import Trial, create_study
from rich.pretty import pprint
from torch.cuda import OutOfMemoryError

from rolf.io.data import ReadHDF5
from rolf.tools.toml_reader import ReadConfig
from rolf.training.training import train_model


def map_suggestions(trial: Trial, key: str):
    """Lookup dictionary of hyperparamters and
    trial suggestions.

    Parameters
    ----------
    trial : Trial
        Optuna trial instance.
    key : str
        Key for the hyperparamter.

    Returns
    -------
    mapping
        Mapping of the respective key and trial.
    """
    trial_suggest = {
        "hidden_channels": trial.suggest_categorical,
        "num_groups": trial.suggest_int,
        "block_groups": trial.suggest_int,
        "block_name": trial.suggest_categorical,
        "activation_name": trial.suggest_categorical,
        "optimizer": trial.suggest_categorical,
        "lr": trial.suggest_float,
        "momentum": trial.suggest_float,
        "weight_decay": trial.suggest_float,
        "dropout": trial.suggest_float,
    }

    return trial_suggest[key]


class ParameterOptimization:
    """Main hyperparamter optimization class."""

    def __init__(
        self,
        optim_conf_path: str | Path,
        optuna_path: str | Path,
        data_path: str | Path,
        random_state: int | np.random.RandomState = None,
        validation_ratio: float = 0.1,
        test_ratio: float = 0.05,
        devices: int | list = 1,
    ) -> None:
        """Initializes the hyperparamter optimization.

        Parameters
        ----------
        optim_conf_path : str or Path
            Path of the optimization config file.
        optuna_path : str or Path
            Optuna study output path. Saves to a sqlite database.
        data_path : str or Path
            Path to the data files.
        random_state : int or np.random.RandomState
            Seed for the data loaders.
        validation_ratio : float
            Validation ratio for the data split.
        test_ratio : float
            Test ratio for the data split.
        devices : int or list
            Number of devices or list of specific device
            IDs.
        """
        self.optim_conf_path = Path(optim_conf_path)
        reader = ReadConfig(self.optim_conf_path)

        self.model_config = reader.training()
        self.tuning_config = reader.tuning()
        self.optuna_path = optuna_path
        self.data_path = Path(data_path)

        self.validation_ratio = validation_ratio
        self.test_ratio = test_ratio

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
        """Primes image path, loads data.

        Parameters
        ----------
        image_path : str or Path
            Path to the images.
        """
        self.image_path = image_path
        self._load_data()

    def _load_data(self):
        """Main data loader method"""
        self.data = ReadHDF5(
            self.data_path,
            validation_ratio=self.validation_ratio,
            test_ratio=self.test_ratio,
            random_state=self.random_state,
        )
        self.data.make_transformer()

        (
            self.train_loader,
            self.val_loader,
            self.test_loader,
        ) = self.data.create_data_loaders(
            batch_size=self.model_config["batch_size"], img_dir=self.image_path
        )

    def make_network(self, trial: Trial) -> None:
        """Creates network for the current trial.

        Parameters
        ----------
        trial : Trial
            Current trial.
        """
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

        use_tuning["hidden_channels"] *= 2 ** np.arange(use_tuning["num_groups"])
        use_tuning["hidden_channels"] = use_tuning["hidden_channels"].tolist()

        use_tuning["block_groups"] = np.full(
            use_tuning["num_groups"], use_tuning["block_groups"]
        ).tolist()

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

        result = None
        while result is None:
            try:
                model, result = self._call_model(
                    model_hparams, use_tuning, optimizer_hparams
                )
            except OutOfMemoryError as e:
                pprint(e)
                if self.model_config["batch_size"] > 10:
                    self.model_config["batch_size"] -= 10
                else:
                    self.model_config["batch_size"] = int(
                        self.model_config["batch_size"] / 2
                    )
                pprint(
                    f"Reducing batch size to {self.model_config['batch_size']} "
                    "and trying again..."
                    ""
                )
                self._load_data()

        self.score = (result["val"]["auc"], result["val"]["acc"])

    def _call_model(
        self, model_hparams: dict, use_tuning: dict, optimizer_hparams: dict
    ) -> tuple:
        """Calls model with current parameter set.

        Parameters
        ----------
        model_hparams : dict
            Dictionary of model hyperparamters.
        use_tuning : dict
            Dictionary of tuning hyperparamters.
        optimizer_hparams : dict
            Dictionary of optimizer hyperparamters.

        Returns
        -------
        model
            Trained model instance.
        result : dict
            Dictionary of the results.
        """
        model, result, _ = train_model(
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
            devices=self.devices,
        )

        return model, result

    def objective(self, trial: Trial) -> float:
        """Objective method.

        Parameters
        ----------
        trial : Trial
            Current trial.

        Returns
        -------
        score : dict
            Dictionary of the results.
        """
        self.make_network(trial)
        return self.score

    def optimize(
        self, study_name: str, direction: str | list, n_trials: int, n_jobs: int
    ) -> None:
        """Main study method. Initializes study
        and starts the optimization process.

        Parameters
        ----------
        study_name : str
            Name of the current study.
        direction : list
            List of study directions for score evaluation.
        n_trials : int
            Number of randomized hyperparamter trials.
        n_jobs : int
            Number of parallel jobs to run.
        """
        self.study = create_study(
            study_name=study_name,
            directions=direction,
            storage=self.optuna_path,
            load_if_exists=True,
        )
        self.study.set_metric_names(["ROC AUC", "Accuracy"])
        self.study.optimize(
            lambda trial: self.objective(trial),
            n_trials=n_trials,
            n_jobs=n_jobs,
            show_progress_bar=True,
            gc_after_trial=True,
        )
