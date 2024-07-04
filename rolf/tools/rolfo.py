from pathlib import Path

import numpy as np
from optuna import Trial, create_study

from rolf.io.data import ReadHDF5
from rolf.tools.toml_reader import ReadConfig
from rolf.training.training import train_model


class ParameterOptimization:

    def __init__(self, optim_conf_path, optuna_path, data_path, random_state=None) -> None:
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
                "random_state has to be None, int or numpy.random.mtrand.RandomState instance!"
            )
    
    def load_data(self, image_path) -> None:
        self.data = ReadHDF5(self.data_path, validation_ratio=0.1, test_ratio=0.05, random_state=self.random_state)
        self.train_loader, self.val_loader, self.test_loader = self.data.create_data_loaders(batch_size=self.model_config["batch_size"], img_dir=image_path)

    def make_network(self, trial : Trial) -> None:
        
        self.use_tuning = {}
        for key in self.tuning_config:
            pars = self.tuning_config[key]
            if isinstance(pars, tuple):
                if np.all([isinstance(pars[i], int) for i in range(3)]):
                    self.use_tuning[key] = trial.suggest_int(key, low=pars[0], high=pars[1], step=pars[2])
                elif np.all([isinstance(pars[i], float) for i in range(3)]):
                    self.use_tuning[key] = trial.suggest_float(key, low=pars[0], high=pars[1], step=pars[2])
                else:
                    raise TypeError("All parameters have to be of the same type")
            elif isinstance(pars, list):
                self.use_tuning[key] = trial.suggest_categorical(key, pars)
            else:
                self.use_tuning[key] = pars
        
        self.use_tuning["block_groups"] = [self.use_tuning["block_groups"]]*4

        if self.use_tuning["optimizer"] == "Adam":
            hparams_keys = ["lr", "weight_decay"]
        else:
            hparams_keys = ["lr", "momentum", "weight_decay"]

        optimizer_hparams = {}
        for key in hparams_keys:
            val = self.use_tuning[key]
            optimizer_hparams[key] = val

        model_hparams = {}
        for key in ["block_groups", "block_name", "activation_name"]:
            val = self.use_tuning[key]
            model_hparams[key] = val
        model_hparams["num_classes"] = self.model_config["net_hyperparams"]["num_classes"]
        model_hparams["hidden_channels"] = self.model_config["net_hyperparams"]["hidden_channels"]

        self.model, self.result, _ = train_model(
            model_name=self.model_config["model_name"],
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            test_loader=self.test_loader,
            checkpoint_path=self.model_config["paths"]["model"],
            save_name=self.model_config["save_name"],
            model_hparams=model_hparams,
            optimizer_name=self.use_tuning["optimizer"],
            optimizer_hparams=optimizer_hparams,
            epochs=self.model_config["epochs"]
        )

        self.score = self.result["val"]

    def objective(self, trial : Trial) -> float:
        self.make_network(trial)
        return self.score
    
    def optimize(self, study_name, direction, n_trials, n_jobs) -> None:
        self.study = create_study(study_name=study_name, direction=direction, storage=self.optuna_path, load_if_exists=True)
        self.study.optimize(lambda trial: self.objective(trial), n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=True, gc_after_trial=True)