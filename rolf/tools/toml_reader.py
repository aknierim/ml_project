from pathlib import Path

import tomli


class ReadConfig:
    def __init__(self, config_path) -> None:
        self.config_path = Path(config_path)
        with open(self.config_path, "rb") as f:
            self.toml = tomli.load(f)
        self.paths = self.toml["paths"]
        self.mode = self.toml["mode"]
        self.meta = self.toml["meta"]
        self.model = self.toml["model"]
        self.net_hyperparams = self.toml["net_hyperparams"]
        self.optimizer = self.toml["optimizer"]
        self.opt_hyperparams = self.toml["optimizer_hyperparams"]

    def training(self) -> dict:
        self.training_config = {}
        self.training_config["paths"] = {
            "data": Path(self.paths["data"]),
            "model": Path(self.paths["model"]),
        }
        self.training_config["mode"] = self.mode
        self.training_config["parameters"] = self.meta
        self.training_config["model_name"] = self.model["name"]
        self.training_config["net_hyperparams"] = self.net_hyperparams
        self.training_config["optimizer"] = self.optimizer["name"]
        self.training_config["opt_hyperparams"] = self.opt_hyperparams
        self.training_config["save_name"] = self.meta["save_name"]
        self.training_config["batch_size"] = self.meta["batch_size"]
        self.training_config["epochs"] = self.meta["epochs"]

        return self.training_config

    # def validation(self) -> dict:
    #     self.validation_config = self.eval
    #
    #     return self.validation_config
