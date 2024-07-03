from pathlib import Path

import tomli


class ReadConfig:
    def __init__(self, config_path) -> None:
        self.config_path = Path(config_path)
        with open(self.config_path, "rb") as f:
            self.toml = tomli.load(f)

        try:
            self.tuning_toml = self.toml["tuning"]
            self.tuning_mode = True
        except KeyError:
            self.tuning_mode = False

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
        self.training_config["model_name"] = self.model["name"]
        self.training_config["net_hyperparams"] = self.net_hyperparams
        self.training_config["optimizer"] = self.optimizer["name"]
        self.training_config["opt_hyperparams"] = self.opt_hyperparams
        self.training_config["save_name"] = self.meta["save_name"]
        self.training_config["batch_size"] = self.meta["batch_size"]
        self.training_config["epochs"] = self.meta["epochs"]

        return self.training_config

    def tuning(self) -> dict:
        if not self.tuning_mode:
            raise ValueError(f"No tuning configuration found in {self.config_path}!")

        self.tuning_cfg = {}

        self.tuning_cfg["hidden_channels"] = tuple(self.tuning_toml["hidden_channels"])
        self.tuning_cfg["block_groups"] = tuple(self.tuning_toml["block_groups"])
        self.tuning_cfg["block_name"] = self.tuning_toml["block_names"]
        self.tuning_cfg["activation_name"] = self.tuning_toml["activation_names"]
        self.tuning_cfg["optimizer"] = self.tuning_toml["optimizers"]
        self.tuning_cfg["lr"] = tuple(self.tuning_toml["lr"])
        self.tuning_cfg["momentum"] = tuple(self.tuning_toml["momentum"])
        self.tuning_cfg["weight_decay"] = tuple(self.tuning_toml["weight_decay"])

        return self.tuning_cfg
