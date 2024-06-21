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
        self.eval = self.toml["eval"]

    def training(self) -> dict:
        self.training_config = {}
        self.training_config["paths"] = {
            "data": Path(self.paths["data"]),
            "model": Path(self.paths["model"]),
        }
        self.training_config["mode"] = self.mode
        self.training_config["parameters"] = self.meta

        return self.training_config

    def validation(self) -> dict:
        self.validation_config = self.eval

        return self.validation_config
