import datetime
import uuid
from pathlib import Path

import click
import tomli


@click.command()
@click.option(
    "--config_path",
    "-c",
    type=click.Path(exists=False, dir_okay=True),
    help="Path to save the model to.",
)
def main(config_path):
    """Start training of the network."""

    if not config_path:
        raise ValueError("Please provide a path for the configuration file!")

    with open(config_path, "rb") as f:
        config = tomli.load(f)

    print(config)

    model_path = Path(config["paths"]["model"]).absolute()

    model_id = str(uuid.uuid1().hex)
    model_id += datetime.datetime.now().strftime("-%Y-%m-%d-T%H%M%S")
    model_path /= Path(model_id + ".model")
    print(model_path)


if __name__ == "__main__":
    main()
