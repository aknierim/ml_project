"""Starts the training process."""

from pathlib import Path

import click
from rich.console import Console
from rich.progress import Progress

from rolf.io import ReadHDF5
from rolf.tools.toml_reader import ReadConfig
from rolf.training import train_model

FILE_DIR = Path(__file__).parent.resolve()
CONSOLE = Console()


@click.command()
@click.option(
    "--config-path",
    "-c",
    type=click.Path(exists=True, dir_okay=False, file_okay=True),
    help="Path to the config file.",
)
@click.option("--seed", "-s", type=int, help="Random state for the data split.")
@click.option(
    "--validation_ratio", type=float, help="Validation ratio for the data split."
)
@click.option("--test_ratio", type=float, help="Test ratio for the data split.")
@click.option(
    "--devices",
    "-d",
    type=str,
    default=1,
    show_default=True,
    help="""Number of devices or list of device IDs.
    Example 1: `-d 1` selects one GPU. \t\t
    Example 2: `-d '[0]'` selects GPU 0 \t\t
    Example 3: `-d '[0, 1]'` selects GPUS 0 and 1
    """,
)
def main(
    config_path: str | Path,
    seed: int,
    validation_ratio: float,
    test_ratio: float,
    devices: str,
):
    devices = eval(devices)

    if isinstance(devices, int):
        pass
    elif isinstance(devices, list):
        if isinstance(devices[0], int):
            pass
        pass
    else:
        raise TypeError(
            "Unsupported type for devices! "
            "Try 'rolf-train --help' for more information."
        )

    train_config = _get_config(config_path)
    data = _dataset(config_path, train_config, seed, test_ratio, validation_ratio)
    model, results = _training(train_config, data, devices)

    print("")
    CONSOLE.print(f"{_status('info')}[bold blue] Results:")
    CONSOLE.print(f"{_status('info')} {results}")
    print("")


def _get_config(config_path: str | Path) -> dict:
    """Read config file at config path.

    Parameters
    ----------
    config_path : str or Path
        Path to the config file.

    Returns
    -------
    train_config : dict
        Dictionary containing the training config.
    """
    if config_path is None:
        config_path = FILE_DIR / "../../configs/full_train.toml"
        config_path = _default_setter(config_path.absolute().resolve(), "config path")

    config = ReadConfig(config_path)
    train_config = config.training()

    print("")
    CONSOLE.print(f"{_status('info')}[bold blue] Training config:")
    CONSOLE.print(train_config)

    return train_config


def _dataset(
    config_path: str | Path,
    train_config: dict,
    seed: int,
    validation_ratio: float,
    test_ratio: float,
) -> ReadHDF5:
    """Creates a ReadHDF5 data object. Applies splits to the data.

    Parameters
    ----------
    config_path : str or Path
        Path to the config file.
    train_config : dict
        Dictionary containing the training config.
    seed : int
        Seed for the data split.
    validation_ratio : float
        Validation ratio of the data split.
    test_ratio : float
        Test ratio of the data split.

    Returns
    -------
    data : ReadHDF5 instance
        ReadHDF5 instance with data splits applied.
    """
    print("")
    CONSOLE.print(f"{_status('info')}[bold blue] Loading data:")

    if seed is None:
        seed = _default_setter(423, "seed")

    if test_ratio is None:
        test_ratio = _default_setter(0.2, "test ratio")

    if validation_ratio is None:
        validation_ratio = _default_setter(0.2, "validation ratio")

    data_path = train_config["paths"]["data"].absolute().resolve()
    data_path /= "galaxy_data_h5.h5"
    if data_path.is_file():
        data = ReadHDF5(
            data_path,
            random_state=seed,
            validation_ratio=validation_ratio,
            test_ratio=test_ratio,
        )
        with Progress() as progress:
            task = progress.add_task(f"{_status('info')} Creating dataset", total=3)
            data.make_transformer(progress=progress, task=task)
    else:
        raise FileNotFoundError(
            f"Please make sure that the file '{data_path}' exists! "
            f"Check the key 'data' in the config file '{config_path}'!"
        )

    return data


def _training(train_config: dict, data: ReadHDF5, devices: int | list) -> tuple:
    """Starts the actual training process.

    Parameters
    ----------
    train_config: dict
        Dictionary containing the training config.
    data : ReadHDF5 instance
        ReadHDF5 instance with data splits applied.
    devices : int or list
        Number of devices or list of specific device IDs.

    Returns
    -------
    tuple
        Tuple of model instance and dict with results.
    """
    if isinstance(devices, int):
        msg = f"on {devices} device(s)"
    elif isinstance(devices, list):
        msg = "on devices "
        n_devices = len(devices)
        for idx, device in enumerate(devices):
            if idx == (n_devices - 1):
                msg += f"and {device}"
            else:
                msg += f"{device}, "

    print("")
    CONSOLE.print(f"{_status('info')}[bold blue] Starting training process {msg}:")

    result = None
    batch_size = train_config["batch_size"]
    data_path = train_config["paths"]["data"].absolute().resolve()
    data_path /= "galaxy_data/all"

    ckpt_path = train_config["paths"]["model"]

    while result is None:
        try:
            train_loader, val_loader, test_loader = data.create_data_loaders(
                batch_size=batch_size, img_dir=data_path
            )
            model, result, _ = train_model(
                train_config["model_name"],
                train_loader,
                val_loader,
                test_loader,
                checkpoint_path=ckpt_path,
                epochs=train_config["epochs"],
                save_name=train_config["save_name"],
                model_hparams=train_config["net_hyperparams"],
                optimizer_name=train_config["optimizer"],
                optimizer_hparams=train_config["opt_hyperparams"],
                devices=devices,
                lr_scheduler="cyclic",
                print_model=True,
            )
        except Exception as e:
            CONSOLE.print(f"{_status('err')} {e} Reducing batch size")
            if batch_size > 10:
                batch_size -= 5
            else:
                batch_size = int(batch_size / 2)

            CONSOLE.print(f"{_status('info')} New batch size: {batch_size}")

    return model, result


def _default_setter(val, name):
    """Helper method to set default values if
    no value is provided.
    """
    info_msg = f"{_status('warn')} No {name} provided! "
    info_msg += f"Fallback to default value: '{val}'"

    CONSOLE.print(info_msg)

    return val


def _status(state):
    """Sets status messages."""
    match state:
        case "info":
            return "[cyan][INFO][/cyan]"
        case "warn":
            return "[yellow][WARN][/yellow]"
        case "err":
            return "[red][ERROR][/red]"


if __name__ == "__main__":
    main()
