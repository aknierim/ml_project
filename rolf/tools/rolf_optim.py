from pathlib import Path

import click
from rich.pretty import pprint

from rolf.optim.rolfo import ParameterOptimization

FILE_DIR = Path(__file__).parent.resolve()


@click.command()
@click.option(
    "--input-file",
    "-i",
    type=click.Path(exists=True, dir_okay=False, file_okay=True),
    default=(FILE_DIR / "../../data/galaxy_data_h5.h5").resolve(),
    show_default=True,
    help="H5 file containing the data table.",
)
@click.option(
    "--output-file",
    "-o",
    type=click.Path(exists=False, dir_okay=False, file_okay=True),
    default=(FILE_DIR / "../../build/rolf_study.sqlite3").resolve(),
    show_default=True,
    help="""Output path and filename for sqlite database.
    File extension has to be .sqlite3""",
)
@click.option(
    "--config-file",
    "-c",
    type=click.Path(exists=True, dir_okay=False, file_okay=True),
    default=(FILE_DIR / "../../configs/resnet_tuning.toml").resolve(),
    show_default=True,
    help="H5 file containing the data table.",
)
@click.option(
    "--data-dir",
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
    default=(FILE_DIR / "../../data/galaxy_data/all/").resolve(),
    show_default=True,
    help="Base directory containing the subdirectories of images.",
)
@click.option(
    "--seed",
    "-s",
    type=int,
    default=423,
    show_default=True,
    help="Random state for the data split.",
)
@click.option(
    "--validation_ratio",
    type=float,
    default=0.2,
    show_default=True,
    help="Validation ratio for the data split.",
)
@click.option(
    "--test_ratio",
    type=float,
    default=0.2,
    show_default=True,
    help="Test ratio for the data split.",
)
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
@click.option(
    "--study-name",
    type=str,
    default="ROLF_study",
    show_default=True,
    help="Name of the study shown in the sqlite database.",
)
@click.option(
    "--n_trials",
    type=int,
    default=100,
    show_default=True,
    help="Number of trials to run.",
)
@click.option(
    "--n_jobs",
    type=int,
    default=1,
    show_default=True,
    help="Number of workers to run in parallel.",
)
def main(
    input_file: str | Path,
    output_file: str | Path,
    config_file: str | Path,
    data_dir: str | Path,
    seed: int,
    validation_ratio: float,
    test_ratio: float,
    devices: int | list,
    study_name: str,
    n_trials: int,
    n_jobs: int,
) -> None:
    optimizer = ParameterOptimization(
        optim_conf_path=str(config_file),
        optuna_path=f"sqlite:///{output_file}",
        data_path=str(input_file),
        random_state=seed,
        validation_ratio=0.2,
        test_ratio=0.2,
        devices=1,
    )

    optimizer.load_data(data_dir)

    pprint(optimizer.model_config)
    pprint(optimizer.tuning_config)

    optimizer.optimize(
        study_name, ["maximize", "maximize"], n_trials=n_trials, n_jobs=n_jobs
    )


if __name__ == "__main__":
    main()
