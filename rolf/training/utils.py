import sys
from pathlib import Path

import click
import pandas as pd


def save_on_interrupt(model_path):
    """Utility function that saves the model
    on keyboard interrupt.
    """

    interrupt = click.confirm(
        "KeyboardInterrupt: Do you wish to stop training and save the model?",
        default=False,
        show_default=True,
        abort=False,
    )

    if interrupt:
        print(f"Saving model to file {model_path} after {'EPOCHS'} epoch(s).")

    sys.exit(1)


def get_train_test_dataset(path: str | Path):
    """Get train/test datasets.

    Parameters
    ----------
    path : str or Path
        Path to the full dataset.
    """
    if not isinstance(path, Path):
        path = Path(path)

    try:
        data = pd.read_hdf(path)
    except Exception as e:
        print(f"{e}: Fallback to pd.read_csv!")
        data = pd.read_csv(path)

    return data
