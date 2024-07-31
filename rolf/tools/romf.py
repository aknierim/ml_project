from pathlib import Path

import click
import numpy as np

from rolf.romf.data import LoadData
from rolf.romf.model import RandomForest

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
    default=(FILE_DIR / "../../build/romf.dump").resolve(),
    show_default=True,
    help="""Output path and filename for sqlite database.
    File extension has to be .sqlite3""",
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
def main(
    input_file: str | Path,
    output_file: str | Path,
    seed: int,
    validation_ratio: float,
    test_ratio: float,
) -> None:
    rng = np.random.mtrand.RandomState(seed=seed)

    data = LoadData(str(input_file), rng)

    data.split_data(validation_ratio=validation_ratio, test_ratio=test_ratio)

    parameters = {
        "bootstrap": True,
        "ccp_alpha": 0.0,
        "class_weight": None,
        "criterion": "entropy",
        "max_depth": 70,
        "max_features": None,
        "max_leaf_nodes": None,
        "max_samples": None,
        "min_impurity_decrease": 0.0,
        "min_samples_leaf": 2,
        "min_samples_split": 9,
        "min_weight_fraction_leaf": 0.0,
        "monotonic_cst": None,
        "n_estimators": 300,
        "oob_score": False,
        "random_state": None,
        "verbose": 0,
        "warm_start": False,
    }

    X_train, X_val, X_test, y_train, y_val, y_test = data.get_data()

    rf = RandomForest(str(output_file))

    try:
        rf.load_model()
    except FileNotFoundError:
        rf.make_model(parameters=parameters, n_jobs=-1)
        rf.fit(X_train, y_train)
        rf.save_model()

    rf.predict_model(X_test)
    accuracy, roc_auc = rf.evaluate_model(y_test)

    print(f"The model has an accuracy_score of {accuracy:.3f},")
    print(f"and a roc_auc_score of {roc_auc:.3f}!")


if __name__ == "__main__":
    main()
