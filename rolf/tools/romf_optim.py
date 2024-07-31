"""Starts the hyperparameter optimization for the random forest classifier ROMF"""

from pathlib import Path

import click
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

from rolf.romf.data import LoadData
from rolf.romf.hyper_search import SearchHyperparams

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
    default=(FILE_DIR / "../../build/romf_study.sqlite3").resolve(),
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
def main(input_file: str | Path, output_file: str | Path, seed: int):
    rng = np.random.mtrand.RandomState(seed)

    data = LoadData(input_file, rng)

    data.split_data(validation_ratio=0.2, test_ratio=0.2)

    X_train, X_val, X_test, y_train, y_val, y_test = data.get_data()

    search = SearchHyperparams(
        "./model.model", f"sqlite:///{output_file}", random_state=rng
    )

    search.estimators((50, 1500, 10))
    search.criterion(["gini", "entropy", "log_loss"])
    search.max_depth((10, 250, 10))
    search.min_samples_split((2, 20, 1))
    search.min_samples_leaf((1, 20, 1))
    search.max_features(["sqrt", "log2", None])
    search.bootstrap([True])
    search.class_weight([None])
    search.get_params()

    search.scorer(
        roc_auc_score, scorer_params={"multi_class": "ovo", "labels": [0, 1, 2, 3]}
    )
    search.cross_validate(KFold(n_splits=6, shuffle=True, random_state=rng))

    search.read_data(X_train, y_train)

    search.optimize(
        "double_sore",
        ["maximize", "maximize"],
        n_trials=250,
        n_jobs=1,
        n_forest_jobs=40,
    )


if __name__ == "__main__":
    main()
