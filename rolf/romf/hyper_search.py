from collections.abc import Callable
from pathlib import Path

import numpy as np
from optuna import Trial, create_study
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, make_scorer

from rolf.romf._validation import cross_val_score


class SearchHyperparams:
    def __init__(self, forest_path, optuna_path, random_state=None) -> None:
        self.params = {}
        self.forest_path = Path(forest_path)
        self.optuna_path = optuna_path
        self.cv = None
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

    def estimators(self, value) -> None:
        self.params["n_estimators"] = value

    def criterion(self, categorial) -> None:
        self.params["criterion"] = categorial

    def max_depth(self, value) -> None:
        self.params["max_depth"] = value

    def min_samples_split(self, value) -> None:
        self.params["min_samples_split"] = value

    def min_samples_leaf(self, value) -> None:
        self.params["min_samples_leaf"] = value

    def max_features(self, categorial) -> None:
        self.params["max_features"] = categorial

    def bootstrap(self, categorial) -> None:
        self.params["bootstrap"] = categorial

    def class_weight(self, categorial) -> None:
        self.params["class_weight"] = categorial

    def make_forest(self, trial: Trial, n_forest_jobs: int) -> None:
        self.use = {}
        for key in self.params:
            pars = self.params[key]
            if type(pars) == tuple:
                self.use[key] = trial.suggest_int(
                    key, low=pars[0], high=pars[1], step=pars[2]
                )
            elif isinstance(pars, list):
                self.use[key] = trial.suggest_categorical(key, pars)
            else:
                self.use[key] = pars

        self.rf = RandomForestClassifier(
            **self.use, n_jobs=n_forest_jobs, random_state=self.random_state
        )

    def scorers(self, score1, score2, scorer_params1=None, scorer_params2=None, make1={}, make2={}) -> None:
        self.scorer_params1 = scorer_params1
        self.scorer1 = make_scorer(score1, **make1)
        self.scorer_params2 = scorer_params2
        self.scorer2 = make_scorer(score2, **make2)

    def cross_validate(self, function: Callable) -> None:
        self.cv = function

    def get_params(self) -> dict:
        return self.params

    def objective(self, trial, X, y, n_forest_jobs) -> float:
        self.make_forest(trial, n_forest_jobs)
        scores = cross_val_score(
            self.rf,
            X,
            y,
            scoring=self.scorer1,
            cv=self.cv,
            score_params=self.scorer_params1,
        )
        accu = cross_val_score(
            self.rf,
            X,
            y,
            scoring=self.scorer2,
            cv=self.cv,
            score_params=self.scorer_params2,
        )
        return np.min([np.mean(scores), np.median(scores)]), np.min(
            [np.mean(accu), np.median(accu)]
        )
        
    def read_data(self, X_train, y_train) -> None:
        self.X_train = X_train
        self.y_train = y_train

    def optimize(self, study_name, direction, n_trials, n_jobs, n_forest_jobs) -> None:
        self.study = create_study(
            study_name=study_name,
            directions=direction,
            storage=self.optuna_path,
            load_if_exists=True,
        )
        self.study.optimize(
            lambda trial: self.objective(
                trial, self.X_train, self.y_train, n_forest_jobs
            ),
            n_trials=n_trials,
            n_jobs=n_jobs,
            show_progress_bar=True,
            gc_after_trial=True,
        )
