"""Random Forest model"""

from pathlib import Path

import numpy as np
import sklearn
from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score


class RandomForest:
    def __init__(self, model_path) -> None:
        self.model_path = Path(model_path)

    def make_model(self, parameters, **kwargs) -> None:
        self.model = RandomForestClassifier(**parameters, **kwargs)
        self.parameters = self.model.get_params()

    def load_model(self) -> None:
        with open(self.model_path, "rb") as f:
            self.model = load(f)

    def get_model(self) -> sklearn.ensemble.RandomForestClassifier:
        return self.model

    def fit_model(self, X_train, y_train) -> None:
        self.model.fit(X_train, y_train)

    def fit(self, X_train, y_train) -> None:
        self.model.fit(X_train, y_train)

    def get_params(self, **kwargs):
        return self.model.get_params(**kwargs)

    def predict_model(self, X_test) -> np.ndarray:
        self.y_pred = self.model.predict(X_test)
        self.y_pred_proba = self.model.predict_proba(X_test)
        return self.y_pred, self.y_pred_proba

    def evaluate_model(self, y_test) -> float:
        ac_score = accuracy_score(y_test, self.y_pred) 
        ra_score = roc_auc_score(y_test, self.y_pred_proba, multi_class="ovo")
        return ac_score, ra_score

    def save_model(self) -> None:
        with open(self.model_path, "wb") as f:
            dump(self.model, f)
