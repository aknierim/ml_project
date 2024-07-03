from rolf.io.data import read_hdf5
from sklearn.model_selection import train_test_split
from pathlib import Path
import numpy as np


class LoadData:
    def __init__(self, data_path, random_state=None) -> None:
        self.data_path = Path(data_path)
        self.data = read_hdf5(self.data_path)
        if random_state == None:
            self.random_state = np.random.mtrand.RandomState()
        elif type(random_state) == int:
            self.random_state = np.random.mtrand.RandomState(random_state)
        elif type(random_state) == np.random.mtrand.RandomState:
            self.random_state = random_state
        else:
            raise TypeError(
                "random_state has to be None, int or numpy.random.mtrand.RandomState instance!"
            )

    def split_data(self, validation_ratio=0.1, test_ratio=0.05) -> None:
        self.X, self.y = self.data["img"], self.data["label"]
        self.X = self.X.reshape((self.X.shape[0], np.prod(self.X.shape[1:])))
        self.X = self.X / np.max(self.X)
        val_ratio = validation_ratio / (1 - test_ratio)

        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=test_ratio,
            shuffle=True,
            random_state=self.random_state,
        )
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp,
            y_temp,
            test_size=val_ratio,
            shuffle=True,
            random_state=self.random_state,
        )

    def get_data(self) -> np.ndarray:
        return (
            self.X_train,
            self.X_val,
            self.X_test,
            self.y_train,
            self.y_val,
            self.y_test,
        )
