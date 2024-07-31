import numpy as np

from rolf.romf.model import RandomForest
from rolf.romf.data import LoadData

rng = np.random.mtrand.RandomState(seed=423)

data = LoadData("../../data/galaxy_data_h5.h5", rng)

data.split_data(validation_ratio=0.2, test_ratio=0.2)

parameters = {'bootstrap': True,
    'ccp_alpha': 0.0,
    'class_weight': None,
    'criterion': 'entropy',
    'max_depth': 70,
    'max_features': None,
    'max_leaf_nodes': None,
    'max_samples': None,
    'min_impurity_decrease': 0.0,
    'min_samples_leaf': 2,
    'min_samples_split': 9,
    'min_weight_fraction_leaf': 0.0,
    'monotonic_cst': None,
    'n_estimators': 300,
    'oob_score': False,
    'random_state': None,
    'verbose': 0,
    'warm_start': False}

X_train, X_val, X_test, y_train, y_val, y_test = data.get_data()

rf = RandomForest("../../trained_models/romf.dump")

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
