from rolf.romf.data import LoadData
from rolf.romf.hyper_search import SearchHyperparams
import numpy as np
from optuna.study import Study
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import KFold

rng = np.random.mtrand.RandomState(423)

data = LoadData("../data/galaxy_data_h5.h5", rng)

data.split_data(validation_ratio=0.2, test_ratio=0.2)

X_train, X_val, X_test, y_train, y_val, y_test = data.get_data()

search = SearchHyperparams(
    "./model.model", "sqlite:///romf_study.sqlite3", random_state=rng
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

search.scorer(roc_auc_score, scorer_params={'multi_class':"ovo", 'labels':[0,1,2,3]})
search.cross_validate(KFold(n_splits=6, shuffle=True, random_state=rng))

search.read_data(X_train, y_train, X_val, y_val)

search.optimize("double_sore", ["maximize", "maximize"], n_trials=250, n_jobs=1, n_forest_jobs=40)
