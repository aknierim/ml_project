from rich.pretty import pprint

from rolf.optim.rolfo import ParameterOptimization


def main():
    optimizer = ParameterOptimization(
        optim_conf_path="../configs/resnet_tuning.toml",
        optuna_path="sqlite:////cephfs/projects/RADIO/radio_ml/resnet2.sqlite3",
        data_path="../data/galaxy_data_h5.h5",
        random_state=423,
        validation_ratio=0.2,
        test_ratio=0.2,
    )

    optimizer.load_data("../data/galaxy_data/all/")

    pprint(optimizer.model_config)
    pprint(optimizer.tuning_config)

    optimizer.optimize(
        "Optim_AUC_ACC", ["maximize", "maximize"], n_trials=100, n_jobs=1
    )


if __name__ == "__main__":
    main()
