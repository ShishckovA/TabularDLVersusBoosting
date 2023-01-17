import traceback

import numpy as np
import optuna
import torch
import wandb
from catboost import CatBoostRegressor, Pool
from optuna.samplers import TPESampler
from optuna.visualization import plot_optimization_history
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from wandb.wandb_run import Run

from generate_data import import_real_data
from mlp import create_mlp
from resnet import create_resnet_skorch


def objective_tabnet(trial: optuna.Trial) -> float:
    param = {
        "n_d": trial.suggest_int("n_d", 2, 12),
        "n_a": trial.suggest_int("n_a", 2, 12),
        "n_steps": trial.suggest_int("n_steps", 3, 10),
        "optimizer_params": {
            "lr": trial.suggest_float("optimizer_params__lr", 1e-4, 1e-1, log=True)
        },
        "scheduler_params": {
            "step_size": trial.suggest_int("step_size", 5, 50),
            "gamma": trial.suggest_float("gamma", 0.1, 1),
        },
        "scheduler_fn": torch.optim.lr_scheduler.StepLR,
        "seed": 42,
        "verbose": 0,
    }

    model = TabNetRegressor(**param)

    model.fit(X_train, y_train.reshape(-1, 1), eval_set=[(X_val, y_val.reshape(-1, 1))])
    return r2_score(y_test, model.predict(X_test))  # type: ignore


def objective_boosting(trial: optuna.Trial) -> float:
    param = {
        "objective": trial.suggest_categorical("objective", ["RMSE", "MAPE", "MAE"]),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.01, 0.1),
        "depth": trial.suggest_int("depth", 1, 12),
        "boosting_type": trial.suggest_categorical(
            "boosting_type", ["Ordered", "Plain"]
        ),
        "bootstrap_type": trial.suggest_categorical(
            "bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]
        ),
        "n_estimators": trial.suggest_int("n_estimators", 10, 10000, log=True),
        "early_stopping_rounds": trial.suggest_int(
            "early_stopping_rounds", 10, 500, log=True
        ),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),
        "used_ram_limit": "80gb",
        "verbose": False,
        "use_best_model": True,
        "random_state": 42,
    }

    if param["bootstrap_type"] == "Bayesian":
        param["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 10)
    elif param["bootstrap_type"] == "Bernoulli":
        param["subsample"] = trial.suggest_float("subsample", 0.1, 1)

    train_pool = Pool(X_train, y_train)
    val_pool = Pool(X_val, y_val)
    test_pool = Pool(X_test, y_test)

    model = CatBoostRegressor(**param)

    model.fit(train_pool, eval_set=val_pool)
    return r2_score(y_test, model.predict(test_pool))  # type: ignore


def objective_resnet(trial: optuna.Trial) -> float:
    resnet_config = {
        "optimizer": trial.suggest_categorical("optimizer", ["adamw", "sgd", "adam"]),
        "batch_size": 512,
        "max_epochs": 100,
        "module__activation": "reglu",
        "module__normalization": "batchnorm",
        "module__n_layers": trial.suggest_int("module__n_layers", 2, 10),
        "module__d": trial.suggest_int("module__d", 16, 512, log=True),
        "module__d_hidden_factor": trial.suggest_int("module__d_hidden_factor", 1, 16),
        "module__hidden_dropout": trial.suggest_float("module__hidden_dropout", 0, 1),
        "module__residual_dropout": trial.suggest_float(
            "module__residual_dropout", 0, 1
        ),
        "lr": trial.suggest_float("lr", 1e-4, 1e-1, log=True),
        "optimizer__weight_decay": 1e-7,
        "module__d_embedding": trial.suggest_int(
            "module__d_embedding", 64, 512, log=True
        ),
        "lr_sched_step": trial.suggest_int("lr_sched_step", 5, 50),
        "lr_sched_gamma": trial.suggest_float("lr_sched_gamma", 0.1, 1),
        "device": "cuda",
    }
    model = create_resnet_skorch(**resnet_config)
    try:
        model.fit(X_train_val.astype(np.float32), y_train_val.reshape(-1, 1))
        return r2_score(y_test, model.predict(X_test.astype(np.float32)))  # type: ignore
    except Exception:
        traceback.print_exc()
        return -1543


def objective_mlp(trial: optuna.Trial) -> float:
    mlp_config = {
        "module__input_size": X_train.shape[1],
        "optimizer": trial.suggest_categorical("optimizer", ["adamw", "sgd", "adam"]),
        "batch_size": 512,
        "max_epochs": 100,
        "module__activation": trial.suggest_categorical(
            "module__activation", ["relu", "gelu", "sigmoid"]
        ),
        "module__dropout": trial.suggest_float("module__dropout", 0, 1),
        "module__n_hidden": trial.suggest_int("module__n_hidden", 1, 6),
        "module__hidden_size": trial.suggest_int(
            "module__hidden_size", 16, 256, log=True
        ),
        "lr": trial.suggest_float("lr", 1e-4, 1e-1, log=True),
        "optimizer__weight_decay": 1e-7,
        "lr_sched_step": trial.suggest_int("lr_sched_step", 5, 50),
        "lr_sched_gamma": trial.suggest_float("lr_sched_gamma", 0.1, 1),
        "module__use_batch_norm": trial.suggest_categorical(
            "module__use_batch_norm", [False, True]
        ),
        "device": "cuda",
        "verbose": False,
    }
    model = create_mlp(mlp_config)
    try:
        model.fit(X_train_val.astype(np.float32), y_train_val.reshape(-1, 1))
        return r2_score(y_test, model.predict(X_test.astype(np.float32)))  # type: ignore
    except Exception:
        traceback.print_exc()
        return -1543


if __name__ == "__main__":
    MODEL = "boosting"
    for ds in ["fifa", "wine_quality"]:
        for suffix in ["", "_1_trash", "_5_trash", "_20_trash"]:
            suffix2_list = (
                [""] if suffix in ["_1_trash", "_5_trash"] else ["", "_rotated"]
            )
            for suffix2 in suffix2_list:
                # dataset = "fifa_20_trash"
                # tabnet, boosting, resnet, mlp
                # MODEL = "mlp"
                DATASET = f"{ds}{suffix}{suffix2}"
                ROUNDS = 100
                print(f"Starting to optimize\n{DATASET=}\n{MODEL=}\n{ROUNDS=}")
                X: np.ndarray
                y: np.ndarray
                X_train: np.ndarray
                y_train: np.ndarray
                X_val: np.ndarray
                y_val: np.ndarray
                X_test: np.ndarray
                y_test: np.ndarray
                X_train_val: np.ndarray
                y_train_val: np.ndarray

                X, y = import_real_data(DATASET, path_to_dir="data")
                print(X.shape, y.shape)
                if MODEL in ["tabnet", "resnet", "mlp"]:
                    qt = QuantileTransformer(random_state=42)
                    X = qt.fit_transform(X)

                X_train_val, X_test, y_train_val, y_test = train_test_split(
                    X, y, random_state=42
                )  # type: ignore

                X_train, X_val, y_train, y_val = train_test_split(
                    X_train_val, y_train_val, random_state=42
                )  # type: ignore

                sampler = TPESampler(seed=42)
                study = optuna.create_study(direction="maximize", sampler=sampler)
                if MODEL == "tabnet":
                    study.optimize(objective_tabnet, n_trials=ROUNDS)
                if MODEL == "boosting":
                    study.optimize(objective_boosting, n_trials=ROUNDS)
                if MODEL == "resnet":
                    study.optimize(objective_resnet, n_trials=ROUNDS)
                if MODEL == "mlp":
                    study.optimize(objective_mlp, n_trials=ROUNDS)

                print(f"Number of finished trials: {len(study.trials)}")

                print("Best trial:")
                trial = study.best_trial

                print(f"  Value: {trial.value}")

                print("  Params: ")
                for key, value in trial.params.items():
                    print(f"    {key}: {value}")

                summary: Run = wandb.init(
                    project="tabular_final",
                    name=f"{DATASET}_{MODEL}",
                    entity="shishckova",
                    job_type="logging",
                    reinit=True,
                    dir="wandb_logs",
                )  # type: ignore

                for step, trial in enumerate(study.trials):
                    summary.log({"r2": trial.value}, step=step)
                    for k, v in trial.params.items():
                        summary.log({k: v}, step=step)

                fig = plot_optimization_history(study)
                ymax = max(study.trials, key=lambda x: x.value).value  # type: ignore
                fig.update_yaxes(range=[0, ymax * 1.1])
                fig.write_image(f"images_final/{DATASET}_{MODEL}_{ymax:.4f}.png")
                summary.finish()
