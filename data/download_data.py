import os
import pickle

import numpy as np
import openml
from sklearn.preprocessing import LabelEncoder

openml.config.cache_directory = os.path.expanduser(os.getcwd() + "/openml_cache")


def save_suite(suite_id, dir_name, save_categorical_indicator=False, regression=True):
    benchmark_suite = openml.study.get_suite(suite_id)
    for task_id in benchmark_suite.tasks:
        task = openml.tasks.get_task(task_id)
        dataset = task.get_dataset()
        print(f"Downloading dataset {dataset.name}")
        X, y, categorical_indicator, _ = dataset.get_data(
            dataset_format="dataframe", target=dataset.default_target_attribute
        )
        X = np.array(X).astype(np.float32)
        if regression:
            y = np.array(y).astype(np.float32)
        else:
            le = LabelEncoder()
            y = le.fit_transform(np.array(y))
        with open(f"{dir_name}/data_{dataset.name}", "wb") as f:
            if save_categorical_indicator:
                pickle.dump((X, y, categorical_indicator), f)
            else:
                pickle.dump((X, y), f)


suites_id = {
    "numerical_regression": 297,
    "numerical_classification": 298,
    "categorical_regression": 299,
    "categorical_classification": 304,
}

print(f"Saving datasets from suite: {'numerical_regression'}")
save_suite(
    suites_id["numerical_regression"],
    "data/numerical_only/regression",
    save_categorical_indicator=False,
)

print(f"Saving datasets from suite: {'numerical_classification'}")
save_suite(
    suites_id["numerical_classification"],
    "data/numerical_only/balanced",
    save_categorical_indicator=False,
    regression=False,
)

print(f"Saving datasets from suite: {'categorical_regression'}")
save_suite(
    suites_id["categorical_regression"],
    "data/num_and_cat/regression",
    save_categorical_indicator=True,
)

print(f"Saving datasets from suite: {'categorical_classification'}")
save_suite(
    suites_id["categorical_classification"],
    "data/num_and_cat/balanced",
    save_categorical_indicator=True,
    regression=False,
)
