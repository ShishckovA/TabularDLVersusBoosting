import pickle

import numpy as np


def import_real_data(
    keyword=None,
    path_to_dir="../data",
) -> tuple[np.ndarray, np.ndarray]:
    with open(
        f"{path_to_dir}/numerical_only/regression/data_{keyword}",
        "rb",
    ) as f:
        X, y = pickle.load(f)

    return np.array(X), np.array(y)
