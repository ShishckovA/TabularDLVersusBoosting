import pickle

import numpy as np
from scipy.stats import special_ortho_group

if __name__ == "__main__":
    # numeric_only, num_and_cat
    cat_type = "numerical_only"
    # regression, balanced
    problem_type = "regression"
    for dataset in ["wine_quality", "wine_quality_20_trash", "fifa", "fifa_20_trash"]:
        path = f"data/{cat_type}/{problem_type}/data_{dataset}"
        with open(path, "rb") as fin:
            X, y = pickle.load(fin)
        rng = np.random.RandomState(42)
        N = len(X)

        num_samples, num_features = X.shape
        rotation_matrix = special_ortho_group.rvs(num_features, random_state=rng)

        new_x = X @ rotation_matrix
        with open(path + "_rotated", "wb") as fout:
            pickle.dump((new_x, y), fout)
