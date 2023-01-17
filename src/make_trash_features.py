import pickle

import numpy as np

if __name__ == "__main__":
    # numeric_only, num_and_cat
    cat_type = "numerical_only"
    # regression, balanced
    problem_type = "regression"
    dataset = "wine_quality"

    path = f"data/{cat_type}/{problem_type}/data_{dataset}"
    with open(path, "rb") as fin:
        X, y = pickle.load(fin)
    rng = np.random.RandomState(0)
    N = len(X)
    for fake_n in [1, 5, 20]:
        all_data = np.zeros((N, fake_n))
        for i in range(fake_n):
            if rng.random() > 0.5:
                loc, scale = rng.randint(0, 250), rng.randint(0, 50)
                data = rng.normal(loc, scale, size=N)
            else:
                l = rng.randint(0, 250)
                r = l + rng.randint(0, 50)
                data = rng.uniform(l, r, size=N)
            all_data[:, i] = data
        new_x = np.concatenate([X, all_data], axis=1)
        with open(path + f"_{fake_n}_trash", "wb") as fout:
            pickle.dump((new_x, y), fout)
