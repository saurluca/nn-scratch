import numpy as np


def load_data_number_classifier(n_samples=10, seed=42):
    np.random.seed(seed)
    x = np.random.uniform(-10, 10, n_samples)
    # generates label 0 for negative 1 for positve
    y = np.maximum(0, np.sign(x))
    data = list(zip(x, y))
    return data


def load_data_sin_regression(n_samples=10, seed=42):
    np.random.seed(seed)
    x = np.random.uniform(0, 10, n_samples)
    y = np.sin(x)
    data = list(zip(x, y))
    return data
