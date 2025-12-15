# src/data_utils.py

import os
import numpy as np


def get_project_root():
    """
    Returns the absolute path to the project root directory.
    Assumes this file is in <root>/src/.
    """
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def load_benchmark_dataset(
    filename: str = "har_benchmark_windows.npz",
):
    """
    Load a benchmark dataset for testing the secure inference wrapper.

    Parameters
    ----------
    filename : str
        Name of the .npz file inside the data/ directory.

    Returns
    -------
    X_bench : np.ndarray
        Array of windows with shape (num_windows, window_length, num_channels).
    y_bench : np.ndarray
        Array of integer labels with shape (num_windows,).
    """
    project_root = get_project_root()
    data_dir = os.path.join(project_root, "data")
    path = os.path.join(data_dir, filename)

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Benchmark file not found: {path}.\n"
            f"Please make sure '{filename}' is placed in the 'data' folder."
        )

    data = np.load(path)
    X_bench = data["X_bench"]
    y_bench = data["y_bench"]
    return X_bench, y_bench
