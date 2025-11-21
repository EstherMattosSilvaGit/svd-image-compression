"""Data loading and preprocessing helpers for MNIST."""
from __future__ import annotations
from typing import Tuple
import numpy as np
from sklearn.datasets import fetch_openml


def load_mnist(sample: int | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """Load MNIST (mnist_784) and optionally return a subset.

    Returns (X, y) where X is shape (n_samples, 784) and y are integer labels.
    """
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
    X = X.astype(np.float64)
    y = y.astype(np.int64)
    if sample is not None and sample < X.shape[0]:
        X = X[:sample]
        y = y[:sample]
    return X, y


def normalize(X: np.ndarray) -> np.ndarray:
    """Scale pixel values to [0, 1]."""
    return X / 255.0


def center(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Center X by subtracting the per-feature mean.

    Returns (X_centered, mean)
    """
    mean = X.mean(axis=0)
    return X - mean, mean
