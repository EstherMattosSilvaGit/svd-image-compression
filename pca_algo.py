"""PCA algorithm primitives implemented with NumPy (no high-level eig/svd calls).

Functions:
 - covariance_matrix
 - power_iteration
 - top_k_eigenpairs
 - project
"""
from __future__ import annotations
import numpy as np


def covariance_matrix(X_centered: np.ndarray) -> np.ndarray:
    """Return the covariance matrix (features x features).

    Cov = (X^T X) / (n - 1)
    """
    n = X_centered.shape[0]
    return (X_centered.T @ X_centered) / float(n - 1)


def power_iteration(A: np.ndarray, max_iter: int = 1000, tol: float = 1e-8) -> tuple[float, np.ndarray]:
    """Compute the dominant eigenpair (lambda, v) of symmetric matrix A using power iteration."""
    n = A.shape[0]
    b = np.random.randn(n)
    b = b / np.linalg.norm(b)
    eigenvalue = None
    for _ in range(max_iter):
        Ab = A @ b
        norm_Ab = np.linalg.norm(Ab)
        if norm_Ab == 0:
            return 0.0, b
        b_next = Ab / norm_Ab
        lambda_est = float(b_next @ (A @ b_next))
        if eigenvalue is not None and abs(lambda_est - eigenvalue) < tol:
            eigenvalue = lambda_est
            b = b_next
            break
        eigenvalue = lambda_est
        b = b_next
    return eigenvalue, b


def top_k_eigenpairs(A: np.ndarray, k: int, max_iter: int = 1000, tol: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
    """Compute the top-k eigenvalues and eigenvectors using power iteration + deflation.

    Returns (eigvals, eigvecs) where eigvals.shape = (k,) and eigvecs.shape = (k, n_features).
    """
    A_work = A.copy()
    n = A.shape[0]
    eigvals = []
    eigvecs = []
    for i in range(k):
        val, vec = power_iteration(A_work, max_iter=max_iter, tol=tol)
        if val is None:
            break
        eigvals.append(val)
        eigvecs.append(vec)
        # Deflation
        A_work = A_work - val * np.outer(vec, vec)
    if eigvecs:
        eigvecs = np.vstack(eigvecs)
    else:
        eigvecs = np.empty((0, n))
    eigvals = np.array(eigvals)
    return eigvals, eigvecs


def project(X_centered: np.ndarray, components: np.ndarray) -> np.ndarray:
    """Project centered data onto principal components.

    `components` is expected with shape (k, n_features).
    Returns projected data shape (n_samples, k).
    """
    return X_centered @ components.T
