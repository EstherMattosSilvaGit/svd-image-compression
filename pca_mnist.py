"""Runner script for PCA on MNIST using modular helpers."""
from __future__ import annotations
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

from data_utils import load_mnist, normalize, center
from pca_algo import covariance_matrix, top_k_eigenpairs, project


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-components", "-k", type=int, default=50)
    parser.add_argument("--sample", type=int, default=None, help="Use subset of MNIST for faster runs")
    parser.add_argument("--plot-2d", action="store_true", help="Plot 2D projection when k>=2")
    args = parser.parse_args()

    print("Loading MNIST...")
    t0 = time.time()
    X, y = load_mnist(sample=args.sample)
    print(f"Loaded X shape: {X.shape}; elapsed {time.time()-t0:.1f}s")

    X = normalize(X)
    Xc, mean = center(X)

    print("Computing covariance matrix...")
    t0 = time.time()
    C = covariance_matrix(Xc)
    print(f"Covariance computed; shape {C.shape}; elapsed {time.time()-t0:.1f}s")

    k = args.n_components
    if k <= 0 or k > C.shape[0]:
        raise ValueError("n_components must be between 1 and number of features")

    print(f"Computing top {k} eigenpairs (power iteration + deflation)...")
    t0 = time.time()
    eigvals, eigvecs = top_k_eigenpairs(C, k)
    print(f"Done; elapsed {time.time()-t0:.1f}s")

    total_variance = np.trace(C)
    explained = eigvals.sum() / total_variance if total_variance != 0 else 0.0
    print(f"Explained variance by top-{k}: {explained*100:.2f}%")

    components = eigvecs  # shape (k, n_features)
    X_proj = project(Xc, components)
    print(f"Projected shape: {X_proj.shape}")

    # Save results
    np.savez_compressed("pca_mnist_results.npz", X_proj=X_proj, eigvals=eigvals, components=components, mean=mean, y=y)
    print("Saved results to pca_mnist_results.npz")

    if args.plot_2d and k >= 2:
        print("Plotting 2D scatter (subset up to 5000 samples)...")
        nplot = min(5000, X_proj.shape[0])
        fig, ax = plt.subplots(figsize=(6, 5))
        sc = ax.scatter(X_proj[:nplot, 0], X_proj[:nplot, 1], c=y[:nplot], s=2, cmap="tab10", alpha=0.6)
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
        ax.set_title("MNIST projection onto first 2 principal components")
        plt.colorbar(sc, ticks=range(10), label="digit")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
