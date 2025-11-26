"""
PCA implementado manualmente aplicado ao dataset MNIST
Uso: python src/pca_mnist.py [--k 50] [--max-samples 5000]

Este script executa:
- Download dos arquivos MNIST (idx) se não existirem
- Leitura do dataset (train/test)
- Cálculo de PCA via SVD (usando numpy)
- Projeção das imagens nos primeiros k componentes
- Reconstrução com k componentes e cálculo de erro (MSE)
- Plots: variância explicada, imagens originais x reconstruídas, projeção 2D (PC1 vs PC2)

Dependências: numpy, matplotlib (somente)
"""

from __future__ import annotations
import os
import urllib.request
import gzip
import struct
from pathlib import Path
_np_module = None
_plt_module = None

def _import_numpy(strict: bool = True):
    """Lazy import of numpy to avoid importing it at module import time.
    If strict is True, raise ImportError when numpy isn't available; otherwise return None.
    """
    global _np_module
    if _np_module is not None:
        return _np_module
    try:
        import numpy as _np
        _np_module = _np
        return _np_module
    except Exception:
        if strict:
            raise
        return None
def _import_matplotlib(strict: bool = True):
    """Lazy import of matplotlib.pyplot to avoid importing it at module import time."""
    global _plt_module
    if _plt_module is not None:
        return _plt_module
    try:
        import matplotlib.pyplot as _plt
        _plt_module = _plt
        return _plt_module
    except Exception:
        if strict:
            raise
        return None
import argparse
from typing import Tuple, Any
import math
import random

MNIST_URLS = {
    "train_images": "https://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
    "train_labels": "https://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
    "test_images": "https://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
    "test_labels": "https://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz",
}


def download_mnist(dest_dir: str = "data"):
    os.makedirs(dest_dir, exist_ok=True)
    for name, url in MNIST_URLS.items():
        filename = os.path.join(dest_dir, os.path.basename(url))
        if not os.path.exists(filename):
            print(f"Baixando {url} -> {filename}...")
            urllib.request.urlretrieve(url, filename)


def read_idx_images(filepath: str) -> Any:
    np = _import_numpy(strict=True)
    with gzip.open(filepath, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        assert magic == 2051, f"Magic number incorreto: {magic} para {filepath}"
        buf = f.read(rows * cols * num_images)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = data.reshape(num_images, rows * cols)
        return data


def read_idx_labels(filepath: str) -> Any:
    np = _import_numpy(strict=True)
    with gzip.open(filepath, 'rb') as f:
        magic, num_labels = struct.unpack('>II', f.read(8))
        assert magic == 2049, f"Magic number incorreto: {magic} para {filepath}"
        buf = f.read(num_labels)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
        return labels


def load_mnist(dest_dir: str = 'data', kind: str = 'train', max_samples: int | None = None) -> Tuple[Any, Any]:
    assert kind in ('train', 'test')
    if not os.path.exists(dest_dir) or not any(os.path.exists(os.path.join(dest_dir, os.path.basename(u))) for u in MNIST_URLS.values()):
        download_mnist(dest_dir)

    if kind == 'train':
        images_path = os.path.join(dest_dir, os.path.basename(MNIST_URLS['train_images']))
        labels_path = os.path.join(dest_dir, os.path.basename(MNIST_URLS['train_labels']))
    else:
        images_path = os.path.join(dest_dir, os.path.basename(MNIST_URLS['test_images']))
        labels_path = os.path.join(dest_dir, os.path.basename(MNIST_URLS['test_labels']))

    images = read_idx_images(images_path)
    labels = read_idx_labels(labels_path)

    if max_samples is not None:
        images = images[:max_samples]
        labels = labels[:max_samples]

    # Normalize to [0,1]
        np = _import_numpy(strict=True)
        images = images.astype(np.float32) / 255.0

    return images, labels


def read_idx_images_pure(filepath: str) -> list[list[float]]:
    """Read IDX image file and return list of lists (N x D) normalized to [0,1] without numpy."""
    with gzip.open(filepath, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        assert magic == 2051, f"Magic number incorreto: {magic} para {filepath}"
        buf = f.read(rows * cols * num_images)
        # Convert bytes to list of floats
        images = []
        idx = 0
        for i in range(num_images):
            img = [b / 255.0 for b in buf[idx: idx + rows * cols]]
            images.append(img)
            idx += rows * cols
        return images


def read_idx_labels_pure(filepath: str) -> list[int]:
    with gzip.open(filepath, 'rb') as f:
        magic, num_labels = struct.unpack('>II', f.read(8))
        assert magic == 2049, f"Magic number incorreto: {magic} para {filepath}"
        buf = f.read(num_labels)
        labels = [int(b) for b in buf]
        return labels


def load_mnist_pure(dest_dir: str = 'data', kind: str = 'train', max_samples: int | None = None) -> Tuple[list[list[float]], list[int]]:
    """Load MNIST and return pure python lists for images and labels."""
    assert kind in ('train', 'test')
    if not os.path.exists(dest_dir) or not any(os.path.exists(os.path.join(dest_dir, os.path.basename(u))) for u in MNIST_URLS.values()):
        download_mnist(dest_dir)
    if kind == 'train':
        images_path = os.path.join(dest_dir, os.path.basename(MNIST_URLS['train_images']))
        labels_path = os.path.join(dest_dir, os.path.basename(MNIST_URLS['train_labels']))
    else:
        images_path = os.path.join(dest_dir, os.path.basename(MNIST_URLS['test_images']))
        labels_path = os.path.join(dest_dir, os.path.basename(MNIST_URLS['test_labels']))

    images = read_idx_images_pure(images_path)
    labels = read_idx_labels_pure(labels_path)
    if max_samples is not None:
        images = images[:max_samples]
        labels = labels[:max_samples]
    return images, labels



def compute_pca_full(X: Any) -> Tuple[Any, Any, Any]:
    """
    Compute full PCA via SVD on centered data X.
    Returns components (D x D or min(N,D) x D), explained_variance (min(N,D),), mean (D,)
    """
    np = _import_numpy(strict=True)
    N, D = X.shape
    mean = X.mean(axis=0)
    X_centered = X - mean

    # SVD
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    components_all = Vt  # shape (min(N, D), D)

    explained_variance_all = (S ** 2) / (N - 1)

    return components_all, explained_variance_all, mean


def compute_pca(X: Any, n_components: int, pca_full: tuple | None = None) -> Tuple[Any, Any, Any]:
    """Return top n_components using full PCA result if provided, otherwise compute once."""
    if pca_full is None:
        components_all, explained_variance_all, mean = compute_pca_full(X)
    else:
        components_all, explained_variance_all, mean = pca_full
    components = components_all[:n_components, :]
    explained_variance = explained_variance_all[:n_components]
    return components, explained_variance, mean


def project(X: Any, components: Any, mean: Any) -> Any:
    """Project samples X to PC space (scores). X: N x D, components: k x D"""
    np = _import_numpy(strict=True)
    Xc = X - mean
    scores = np.dot(Xc, components.T)
    return scores


def reconstruct(scores: Any, components: Any, mean: Any) -> Any:
    """Reconstruct X from scores and components"""
    np = _import_numpy(strict=True)
    recon = np.dot(scores, components) + mean
    return recon


### --- Pure Python linear algebra helpers and PCA implementation ---

def dot(u: list[float], v: list[float]) -> float:
    return sum(ui * vi for ui, vi in zip(u, v))


def add_vec(u: list[float], v: list[float]) -> list[float]:
    return [ui + vi for ui, vi in zip(u, v)]


def sub_vec(u: list[float], v: list[float]) -> list[float]:
    return [ui - vi for ui, vi in zip(u, v)]


def scalar_mul_vec(a: float, v: list[float]) -> list[float]:
    return [a * vi for vi in v]


def norm(v: list[float]) -> float:
    return math.sqrt(dot(v, v))


def mat_vec_mul(A: list[list[float]], v: list[float]) -> list[float]:
    return [dot(row, v) for row in A]


def outer(u: list[float], v: list[float]) -> list[list[float]]:
    return [[ui * vj for vj in v] for ui in u]


def mat_sub(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    return [[aij - bij for aij, bij in zip(ai, bi)] for ai, bi in zip(A, B)]


def mat_add(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    return [[aij + bij for aij, bij in zip(ai, bi)] for ai, bi in zip(A, B)]


def scalar_mul_mat(a: float, A: list[list[float]]) -> list[list[float]]:
    return [[a * aij for aij in ai] for ai in A]


def compute_mean_pure(X: list[list[float]]) -> list[float]:
    N = len(X)
    D = len(X[0])
    mean = [0.0] * D
    for i in range(N):
        for j in range(D):
            mean[j] += X[i][j]
    return [m / N for m in mean]


def center_data_pure(X: list[list[float]], mean: list[float]) -> list[list[float]]:
    return [[xij - mean[j] for j, xij in enumerate(xi)] for xi in X]


def covariance_matrix_pure(X_centered: list[list[float]]) -> list[list[float]]:
    # Computes covariance matrix (D x D) = (1/(N-1)) * X^T X
    N = len(X_centered)
    D = len(X_centered[0])
    cov = [[0.0] * D for _ in range(D)]
    for i in range(N):
        xi = X_centered[i]
        for j in range(D):
            xij = xi[j]
            rowj = cov[j]
            for l in range(D):
                rowj[l] += xij * xi[l]
    denom = N - 1 if N > 1 else 1
    for j in range(D):
        for l in range(D):
            cov[j][l] /= denom
    return cov


def power_iteration(A: list[list[float]], max_iter: int = 1000, tol: float = 1e-6) -> tuple[list[float], float]:
    D = len(A)
    v = [random.random() for _ in range(D)]
    vnorm = norm(v)
    v = [vi / vnorm for vi in v]
    for it in range(max_iter):
        w = mat_vec_mul(A, v)
        wnorm = norm(w)
        if wnorm == 0:
            break
        v_next = [wi / wnorm for wi in w]
        # check convergence
        if norm([vi - vni for vi, vni in zip(v, v_next)]) < tol:
            v = v_next
            break
        v = v_next
    # Rayleigh quotient as eigenvalue
    Av = mat_vec_mul(A, v)
    eigenvalue = dot(v, Av)
    return v, eigenvalue


def deflate(A: list[list[float]], eigenvalue: float, eigenvector: list[float]) -> None:
    # A <- A - eigenvalue * eigenvector * eigenvector^T
    D = len(A)
    for i in range(D):
        for j in range(D):
            A[i][j] -= eigenvalue * eigenvector[i] * eigenvector[j]


def compute_pca_pure(X: list[list[float]], n_components: int) -> tuple[list[list[float]], list[float], list[float]]:
    # Returns components (k x D), eigenvalues list length k, mean list length D
    mean = compute_mean_pure(X)
    Xc = center_data_pure(X, mean)
    cov = covariance_matrix_pure(Xc)
    components = []
    eigenvalues = []
    for _ in range(n_components):
        v, ev = power_iteration(cov, max_iter=1000, tol=1e-6)
        components.append(v)
        eigenvalues.append(ev)
        deflate(cov, ev, v)
    return components, eigenvalues, mean


def project_pure(X: list[list[float]], components: list[list[float]], mean: list[float]) -> list[list[float]]:
    # X: N x D, components: k x D -> scores N x k
    N = len(X)
    k = len(components)
    scores = [[0.0] * k for _ in range(N)]
    for i in range(N):
        xi = X[i]
        for j in range(k):
            comp = components[j]
            scores[i][j] = dot(sub_vec(xi, mean), comp)
    return scores


def reconstruct_pure(scores: list[list[float]], components: list[list[float]], mean: list[float]) -> list[list[float]]:
    N = len(scores)
    k = len(components)
    D = len(mean)
    recon = [[mean[j] for j in range(D)] for _ in range(N)]
    for i in range(N):
        for j in range(k):
            coef = scores[i][j]
            comp = components[j]
            for d in range(D):
                recon[i][d] += coef * comp[d]
    return recon


def mse_pure(X: list[list[float]], X_hat: list[list[float]]) -> float:
    N = len(X)
    D = len(X[0])
    total = 0.0
    for i in range(N):
        for j in range(D):
            diff = X[i][j] - X_hat[i][j]
            total += diff * diff
    return total / (N * D)



def show_pca_math_comments():
    """Função de demonstração que imprime os passos matriciais do PCA para fins didáticos.
    Use isso em uma apresentação ao professor para mostrar como se relacionam as matrizes U, S, Vt com PCA.
    """
    print("PCA - Passos principais (matricial):")
    print("1) Dados X\n2) Média por coluna μ = (1/N) Σ x_i\n3) Centraliza: X_c = X - μ")
    print("4) SVD: X_c = U Σ V^T")
    print('5) Componentes principais (autovetores) = linhas de V^T (i.e., V^T[0:k, :]')
    print('6) Projeção: scores = X_c @ V^T[0:k, :].T; Reconstrução: X_hat = scores @ V^T[0:k, :] + μ')


def mse(X: Any, X_hat: Any) -> float:
    np = _import_numpy(strict=True)
    return float(np.mean((X - X_hat) ** 2))


def plot_explained_variance(explained_variance: Any):
    np = _import_numpy(strict=False)
    plt = _import_matplotlib(strict=True)
    if np is None:
        total = sum(explained_variance)
        running = 0.0
        ratios = []
        for ev in explained_variance:
            running += float(ev)
            ratios.append(running / total if total != 0 else 0)
    else:
        ev = np.array(explained_variance)
        ratios = np.cumsum(ev) / ev.sum()
    plt.figure(figsize=(8, 4))
    xs = (np.arange(1, len(ratios)+1) if np is not None else range(1, len(ratios)+1))
    plt.plot(xs, ratios, marker='o')
    plt.xlabel('Número de componentes principais (k)')
    plt.ylabel('Variância explicada cumulativa')
    plt.grid(True)
    plt.title('Variância explicada cumulativa por componentes')
    plt.tight_layout()


def _shape_of(array: Any) -> tuple[int, int]:
    np = _import_numpy(strict=False)
    if np is not None and hasattr(array, 'shape'):
        return array.shape[0], array.shape[1]
    # else assume a list of lists or list of flat
    if isinstance(array, list):
        N = len(array)
        D = len(array[0]) if N > 0 else 0
        return N, D
    raise ValueError('Unsupported array type for shape')


def _vec_to_2d(image: Any) -> list[list[float]]:
    # image can be 1D numpy array or flat list
    np = _import_numpy(strict=False)
    if np is not None and hasattr(image, 'reshape'):
        h = int(np.sqrt(image.size))
        return image.reshape((h, h)).tolist()
    # pure-python list
    D = len(image)
    h = int(math.sqrt(D))
    return [image[i*h:(i+1)*h] for i in range(h)]


def plot_reconstructions(originals: Any, reconstructions: Any, k: int, n_cols: int = 10):
    n, D = _shape_of(originals)
    n = min(n, n_cols)
    h = int(math.sqrt(D))

    plt = _import_matplotlib(strict=True)
    plt.figure(figsize=(2 * n, 4))
    for i in range(n):
        # original
        ax = plt.subplot(2, n, i+1)
        orig2d = _vec_to_2d(originals[i])
        plt.imshow(orig2d, cmap='gray')
        plt.axis('off')
        if i == 0:
            ax.set_title('Original')
        # recon
        ax = plt.subplot(2, n, n + i + 1)
        # Clip values between 0 and 1
        rec2d = _vec_to_2d(reconstructions[i])
        # ensure clipping
        rec2d = [[min(max(float(v), 0.0), 1.0) for v in row] for row in rec2d]
        plt.imshow(rec2d, cmap='gray')
        plt.axis('off')
        if i == 0:
            ax.set_title(f'Recon k={k}')
    plt.suptitle('Originais (linha superior) vs Reconstruções (linha inferior)')
    plt.tight_layout(rect=[0, 0, 1, 0.95])


def plot_2d_projection(scores: Any, labels: Any):
    plt = _import_matplotlib(strict=True)
    plt.figure(figsize=(8, 6))
    np = _import_numpy(strict=False)
    if np is not None and hasattr(scores, 'shape'):
        # numpy arrays are easy
        for digit in range(10):
            mask = labels == digit
            plt.scatter(scores[mask, 0], scores[mask, 1], label=str(digit), alpha=0.6, s=8)
    else:
        # lists
        for digit in range(10):
            xs = [s[0] for s, lab in zip(scores, labels) if lab == digit]
            ys = [s[1] for s, lab in zip(scores, labels) if lab == digit]
            plt.scatter(xs, ys, label=str(digit), alpha=0.6, s=8)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend(title='Dígito')
    plt.grid(True)
    plt.title('Projeção 2D (PC1 vs PC2) do MNIST')
    plt.tight_layout()


def run_experiment(k_values: list[int], max_samples: int | None = 5000, dest_dir: str = "data", pure_python: bool = False):
    plt = _import_matplotlib(strict=True)
    if pure_python:
        print('Modo pure-python: PCA será realizado sem NumPy (muito lento para N grande).')
        X, y = load_mnist_pure(dest_dir=dest_dir, kind='train', max_samples=max_samples)
    else:
        X, y = load_mnist(dest_dir=dest_dir, kind='train', max_samples=max_samples)
    print(f"Dados carregados: {X.shape[0]} amostras, dimensão = {X.shape[1]}")

    # full SVD once and reuse components for different k values to save compute
    # However, we use compute_pca per k for clarity (since n_components affects components returned)
    results = []
    # Compute full PCA once
    k_max = max(k_values) if len(k_values) > 0 else 0
    if pure_python:
        # compute only as many components as needed
        components_all, explained_variance_all, mean_all = compute_pca_pure(X, n_components=k_max)
        # components_all is k x D here; but we only need top-k for requested k values
    else:
        components_all, explained_variance_all, mean_all = compute_pca_full(X)
    # Plot cumulative explained variance for full set
    plt.figure(figsize=(8, 4))
    if pure_python:
        total_ev = sum(explained_variance_all)
        cumsum = []
        running = 0.0
        for ev in explained_variance_all:
            running += ev
            cumsum.append(running / total_ev if total_ev != 0 else 0)
        ratios = cumsum
    else:
        np = _import_numpy(strict=True)
        ratios = np.cumsum(explained_variance_all) / np.sum(explained_variance_all)
    plt.plot(np.arange(1, len(ratios) + 1), ratios)
    plt.xlabel('Número de componentes principais (k)')
    plt.ylabel('Variância explicada cumulativa')
    plt.grid(True)
    plt.title('Variância explicada cumulativa por componentes (full PCA)')
    out_dir = Path('outputs')
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / 'explained_variance_cumulative_full.png')
    plt.close()

    for k in k_values:
        k = int(k)
        print(f"Calculando PCA com k={k} componentes...")
        if pure_python:
            # select top-k components from full set calculated via compute_pca_pure
            components = components_all[:k]
            explained_variance = explained_variance_all[:k]
            mean = mean_all
            scores = project_pure(X, components, mean)
            Xrec = reconstruct_pure(scores, components, mean)
            err = mse_pure(X, Xrec)
        else:
            components, explained_variance, mean = compute_pca(X, n_components=k, pca_full=(components_all, explained_variance_all, mean_all))
            scores = project(X, components, mean)
            Xrec = reconstruct(scores, components, mean)
            err = mse(X, Xrec)
        print(f"MSE de reconstrução para k={k}: {err:.6f} | soma variância explicada: {explained_variance.sum()/explained_variance_all.sum():.6f}")

        # Plots
        # Plot explained variance for the given k (cumulative ratio up to k)
        plt.figure(figsize=(6, 3))
        np = _import_numpy(strict=False)
        if np is None:
            # pure-python: build cumulative ratio list manually
            total_ev = sum(explained_variance)
            running = 0.0
            ratios_k = []
            for ev in explained_variance:
                running += float(ev)
                ratios_k.append(running / total_ev if total_ev != 0 else 0)
            plt.plot(range(1, len(ratios_k) + 1), ratios_k, marker='o')
        else:
            plt.plot(np.arange(1, len(explained_variance)+1), np.cumsum(explained_variance) / explained_variance_all.sum(), marker='o')
        plt.xlabel('k')
        plt.ylabel('Variância explicada cumulativa (até k)')
        plt.grid(True)
        plt.title(f'Variância explicada até k={k}')
        plt.tight_layout()
        plt.savefig(out_dir / f'explained_variance_k_{k}.png')
        plt.close()
        # Plot reconstructions — convert to numpy arrays if not pure-python for consistency
        if pure_python:
            # X and Xrec are lists — pass lists to plotting helper which supports lists too
            plot_reconstructions(X, Xrec, k=k, n_cols=10)
        else:
            plot_reconstructions(X, Xrec, k=k, n_cols=10)
        if k >= 2:
            # 2D scatter with first two components (from full PCA)
            if pure_python:
                comps2 = components_all[:2]
                sc2 = project_pure(X, comps2, mean_all)
            else:
                comps2 = components_all[:2, :]
                sc2 = project(X, comps2, mean_all)
            plot_2d_projection(sc2, y)
            plt.savefig(out_dir / f'proj_2d_k_{k}.png')
            plt.close()

        results.append({'k': k, 'mse': err, 'explained_variance_total': explained_variance.sum()})

        # Salvar figura
        out_dir = Path('outputs')
        out_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_dir / f'plots_k_{k}.png')
        plt.close('all')

    print("Experimento finalizado. Figuras salvas em ./outputs")
    # Save results summary
    import csv
    csv_path = out_dir / 'mse_results.csv'
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['k', 'mse', 'explained_variance_total']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f"Resumo de resultados salvo em {csv_path}")
    return results


def quick_demo(n_samples: int = 100, dim: int = 64, k_values: list[int] | None = None):
    """Quick demo using synthetic data, runs both implementations and prints metrics."""
    if k_values is None:
        k_values = [5, 10, 20]
    # generate synthetic data
    random.seed(42)
    X = [[random.random() for _ in range(dim)] for _ in range(n_samples)]
    np = _import_numpy(strict=False)
    # run numpy PCA (if numpy available)
    if np is not None:
        X_np = np.array(X, dtype=np.float32)
        components_all, explained_variance_all, mean_all = compute_pca_full(X_np)
        components_k, ev_k, mean_k = compute_pca(X_np, n_components=max(k_values), pca_full=(components_all, explained_variance_all, mean_all))
        scores = project(X_np, components_k, mean_k)
        Xrec = reconstruct(scores, components_k, mean_k)
        print('NumPy MSE full:', float(mse(X_np, Xrec)))
    # run pure PCA
    comp_p, ev_p, mean_p = compute_pca_pure(X, n_components=max(k_values))
    scores_p = project_pure(X, comp_p, mean_p)
    Xrec_p = reconstruct_pure(scores_p, comp_p, mean_p)
    print('Pure Python MSE full:', mse_pure(X, Xrec_p))
    # Print MSE per k
    # Print MSE per k (if NumPy available)
    if np is not None:
        for k in k_values:
            comps = components_k[:k]
            comps_np = np.array(comps) if not isinstance(comps, np.ndarray) else comps
            scores = project(X_np, comps_np, mean_k)
            Xrec = reconstruct(scores, comps_np, mean_k)
            print(f'NumPy MSE k={k}:', float(mse(X_np, Xrec)))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PCA no MNIST (implementação manual)')
    parser.add_argument('--k', nargs='+', type=int, default=[10, 20, 50, 100], help='Lista de componentes k para testar')
    parser.add_argument('--max-samples', type=int, default=5000, help='Número máximo de amostras para executar (padrão 5000)')
    parser.add_argument('--dest', type=str, default='data', help='Diretório onde baixar MNIST')
    parser.add_argument('--pure-python', action='store_true', help='Usar implementação Pure-Python (sem NumPy) — muito lento, apenas para fins didáticos')
    parser.add_argument('--quicktest', action='store_true', help='Executar demo rápido com dados sintéticos sem baixar MNIST (útil para testar)')
    args = parser.parse_args()

    if args.quicktest:
        quick_demo(n_samples=min(200, args.max_samples or 200), dim=64, k_values=args.k)
    else:
        run_experiment(args.k, max_samples=args.max_samples, dest_dir=args.dest, pure_python=args.pure_python)
