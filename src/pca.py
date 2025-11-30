"""
improved_pca_mnist_modes.py

Modo de uso: ajuste PCA_METHOD para 'manual_eigh', 'manual_power' ou 'sklearn'.

- 'manual_eigh': calcula autovalores/autovetores com np.linalg.eigh (manual no sentido de não usar sklearn.PCA)
- 'manual_power': calcula autovalores/autovetores com método das potências + deflação (completamente manual; educativo)
- 'sklearn': usa sklearn.decomposition.PCA

Em todos os casos a PCA via sklearn é feita para comparação.
"""
import sys
import io
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# -----------------------
# Configurações gerais
# -----------------------
RANDOM_STATE = 42
MAX_SAMPLES = 5000
PCA_VARIANCE_KEEP = 0.95

# Escolha do método de PCA principal:
# - 'manual_eigh' : calcular autovalores/autovetores com np.linalg.eigh (manualmente, sem sklearn.PCA)
# - 'manual_power': calcular autovalores/autovetores com método das potências + deflação (totalmente manual, educativo)
# - 'sklearn'     : usar sklearn.decomposition.PCA
PCA_METHOD = "manual_power"  # troque aqui conforme desejar

# -----------------------
# Capture stdout (opcional)
# -----------------------
class Tee:
    def __init__(self, stdout, buffer):
        self.stdout = stdout
        self.buffer = buffer

    def write(self, s):
        self.stdout.write(s)
        self.buffer.write(s)

    def flush(self):
        try:
            self.stdout.flush()
        except Exception:
            pass
        try:
            self.buffer.flush()
        except Exception:
            pass

_stdout_buf = io.StringIO()
_tee = Tee(sys.stdout, _stdout_buf)
sys.stdout = _tee

# -----------------------
# Helpers PCA manuais
# -----------------------
def compute_covariance(X):
    # X assumed centered/scaled; rowvar=False -> variables are columns
    return np.cov(X, rowvar=False)

def manual_eigh_pca(cov_matrix, var_keep=PCA_VARIANCE_KEEP):
    # usa np.linalg.eigh para obter autovalores/autovetores
    e_vals, e_vecs = np.linalg.eigh(cov_matrix)  # crescente
    idx = np.argsort(e_vals)[::-1]
    e_vals = e_vals[idx]
    e_vecs = e_vecs[:, idx]
    explained_ratio = e_vals / np.sum(e_vals)
    cum = np.cumsum(explained_ratio)
    k = int(np.argmax(cum >= var_keep) + 1)
    components = e_vecs[:, :k]
    return e_vals, e_vecs, components, explained_ratio, k

def power_iteration(A, num_iter=5000, tol=1e-6, random_state=None):
    rng = np.random.RandomState(random_state)
    n = A.shape[0]
    v = rng.normal(size=n)
    v = v / np.linalg.norm(v)
    for _ in range(num_iter):
        v_new = A @ v
        norm = np.linalg.norm(v_new)
        if norm == 0:
            break
        v_new = v_new / norm
        if np.linalg.norm(v - v_new) < tol:
            v = v_new
            break
        v = v_new
    eigenvalue = float(v.T @ A @ v)
    return eigenvalue, v

def manual_power_pca(cov_matrix, var_keep=PCA_VARIANCE_KEEP, max_components=None, random_state=None):
    # Extrai autovalores/autovetores usando método das potências + deflação (educacional).
    n = cov_matrix.shape[0]
    if max_components is None:
        max_components = n
    A = cov_matrix.copy().astype(float)
    eigenvalues = []
    eigenvectors = []
    for i in range(max_components):
        val, vec = power_iteration(A, random_state=(None if random_state is None else random_state + i))
        if np.isnan(val) or np.isclose(val, 0.0):
            break
        eigenvalues.append(val)
        eigenvectors.append(vec)
        # deflação simples (remove componente encontrada)
        A = A - val * np.outer(vec, vec)
        # small numerical stabilization
        A = (A + A.T) / 2.0
        # stop if cumulative variance reached
        total_var = np.sum(eigenvalues) + 1e-12
        explained_ratio = np.array(eigenvalues) / total_var
        if np.cumsum(explained_ratio).max() >= var_keep:
            break
    if len(eigenvalues) == 0:
        raise RuntimeError("Power iteration failed to converge any eigenpairs.")
    eigenvalues = np.array(eigenvalues)
    eigenvectors = np.column_stack(eigenvectors)
    # compute explained ratio relative to full covariance trace
    total_all = np.trace(cov_matrix)
    explained_ratio_full = eigenvalues / total_all
    cum = np.cumsum(explained_ratio_full)
    k = int(np.argmax(cum >= var_keep) + 1) if cum.max() >= var_keep else eigenvalues.shape[0]
    components = eigenvectors[:, :k]
    return eigenvalues, eigenvectors, components, explained_ratio_full, k

# -----------------------
# Carregar MNIST
# -----------------------
print("\nBuscando MNIST (OpenML)...")
mnist = fetch_openml("mnist_784", version=1, as_frame=False)
X_all = mnist.data
y_all = mnist.target.astype(int)

num_samples = min(MAX_SAMPLES, X_all.shape[0])
X = X_all[:num_samples]
y = y_all[:num_samples]
print(f"Usando {num_samples} amostras - shape: {X.shape}")

# DataFrame e escala
feature_names = [f"pixel_{r}_{c}" for r in range(28) for c in range(28)]
df = pd.DataFrame(X, columns=feature_names)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)  # já centralizado

# Baseline raw
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
model_raw = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=RANDOM_STATE)
model_raw.fit(X_train, y_train)
accuracy_raw = model_raw.score(X_test, y_test)
print("\nAccuracy (raw / scaled):", accuracy_raw)

# -----------------------
# Escolha e execução do PCA "principal"
# -----------------------
cov_matrix = compute_covariance(X_scaled)

if PCA_METHOD == "manual_eigh":
    print("\nExecutando PCA manual com np.linalg.eigh...")
    e_vals_m, e_vecs_m, proj_manual, explained_ratio_m, num_components_manual = manual_eigh_pca(cov_matrix, PCA_VARIANCE_KEEP)
    X_pca_manual = X_scaled.dot(proj_manual)
elif PCA_METHOD == "manual_power":
    print("\nExecutando PCA totalmente manual com método das potências + deflação (pode ser lento)...")
    e_vals_m, e_vecs_m, proj_manual, explained_ratio_m, num_components_manual = manual_power_pca(cov_matrix, PCA_VARIANCE_KEEP, max_components=300, random_state=RANDOM_STATE)
    X_pca_manual = X_scaled.dot(proj_manual)
elif PCA_METHOD == "sklearn":
    print("\nExecutando PCA via sklearn (modo principal)...")
    pca_main = PCA(n_components=PCA_VARIANCE_KEEP, svd_solver="full", random_state=RANDOM_STATE)
    X_pca_main = pca_main.fit_transform(X_scaled)
    # for compatibility, set 'manual' outputs to sklearn-derived values
    proj_manual = pca_main.components_.T
    e_vals_m = pca_main.explained_variance_
    explained_ratio_m = pca_main.explained_variance_ratio_
    num_components_manual = pca_main.n_components_
    X_pca_manual = X_pca_main
else:
    raise ValueError("PCA_METHOD inválido. Use manual_eigh, manual_power ou sklearn.")

print(f"PCA principal ({PCA_METHOD}) - n_components: {num_components_manual}")
print("Explained variance (first 10):", np.round(explained_ratio_m[:10], 6))

# -----------------------
# PCA sklearn (sempre fazer para comparar)
# -----------------------
pca_sklearn = PCA(n_components=PCA_VARIANCE_KEEP, svd_solver="full", random_state=RANDOM_STATE)
X_pca_sklearn = pca_sklearn.fit_transform(X_scaled)
print("\nSklearn PCA - n_components:", pca_sklearn.n_components_)

# -----------------------
# Comparações e reconstruções
# -----------------------
min_components = min(num_components_manual, pca_sklearn.n_components_)
# reconstruct (centered/scaled space)
X_recon_manual = X_pca_manual.dot(proj_manual.T)
X_recon_sklearn = pca_sklearn.inverse_transform(X_pca_sklearn)
mse_manual = np.mean((X_scaled - X_recon_manual) ** 2)
mse_sklearn = np.mean((X_scaled - X_recon_sklearn) ** 2)

print("\nComparação:")
print("Manual components:", num_components_manual, "Sklearn components:", pca_sklearn.n_components_)
print("MSE recon (manual):", mse_manual, "MSE recon (sklearn):", mse_sklearn)
# comparar variâncias acumuladas (primeiros k)
expl_manual_cum = np.cumsum((e_vals_m / np.sum(e_vals_m))[:min_components]) if e_vals_m is not None else None
expl_sklearn_cum = np.cumsum(pca_sklearn.explained_variance_ratio_[:min_components])
if expl_manual_cum is not None:
    print("Cumulated explained variance (manual first k):", expl_manual_cum)
print("Cumulated explained variance (sklearn first k):", expl_sklearn_cum)

# -----------------------
# Treinar modelos reduzidos (sklearn PCA) e PCA 2D
# -----------------------
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca_sklearn, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
model_pca95 = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=RANDOM_STATE)
model_pca95.fit(X_train_pca, y_train_pca)
score_pca95 = model_pca95.score(X_test_pca, y_test_pca)
print(f"\nScore PCA (95% var, sklearn n_components={pca_sklearn.n_components_}):", score_pca95)

pca_2d = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca_2d = pca_2d.fit_transform(X_scaled)
X_train_2d, X_test_2d, y_train_2d, y_test_2d = train_test_split(X_pca_2d, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
model_pca2d = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=RANDOM_STATE)
model_pca2d.fit(X_train_2d, y_train_2d)
score_pca2d = model_pca2d.score(X_test_2d, y_test_2d)
print("Score PCA 2D:", score_pca2d)

# -----------------------
# Relatório final salvo
# -----------------------
print("\nResumo final:")
print("Accuracy (raw):", accuracy_raw)
print("Score PCA (sklearn 95%):", score_pca95)
print("Score PCA (2D):", score_pca2d)

try:
    base_dir = Path(__file__).resolve().parent
except NameError:
    base_dir = Path.cwd()
out_dir = base_dir / "outputs"
out_dir.mkdir(parents=True, exist_ok=True)
report_path = out_dir / "pca_report.txt"

with report_path.open("w", encoding="utf-8") as f:
    f.write("PCA Analysis Report\n")
    f.write(f"Generated: {datetime.now(timezone.utc).isoformat()} UTC\n\n")
    f.write(f"PCA_METHOD: {PCA_METHOD}\n")
    f.write(f"Num samples: {X.shape[0]}\n\n")
    f.write(f"Accuracy (raw): {accuracy_raw:.6f}\n")
    f.write(f"PCA principal method components: {num_components_manual}\n")
    f.write(f"MSE recon (manual): {mse_manual:.8f}\n")
    f.write(f"Sklearn PCA components: {pca_sklearn.n_components_}\n")
    f.write(f"MSE recon (sklearn): {mse_sklearn:.8f}\n")
    f.write("\nFull stdout log:\n")
    try:
        f.write(_stdout_buf.getvalue())
    except Exception as e:
        f.write(f"Failed to capture full log: {e}\n")


print(f"\nReport written to: {report_path}")