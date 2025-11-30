"""
improved_lda_mnist_modes.py

Versão com "modos" de LDA — permite comparar implementação manual (autovalores/autovetores)
com sklearn. Modos disponíveis:
 - 'manual_eig'   : calcula autovalores/autovetores do problema generalizado via np.linalg.eig
 - 'manual_power' : calcula autovalores/autovetores usando método das potências + deflação
 - 'sklearn'      : usa sklearn.discriminant_analysis.LinearDiscriminantAnalysis

Em todos os casos a LDA via sklearn é executada para comparação.
Salva relatório em outputs/lda_report.txt e imagens das componentes.
"""
import sys
import io
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# -----------------------
# Configurações gerais
# -----------------------
RANDOM_STATE = 42
MAX_SAMPLES = 5000
LDA_METHOD = "manual_power"  # 'manual_eig' | 'manual_power' | 'sklearn'
VERBOSE = True

# -----------------------
# Capture stdout (opcional)
# -----------------------
class Tee:
    def __init__(self, stdout, buffer):
        self.stdout = stdout
        self.buffer = buffer

    def write(self, s):
        # escreve tanto no stdout original quanto no buffer
        try:
            self.stdout.write(s)
        except Exception:
            pass
        try:
            self.buffer.write(s)
        except Exception:
            pass

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
# Helpers LDA manuais
# -----------------------
def compute_class_stats(X, y):
    classes = np.unique(y)
    mu_k = {}
    n_k = {}
    for cls in classes:
        mask = (y == cls)
        Xc = X[mask]
        mu_k[cls] = Xc.mean(axis=0)
        n_k[cls] = Xc.shape[0]
    mu = X.mean(axis=0)
    return classes, mu_k, n_k, mu

def compute_SW_SB(X, y, classes, mu_k, n_k, mu):
    D = X.shape[1]
    S_W = np.zeros((D, D), dtype=np.float64)
    S_B = np.zeros((D, D), dtype=np.float64)
    for cls in classes:
        mask = (y == cls)
        Xc = X[mask]
        Xc_centered = Xc - mu_k[cls]
        S_W += Xc_centered.T.dot(Xc_centered)
        mean_diff = (mu_k[cls] - mu).reshape(-1, 1)
        S_B += n_k[cls] * (mean_diff).dot(mean_diff.T)
    return S_W, S_B

def manual_eig_lda(S_W, S_B, num_components):
    # resolve o problema generalizado via inversa pseudoinversa e eig
    S_W_inv = np.linalg.pinv(S_W)
    mat = S_W_inv.dot(S_B)
    eigvals, eigvecs = np.linalg.eig(mat)
    # ordenar por autovalores decrescentes (parte real)
    idx = np.argsort(np.real(eigvals))[::-1]
    eigvals = np.real(eigvals[idx])
    eigvecs = np.real(eigvecs[:, idx])
    W = eigvecs[:, :num_components]
    return eigvals, eigvecs, W

def power_iteration(A, num_iter=2000, tol=1e-6, random_state=None):
    rng = np.random.RandomState(random_state)
    n = A.shape[0]
    v = rng.normal(size=n)
    v /= np.linalg.norm(v)
    for _ in range(num_iter):
        v_new = A @ v
        norm = np.linalg.norm(v_new)
        if norm == 0:
            break
        v_new /= norm
        if np.linalg.norm(v - v_new) < tol:
            v = v_new
            break
        v = v_new
    eigenvalue = float(v.T @ A @ v)
    return eigenvalue, v

def manual_power_lda(S_W, S_B, num_components, max_components=None, random_state=None):
    # monta matriz (S_W^{-1} S_B) e aplica power iteration + deflação para obter num_components
    S_W_inv = np.linalg.pinv(S_W)
    A = S_W_inv.dot(S_B)
    n = A.shape[0]
    if max_components is None:
        max_components = num_components
    eigvals = []
    eigvecs = []
    A_work = A.copy().astype(float)
    for i in range(max_components):
        val, vec = power_iteration(A_work, random_state=(None if random_state is None else random_state + i))
        if np.isnan(val) or np.isclose(val, 0.0):
            break
        eigvals.append(val)
        eigvecs.append(vec)
        # deflação: remove componente dominante encontrada
        A_work = A_work - val * np.outer(vec, vec)
        # simetriza para evitar erros numéricos acumulados
        A_work = (A_work + A_work.T) / 2.0
        if len(eigvals) >= num_components:
            break
    if len(eigvals) == 0:
        raise RuntimeError("Power iteration failed to converge any eigenpairs.")
    eigvals = np.array(eigvals)
    eigvecs = np.column_stack(eigvecs)
    W = eigvecs[:, :num_components]
    return eigvals, eigvecs, W

# -----------------------
# Main pipeline
# -----------------------
def run_lda(max_samples=MAX_SAMPLES, lda_method=LDA_METHOD):
    print("\nIniciando pipeline LDA (MNIST subset)...")

    # Carregar MNIST
    print("\nBuscando MNIST (OpenML)...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X_all = mnist.data
    y_all = mnist.target.astype(int)

    # Subset estratificado
    num_samples = min(max_samples, X_all.shape[0])
    X_subset, _, y_subset, _ = train_test_split(X_all, y_all, train_size=num_samples, stratify=y_all, random_state=RANDOM_STATE)
    X = X_subset
    y = y_subset
    print(f"Usando {X.shape[0]} amostras - shape: {X.shape}")

    # DataFrame (opcional)
    feature_names = [f"pixel_{r}_{c}" for r in range(28) for c in range(28)]
    df = pd.DataFrame(X, columns=feature_names)

    # Normalização
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)
    print("X_scaled shape:", X_scaled.shape)

    # Baseline
    X_tr, X_te, y_tr, y_te = train_test_split(X_scaled, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    model_base = LogisticRegression(solver='lbfgs', max_iter=1000, random_state=RANDOM_STATE)
    model_base.fit(X_tr, y_tr)
    acc_base = model_base.score(X_te, y_te)
    print("Baseline (Logistic on scaled features) accuracy:", acc_base)

    # Stats e scatter matrices
    classes, mu_k, n_k, mu = compute_class_stats(X_scaled, y)
    num_classes = classes.size
    D = X_scaled.shape[1]
    k_lda = num_classes - 1
    print(f"Num classes: {num_classes}, feature dim: {D}, k_lda (max): {k_lda}")

    S_W, S_B = compute_SW_SB(X_scaled, y, classes, mu_k, n_k, mu)
    print("Computed S_W and S_B")

    # Escolha do método manual/principal
    if lda_method == "manual_eig":
        print("\nExecutando LDA manual via np.linalg.eig (problema generalizado)...")
        eigvals_m, eigvecs_m, W_manual = manual_eig_lda(S_W, S_B, k_lda)
    elif lda_method == "manual_power":
        print("\nExecutando LDA totalmente manual via power iteration + deflação (pode ser lento)...")
        eigvals_m, eigvecs_m, W_manual = manual_power_lda(S_W, S_B, k_lda, max_components=min(k_lda, 300), random_state=RANDOM_STATE)
    elif lda_method == "sklearn":
        print("\nModo principal: sklearn LDA (faremos sklearn como principal e também opcionalmente calcular 'manual' via eig para comparação).")
        # build sklearn LDA and set W_manual from sklearn scalings_
        lda_main = LinearDiscriminantAnalysis(n_components=k_lda)
        X_lda_main = lda_main.fit_transform(X_scaled, y)
        # scalings_: shape (n_features, n_components)
        scalings = lda_main.scalings_[:, :k_lda]
        W_manual = scalings  # para compatibilidade de projeção manual
        eigvals_m = None
        eigvecs_m = None
        print("Sklearn LDA computed as principal.")
    else:
        raise ValueError("LDA_METHOD inválido. Use 'manual_eig', 'manual_power' ou 'sklearn'.")

    # Projeção manual (se disponível)
    X_lda_manual = X_scaled.dot(W_manual) if W_manual is not None else None
    if X_lda_manual is not None:
        print("X_lda_manual shape:", X_lda_manual.shape)
        # Reconstrução aproximada e MSE no espaço escalado
        X_recon_manual = X_lda_manual.dot(W_manual.T)
        mse_manual = float(np.mean((X_scaled - X_recon_manual) ** 2))
        print("MSE reconstrução (manual, espaço escalado):", mse_manual)
    else:
        mse_manual = None
        X_recon_manual = None

    # LDA sklearn sempre para comparação
    lda_sklearn = LinearDiscriminantAnalysis(n_components=k_lda)
    X_lda_sklearn = lda_sklearn.fit_transform(X_scaled, y)
    scalings_sklearn = lda_sklearn.scalings_[:, :k_lda]
    print("LDA sklearn - X_lda_sklearn shape:", X_lda_sklearn.shape)

    # Avaliação: treinar Logistic em cada espaço LDA (manual e sklearn)
    def eval_logistic(X_proj, y, name):
        Xtr, Xte, ytr, yte = train_test_split(X_proj, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
        m = LogisticRegression(solver='lbfgs', max_iter=1000, random_state=RANDOM_STATE)
        m.fit(Xtr, ytr)
        return m.score(Xte, yte)

    acc_manual = eval_logistic(X_lda_manual, y, "manual") if X_lda_manual is not None else None
    acc_sklearn = eval_logistic(X_lda_sklearn, y, "sklearn")
    print("Accuracy (Logistic on LDA manual):", acc_manual)
    print("Accuracy (Logistic on LDA sklearn):", acc_sklearn)

    # Comparações
    print("\nComparações:")
    print("k_lda:", k_lda)
    if eigvals_m is not None:
        print("Top eigenvalues (manual):", np.round(eigvals_m[:min(10, eigvals_m.size)], 6))
    print("Sklearn scalings (first 5 elements por componente):")
    print(np.round(scalings_sklearn[:5, :min(3, scalings_sklearn.shape[1])], 4))

    # Salvar imagens das primeiras componentes (manual vs sklearn)
    out_dir = Path(__file__).resolve().parent.parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(min(3, k_lda)):
        if X_lda_manual is not None:
            comp_manual_img = W_manual[:, i].reshape(28, 28)
            plt.figure(figsize=(3, 3))
            plt.imshow(comp_manual_img, cmap="seismic")
            plt.title(f"LDA manual componente {i+1}")
            plt.axis("off")
            plt.savefig(out_dir / f"lda_manual_comp_{i+1}.png")
            plt.close()
        comp_sklearn_img = scalings_sklearn[:, i].reshape(28, 28)
        plt.figure(figsize=(3, 3))
        plt.imshow(comp_sklearn_img, cmap="seismic")
        plt.title(f"LDA sklearn componente {i+1}")
        plt.axis("off")
        plt.savefig(out_dir / f"lda_sklearn_comp_{i+1}.png")
        plt.close()

    print("Saved LDA component images to outputs/")

    # Relatório
    report_path = out_dir / "lda_report.txt"
    try:
        with report_path.open("w", encoding="utf-8") as f:
            f.write("LDA Analysis Report\n")
            f.write(f"Generated: {datetime.now(timezone.utc).isoformat()} UTC\n\n")
            f.write(f"LDA_METHOD: {lda_method}\n")
            f.write(f"Num samples: {X.shape[0]}\n")
            f.write(f"Feature dim: {D}\n")
            f.write(f"Num classes: {num_classes}\n\n")
            f.write(f"Baseline (Logistic on scaled): {acc_base:.6f}\n\n")
            if eigvals_m is not None:
                f.write(f"Manual top eigenvalues: {np.round(eigvals_m[:min(10, eigvals_m.size)],6).tolist()}\n")
            f.write(f"MSE recon (manual, scaled): {mse_manual}\n")
            f.write(f"Accuracy (Logistic on LDA manual): {acc_manual}\n")
            f.write(f"Accuracy (Logistic on LDA sklearn): {acc_sklearn}\n\n")
            f.write("Sklearn scalings_ (first 10 elements por componente):\n")
            f.write(np.array2string(np.round(scalings_sklearn[:10, :min(3, scalings_sklearn.shape[1])], 4), separator=', '))
            f.write("\n\nFull stdout log:\n")
            try:
                f.write(_stdout_buf.getvalue())
            except Exception:
                f.write("Failed to capture full stdout log.\n")
        print(f"Report written to: {report_path}")
    except Exception as e:
        print("Failed to write report:", e)

    print("\nFim do script LDA.")

if __name__ == "__main__":
    run_lda(max_samples=MAX_SAMPLES, lda_method=LDA_METHOD)