import sys
import io
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

"""
Script de LDA (Linear Discriminant Analysis) para MNIST.
Faz: carregamento MNIST (OpenML) -> subset -> normalização -> LDA manual -> LDA sklearn -> avaliação comparativa.

Inclui: comentários em Português, cálculo manual (S_W, S_B, autovetores), comparação com sklearn, plots e relatório em outputs/lda_report.txt.
"""


# Small utility that duplicates stdout to both terminal and a buffer so we can
# save everything printed to disk later. We set it early so all subsequent prints
# are captured.
class Tee:
	def __init__(self, stdout, buffer):
		self.stdout = stdout
		self.buffer = buffer

	def write(self, s):
		# Write to terminal
		self.stdout.write(s)
		# Also write to buffer (no extra newline added)
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


# set up capture
_stdout_buf = io.StringIO()
_tee = Tee(sys.stdout, _stdout_buf)
sys.stdout = _tee


def main(max_samples: int = 5000):
	print("\nIniciando pipeline LDA (MNIST subset)...")

	# Carregar MNIST via OpenML
	print("\nBuscando MNIST (OpenML) — caso não esteja em cache, será baixado...")
	mnist = fetch_openml('mnist_784', version=1, as_frame=False)
	X_all = mnist.data
	y_all = mnist.target.astype(int)

	# Subsetting: para testes rápidos, limitamos o número de amostras (estratificado)
	num_samples = min(max_samples, X_all.shape[0])
	X, _, y, _ = train_test_split(X_all, y_all, train_size=num_samples, stratify=y_all, random_state=42)
	print("\nShape (subset):", X.shape)

	# Monta DataFrame com nomes de pixels (opcional, útil para debug/visualização)
	feature_names = [f"pixel_{r}_{c}" for r in range(28) for c in range(28)]
	data_frame = pd.DataFrame(X, columns=feature_names)
	print("\nDataFrame head:\n", data_frame.head())

	# Normalização (StandardScaler): média 0, desvio padrão 1
	scaler = StandardScaler()
	X_scaled = scaler.fit_transform(data_frame)
	print("\nX_scaled shape:", X_scaled.shape)

	# Baseline (LogisticRegression) no espaço original escalado
	X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(X_scaled, y, test_size=0.2, random_state=30, stratify=y)
	model_raw = LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='multinomial')
	model_raw.fit(X_train_raw, y_train_raw)
	accuracy_raw = model_raw.score(X_test_raw, y_test_raw)
	print("\nBaseline (LogisticRegression) accuracy:", accuracy_raw)

	# -----------------------------------------------------------
	# LDA manual: calcular S_W (intra-class scatter) e S_B (between-class scatter)
	# -----------------------------------------------------------
	classes = np.unique(y)
	num_classes = classes.size
	D = X_scaled.shape[1]  # dimensão dos features
	print("\nNúmero de classes:", num_classes, "Dimensão dos features:", D)

	# Médias por classe e média global
	mu_k = {}
	n_k = {}
	for cls in classes:
		mask = (y == cls)
		X_cls = X_scaled[mask]
		mu_k[cls] = X_cls.mean(axis=0)
		n_k[cls] = X_cls.shape[0]
	mu = X_scaled.mean(axis=0)
	print("\nMédias de classe e média global calculadas.")

	# Inicializa S_W e S_B
	S_W = np.zeros((D, D), dtype=np.float64)
	S_B = np.zeros((D, D), dtype=np.float64)

	# Calcula S_W e S_B
	for cls in classes:
		mask = (y == cls)
		X_cls = X_scaled[mask]
		# dentro da classe: soma de (x - mu_k)(x - mu_k)^T
		Xc = X_cls - mu_k[cls]
		S_W += Xc.T.dot(Xc)
		# entre as classes: n_k * (mu_k - mu)(mu_k - mu)^T
		mean_diff = (mu_k[cls] - mu).reshape(-1, 1)
		S_B += n_k[cls] * (mean_diff).dot(mean_diff.T)

	print("\nS_W shape:", S_W.shape, "S_B shape:", S_B.shape)

	# Resolver o problema generalizado S_W^{-1} S_B w = lambda w
	# Usamos pseudoinversa de S_W para estabilidade numérica
	S_W_inv = np.linalg.pinv(S_W)
	mat = S_W_inv.dot(S_B)
	eigvals, eigvecs = np.linalg.eig(mat)
	# Ordena por autovalores decrescentes (as direções mais discriminantes têm maiores autovalores)
	idx = np.argsort(np.real(eigvals))[::-1]
	eigvals = np.real(eigvals[idx])
	eigvecs = np.real(eigvecs[:, idx])

	# Número máximo de componentes LDA = num_classes - 1
	k_lda = num_classes - 1
	eigvals_top = eigvals[:k_lda]
	eigvecs_top = eigvecs[:, :k_lda]
	print("\nLDA manual - selected components (k):", k_lda)
	print("Top eigenvalues (LDA manual):", eigvals_top)

	# Projeção manual: N x k_lda
	X_lda_manual = X_scaled.dot(eigvecs_top)
	print("X_lda_manual shape:", X_lda_manual.shape)

	# Reconstrução aproximada (de volta ao espaço escalado)
	X_recon_manual = X_lda_manual.dot(eigvecs_top.T)
	mse_manual = np.mean((X_scaled - X_recon_manual) ** 2)
	print("\nMSE de reconstrução (manual LDA, no espaço escalado):", mse_manual)

	# -----------------------------------------------------------
	# LDA com sklearn para comparação
	# -----------------------------------------------------------
	lda_sklearn = LinearDiscriminantAnalysis(n_components=k_lda)
	X_lda_sklearn = lda_sklearn.fit_transform(X_scaled, y)
	print("\nLDA sklearn - X_lda_sklearn shape:", X_lda_sklearn.shape)

	# scalings_ contém as direções discriminantes no sklearn (coeficiente linear)
	scalings = lda_sklearn.scalings_[:, :k_lda]
	print("scalings_ (sklearn) shape:", scalings.shape)

	# Avaliar acurácia após transformação LDA (manual vs sklearn)
	X_train_man, X_test_man, y_train_man, y_test_man = train_test_split(X_lda_manual, y, test_size=0.2, random_state=30, stratify=y)
	model_man = LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='multinomial')
	model_man.fit(X_train_man, y_train_man)
	acc_man = model_man.score(X_test_man, y_test_man)
	print("\nAccuracy (Logistic on LDA manual):", acc_man)

	X_train_sk, X_test_sk, y_train_sk, y_test_sk = train_test_split(X_lda_sklearn, y, test_size=0.2, random_state=30, stratify=y)
	model_sk = LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='multinomial')
	model_sk.fit(X_train_sk, y_train_sk)
	acc_sk = model_sk.score(X_test_sk, y_test_sk)
	print("Accuracy (Logistic on LDA sklearn):", acc_sk)

	# Comparações e prints adicionais
	print("\nComparações:")
	print("k_lda:", k_lda)
	print("Shapes - X_lda_manual:", X_lda_manual.shape, "X_lda_sklearn:", X_lda_sklearn.shape)
	print("Top eigenvalues (manual):", eigvals_top)
	print("Sklearn scalings (first 5 elements por componente):")
	print(np.round(scalings[:5, :min(3, scalings.shape[1])], 4))

	# Salva imagens das componentes discriminantes (manual e sklearn)
	plot_dir = Path(__file__).resolve().parent.parent / "outputs"
	plot_dir.mkdir(parents=True, exist_ok=True)
	for i in range(min(3, k_lda)):
		comp_manual_img = eigvecs_top[:, i].reshape(28, 28)
		plt.figure(figsize=(3, 3))
		plt.imshow(comp_manual_img, cmap="seismic")
		plt.title(f"LDA manual componente {i+1}")
		plt.axis("off")
		plt.savefig(plot_dir / f"lda_manual_comp_{i+1}.png")
		plt.close()

		comp_sklearn_img = scalings[:, i].reshape(28, 28)
		plt.figure(figsize=(3, 3))
		plt.imshow(comp_sklearn_img, cmap="seismic")
		plt.title(f"LDA sklearn componente {i+1}")
		plt.axis("off")
		plt.savefig(plot_dir / f"lda_sklearn_comp_{i+1}.png")
		plt.close()

	print("\nSaved LDA component images to outputs/ (manual and sklearn).")

	# -----------------------------------------------------------
	# Gerar relatório com todo o stdout e métricas
	# -----------------------------------------------------------
	out_dir = Path(__file__).resolve().parent.parent / "outputs"
	out_dir.mkdir(parents=True, exist_ok=True)
	report_path = out_dir / "lda_report.txt"
	try:
		with report_path.open("w", encoding="utf-8") as f:
			f.write("LDA Analysis Report\n")
			f.write(f"Generated: {datetime.utcnow().isoformat()} UTC\n\n")
			f.write(f"Dataset: MNIST (subset)\n")
			f.write(f"Num samples used: {X.shape[0]}\n")
			f.write(f"Feature size: {X.shape[1]} (flattened 28x28)\n\n")

			f.write("Baseline (LogisticRegression on scaled features)\n")
			f.write(f" - Accuracy (raw/scaled): {accuracy_raw:.6f}\n\n")

			f.write("LDA (manual)\n")
			f.write(f" - Components chosen (manual): {k_lda}\n")
			f.write(f" - Top eigenvalues (manual): {np.round(eigvals_top, 6).tolist()}\n")
			f.write(f" - MSE reconstruction (manual, scaled space): {mse_manual:.8f}\n")
			f.write(f" - Accuracy (Logistic after manual LDA): {acc_man:.6f}\n\n")

			f.write("LDA (sklearn)\n")
			f.write(f" - Components chosen (sklearn): {k_lda}\n")
			f.write(f" - Accuracy (Logistic after sklearn LDA): {acc_sk:.6f}\n")
			f.write(f" - scalings_ (first 10 elements de cada componente): {np.round(scalings[:10, :min(3, scalings.shape[1])], 4).tolist()}\n\n")

			f.write("Comparisons & Checks\n")
			f.write(f" - Shapes: manual X_lda={X_lda_manual.shape}, sklearn X_lda={X_lda_sklearn.shape}\n")
			f.write("\nFull stdout log:\n")
			full_log = _stdout_buf.getvalue()
			f.write(full_log)
		print(f"\nReport written to: {report_path}")
	except Exception as e:
		print("Failed to write report:", e)

	print("\nFim do script LDA.")


if __name__ == "__main__":
	main(max_samples=5000)


