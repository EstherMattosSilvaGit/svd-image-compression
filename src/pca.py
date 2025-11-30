import pandas as pd 
from sklearn.datasets import fetch_openml
from matplotlib import pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Links de pesquisa:
# https://youtu.be/cOCeXgMKrY8
# http://www2.ic.uff.br/~aconci/PCA-ACP.pdf
# https://www.datageeks.com.br/analise-de-componentes-principais
# https://www.datacamp.com/pt/tutorial/pca-analysis-r
# https://pt.wikipedia.org/wiki/Análise_de_componentes_principais
# https://www.ime.unicamp.br/~cnaber/aula_ACP_Ana_Multi_2S_2020.pdf
# https://www.ibm.com/br-pt/think/topics/principal-component-analysis
# https://statplace.com.br/blog/analise-de-componentes-principais
# https://www.unievangelica.edu.br/gc/imagens/file/mestrados/artigos/RTINF_003092.pdf
# https://numpy.org/doc/stable/reference/generated/numpy.reshape.html

# Regressão logística﻿ é um jeito de prever respostas que têm só duas possibilidades, 
# tipo "sim" ou "não", usando um conjunto de informações. Imagine que você quer saber 
# se uma pessoa vai comprar um produto (sim ou não), o modelo﻿ vai analisar 
# características das pessoas para tentar adivinhar essa resposta.

# No sklearn, você cria um modelo que "aprende" com dados que você já sabe a resposta 
# (treinamento). Depois, você usa esse modelo para prever novas situações. Isso é feito 
# com poucos passos bem simples. Você separa seus dados em duas partes: uma para ensinar 
# o modelo e outra para testar se ele aprendeu bem. Depois usa o modelo para prever e 
# ver o que ele acertou.
from sklearn.linear_model import LogisticRegression

#Essa parte agora vamos ter que fazer na mao, mas esta aqui pra sabermos como funciona 
# mais ou menos
from sklearn.decomposition import PCA
import sys
import io


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
		# Also write to the buffer (no newlines added)
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


# set up capture
_stdout_buf = io.StringIO()
_tee = Tee(sys.stdout, _stdout_buf)
sys.stdout = _tee


# Carregar MNIST via OpenML (mais completo que load_digits)
print("\nBuscando MNIST (OpenML) — caso não esteja em cache, será baixado...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
keys = list(mnist.keys()) if hasattr(mnist, 'keys') else []

print("\nDataset keys: ", keys)

# X e y completos, mas usaremos subset para testes rápidos
# Carrega todas as imagens (N×784) do MNIST na variável X_all
X_all = mnist.data
# Carrega todos os rótulos e garante que sejam inteiros (0-9)
y_all = mnist.target.astype(int)
# Limite máximo de amostras a usar para testes (evita usar todo o dataset por padrão)
max_samples = 5000
# Garante que não tentemos usar mais amostras do que o dataset possui
num_samples = min(max_samples, X_all.shape[0])
# Seleciona as primeiras num_samples amostras para X (subconjunto para testes)
X = X_all[:num_samples]
# Seleciona os rótulos correspondentes ao subconjunto
y = y_all[:num_samples]
# Mostra a forma do subconjunto (num_samples, 784)
print("\nShape (subset):", X.shape)

# First sample flattened
first_sample = X[0]
print("\nFirst sample (flattened 784): \n", first_sample)

# Converting into two dimensional array to be able to visualize using matplot library

# O método .reshape(28,28) é utilizado para transformar o vetor 784 em uma imagem 2D.
reshaped_data = first_sample.reshape(28,28)
print("\nReshaped data: \n", reshaped_data)

# Showing the first image from the reshaped data
print("\nFirst image showed.")
reshaped_first_sample = first_sample.reshape(28,28)
plt.figure(figsize=(4,4))
plt.imshow(reshaped_first_sample, cmap='gray', interpolation='nearest')
plt.title(f'Dígito: {y[0]}')
plt.axis('off')
plt.show()

# Showing the second image from the reshaped data
print("\nFirst image showed.")
reshaped_first_sample = X[1].reshape(28,28)
plt.figure(figsize=(4,4))
plt.imshow(reshaped_first_sample, cmap='gray', interpolation='nearest')
plt.title(f'Dígito: {y[1]}')
plt.axis('off')
plt.show()

# Os "targets" referem-se às classes ou rótulos associados a cada amostra no 
# conjunto de dados. No contexto do conjunto de dados load_digits, cada imagem 
# de dígito (0 a 9) tem um rótulo correspondente que indica qual número a imagem 
# representa.

# The target (usando subset y)
target = y
print("\nTarget: ", target)

# Unique targets
unique_target = np.unique(y)
print("\nUnique Targets: ", unique_target)

# Showing the 10th image from the reshaped data
print("\n9 image showed.")
reshaped_first_sample = X[9].reshape(28,28)
plt.figure(figsize=(4,4))
plt.imshow(reshaped_first_sample, cmap='gray', interpolation='nearest')
plt.title(f'Dígito: {y[9]}')
plt.axis('off')
plt.show()

# target from the 10th number
print(y[9])

# Data Frame (gera nomes de colunas se não existirem)
feature_names = [f"pixel_{r}_{c}" for r in range(28) for c in range(28)]
data_frame = pd.DataFrame(X, columns=feature_names)
print("\nData Frame: ", data_frame)

# Data frame object
print("\nData Frame object: ", data_frame.head())

# Describe
print("\nDescribe: ", data_frame.describe())

# Store as x and y
# Aqui estamos armazenando os dados em duas variáveis: x e y.
# A variável x contém o DataFrame que representa as características (pixels) das imagens de dígitos.
# Cada linha de x corresponde a uma imagem, e cada coluna representa um pixel da imagem.
# A variável y contém os rótulos (targets) associados a cada imagem, que indicam qual número (0 a 9) 
# cada imagem representa. Esses rótulos são usados para treinar o modelo de aprendizado de máquina, 
# permitindo que ele aprenda a associar as características das imagens (x) aos seus respectivos números (y).
# x já contém DataFrame com X (subset) e y já foi definido pelo subset acima
x = data_frame
# y definido mais acima

print("\nX and Y: ", x, y)

# Usando a biblioteca sklearn antes de construir meu modelo de aprendizado de máquina
# Primeiro, normalizamos os dados usando StandardScaler. Isso significa que estamos 
# ajustando os dados para que tenham média 0 e desvio padrão 1, o que ajuda o modelo 
# a aprender melhor, pois evita que algumas características dominem outras devido 
# a escalas diferentes.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(x)

print("\nX Scaled: ", X_scaled)

# Em seguida, dividimos os dados em conjuntos de treinamento e teste. O conjunto de 
# treinamento (80%) é usado para ensinar o modelo, enquanto o conjunto de teste (20%) 
# é usado para avaliar o desempenho do modelo após o treinamento.
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=30)

# Criamos um modelo de Regressão Logística e o treinamos com os dados de treinamento.
model_raw = LogisticRegression()
model_raw.fit(X_train, y_train)

# Avaliamos a precisão do modelo usando o conjunto de teste. A precisão nos diz 
# quão bem o modelo está se saindo em prever os rótulos corretos para os dados 
# que não viu durante o treinamento.
accuracy_raw = model_raw.score(X_test, y_test)

print("\nAccuracy (raw / scaled features):", accuracy_raw)

# Aplicando PCA manualmente (covariância + autovalores/autovetores)
# O PCA manual segue estes passos principais: centralizar os dados (se necessário),
# calcular a matriz de covariância, encontrar autovalores e autovetores (eigen),
# selecionar os principais componentes e projetar os dados.

# Re-uso do X_scaled (já centrado pelo StandardScaler) para ter comparação justa
X_centered = X_scaled

# 2. Matriz de covariância (usando rowvar=False para deixar colunas como variáveis)
cov_matrix = np.cov(X_centered, rowvar=False)

# 3. Autovalores e autovetores (usar eigh -- mais estável para matrizes simétricas)
eigenvalues_all, eigenvectors_all = np.linalg.eigh(cov_matrix)

# 4. Ordenar por autovalores (maior primeiro)
idx_all = eigenvalues_all.argsort()[::-1]
eigenvalues_all = eigenvalues_all[idx_all]
eigenvectors_all = eigenvectors_all[:, idx_all]

# 5. Selecionar componentes que mantêm 95% da variância
total_var_manual = np.sum(eigenvalues_all)
cum_var_manual = np.cumsum(eigenvalues_all / total_var_manual)
num_components_manual = int(np.argmax(cum_var_manual >= 0.95) + 1)

# 6. Matriz de projeção (primeiros num_components_manual vectores)
projection_matrix_manual = eigenvectors_all[:, :num_components_manual]

# 7. Transformar dados (equivalente a pca.fit_transform)
X_pca_manual = X_centered.dot(projection_matrix_manual)

print("Manual PCA - Shape:", X_pca_manual.shape)
print("Manual explained variance ratio:", eigenvalues_all[:num_components_manual] / total_var_manual)
print("Manual Components:", num_components_manual)

eigenvalues_manual = eigenvalues_all
eigenvectors_manual = eigenvectors_all


# Aplicando PCA com a biblioteca (sklearn) para comparação
# Observação: o sklearn PCA centraliza os dados por padrão. Usamos X_centered
# (que já é X_scaled / centrado) para garantir que ambas as abordagens partam do
# mesmo ponto.
pca_sklearn = PCA(0.95)
X_pca_sklearn = pca_sklearn.fit_transform(X_centered)

print("\nSklearn PCA - Shape:", X_pca_sklearn.shape)
print("\nX_pca_sklearn: ", X_pca_sklearn)
print("\nSklearn explained variance ratio: ", pca_sklearn.explained_variance_ratio_)
print("\nSklearn Components: ", pca_sklearn.n_components_)

# Define X_pca for downstream compatibility (podemos usar X_pca_sklearn)
X_pca = X_pca_sklearn

# Comparativo entre manual e sklearn
num_components_sklearn = pca_sklearn.n_components_
min_components = min(num_components_manual, num_components_sklearn)

explained_variance_ratio_manual = eigenvalues_all / total_var_manual
print("\nNumber of components (manual):", num_components_manual)
print("Number of components (sklearn):", num_components_sklearn)
print("Shapes: manual X_pca", X_pca_manual.shape, "sklearn X_pca", X_pca_sklearn.shape)
print("Explained variance (manual, first k):", explained_variance_ratio_manual[:min_components])
print("Explained variance (sklearn, first k):", pca_sklearn.explained_variance_ratio_[:min_components])
print("Cum. variance (manual, first k):", np.cumsum(explained_variance_ratio_manual[:min_components]))
print("Cum. variance (sklearn, first k):", np.cumsum(pca_sklearn.explained_variance_ratio_[:min_components]))
print("Explained variance close?", np.allclose(np.cumsum(explained_variance_ratio_manual[:min_components]), np.cumsum(pca_sklearn.explained_variance_ratio_[:min_components]), atol=1e-6))


# Comparar resultados e reconstrução (MSE)
print("\nComparação final:")
print("Sklearn components:", pca_sklearn.n_components_)
print("Manual components:", num_components_manual)
print("Shapes: Manual X_pca", X_pca_manual.shape, "Sklearn X_pca", X_pca_sklearn.shape)

# Reconstruções (voltar para o espaço original e comparar erro quadrático médio)
X_recon_manual = X_pca_manual.dot(projection_matrix_manual.T)
X_recon_sklearn = pca_sklearn.inverse_transform(X_pca_sklearn)

mse_manual = np.mean((X_centered - X_recon_manual) ** 2)
mse_sklearn = np.mean((X_centered - X_recon_sklearn) ** 2)

print("MSE reconstrução (manual):", mse_manual)
print("MSE reconstrução (sklearn):", mse_sklearn)

# Podemos usar esse novo DataFrame reduzido (X_pca) para treinar nosso modelo.
# Aqui, dividimos os dados reduzidos em conjuntos de treinamento e teste, 
# onde 80% dos dados serão usados para treinar o modelo e 20% para testá-lo.
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca, y, test_size=0.2, random_state=30)

# Treinamento e avaliação em PCA preservando 95% da variância (sklearn PCA)
model_pca95 = LogisticRegression(max_iter=1000)
model_pca95.fit(X_train_pca, y_train_pca)
score_pca95 = model_pca95.score(X_test_pca, y_test_pca)
print(f"\nScore PCA (95% var, n_components={pca_sklearn.n_components_}): ", score_pca95)


# A seguir, criamos um PCA 2D para visualização e análise.
# Usamos X_centered para manter consistência entre as abordagens.
pca_2d = PCA(n_components=2)
X_pca_2d = pca_2d.fit_transform(X_centered)

print("\nShape pca 2D: ", X_pca_2d.shape)
print("\nDataframe pca 2D: ", X_pca_2d)
print("\nExplained variance ratio pca 2D: ", pca_2d.explained_variance_ratio_)

# Treinar um modelo simples usando as 2 componentes principais e medir acurácia
X_train_2d, X_test_2d, y_train_2d, y_test_2d = train_test_split(X_pca_2d, y, test_size=0.2, random_state=30)
model_pca2d = LogisticRegression(max_iter=1000)
model_pca2d.fit(X_train_2d, y_train_2d)
score_pca2d = model_pca2d.score(X_test_2d, y_test_2d)
print("\nScore PCA 2D: ", score_pca2d)

# (Observação: o treinamento do PCA 95% já foi feito acima com model_pca95)
# Imprimimos um resumo com as métricas para facilitar a comparação.
print("\nResumo final:")
print("Accuracy (raw):", accuracy_raw)
print(f"Accuracy (PCA 95%, n_components={pca_sklearn.n_components_}):", score_pca95)
print(f"Accuracy (PCA 2D, n_components={pca_2d.n_components_}):", score_pca2d)

# --- Save a plain text report to outputs/pca_report.txt ---
from pathlib import Path
from datetime import datetime

out_dir = Path(__file__).resolve().parent.parent / "outputs"
out_dir.mkdir(parents=True, exist_ok=True)
report_path = out_dir / "pca_report.txt"
try:
	with report_path.open("w", encoding="utf-8") as f:
		f.write("PCA Analysis Report\n")
		f.write(f"Generated: {datetime.utcnow().isoformat()} UTC\n\n")
		f.write(f"Dataset: MNIST (subset)\n")
		f.write(f"Num samples used: {X.shape[0]}\n")
		f.write(f"Feature size: {X.shape[1]} (flattened 28x28)\n\n")

		f.write("Baseline (LogisticRegression on scaled features)\n")
		f.write(f" - Accuracy (raw/scaled): {accuracy_raw:.6f}\n\n")

		f.write("PCA (manual)\n")
		f.write(f" - Components chosen (manual): {num_components_manual}\n")
		f.write(f" - MSE reconstruction (manual): {mse_manual:.8f}\n\n")

		f.write("PCA (sklearn 95% variance)\n")
		f.write(f" - Components chosen (sklearn): {pca_sklearn.n_components_}\n")
		f.write(f" - Accuracy (logistic after PCA 95%): {score_pca95:.6f}\n")
		f.write(f" - MSE reconstruction (sklearn): {mse_sklearn:.8f}\n")
		f.write(f" - Explained variance ratio (first 10): {np.round(pca_sklearn.explained_variance_ratio_[:10], 6).tolist()}\n\n")

		f.write("PCA 2D\n")
		f.write(f" - Components: {pca_2d.n_components_}\n")
		f.write(f" - Accuracy (logistic on 2D): {score_pca2d:.6f}\n")
		f.write(f" - Explained variance ratio (2D): {np.round(pca_2d.explained_variance_ratio_, 6).tolist()}\n\n")

		f.write("Comparisons & Checks\n")
		f.write(f" - Manual vs sklearn #components: manual={num_components_manual}, sklearn={pca_sklearn.n_components_}\n")
		f.write(f" - Shapes: manual X_pca={X_pca_manual.shape}, sklearn X_pca={X_pca_sklearn.shape}\n")
		f.write(f" - Explained variance close? {np.allclose(np.cumsum(explained_variance_ratio_manual[:min_components]), np.cumsum(pca_sklearn.explained_variance_ratio_[:min_components]), atol=1e-6)}\n\n")

		f.write("Note: Reconstructions and MSEs are computed in centered/scaled space. To view reconstructed images in original pixel scale, apply scaler.inverse_transform to the reconstructed arrays.\n")
		# Add full captured stdout log
		f.write("\nFull stdout log:\n")
		try:
			# _stdout_buf holds all printed output
			full_log = _stdout_buf.getvalue()
			f.write(full_log)
		except Exception as e:
			f.write(f"Failed to capture full log: {e}\n")
	print(f"\nReport written to: {report_path}")
except Exception as e:
	print("Failed to write report:", e)