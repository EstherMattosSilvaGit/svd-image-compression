import pandas as pd 
from sklearn.datasets import load_digits
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


# Load the dataset using the load digits function
dataset = load_digits()
keys = dataset.keys()

# Dataset keys
print("\nDataset keys: ", keys)

# The amount of samples that have in database plus the amount of columns each sample have
shape = dataset.data.shape
print("\nShape: ", shape)

# First sample of the dataset
first_sample = dataset.data[0]
print("\nFirst sample: \n", first_sample)

# Converting into two dimensional array to be able to visualize using matplot library

# O método .reshape(8,8) é utilizado para alterar a forma (dimensões) de um array sem 
# modificar os dados originais. No seu exemplo, dataset.data[0] é um array 1D com 
# 64 elementos (representando os pixels de uma imagem 8x8). O reshape(8,8) transforma 
# esse array 1D em uma matriz 2D com 8 linhas e 8 colunas, permitindo a visualização 
# da imagem correspondente.
reshaped_data = dataset.data[0].reshape(8,8)
print("\nReshaped data: \n", reshaped_data)

# Showing the first image from the reshaped data
print("\nFirst image showed.")
reshaped_first_sample = first_sample.reshape(8,8)
plt.figure(figsize=(4,4))
plt.imshow(reshaped_first_sample, cmap='gray', interpolation='nearest')
plt.title(f'Dígito: {dataset.target[0]}')
plt.axis('off')
plt.show()

# Showing the second image from the reshaped data
print("\nFirst image showed.")
reshaped_first_sample = dataset.data[1].reshape(8,8)
plt.figure(figsize=(4,4))
plt.imshow(reshaped_first_sample, cmap='gray', interpolation='nearest')
plt.title(f'Dígito: {dataset.target[1]}')
plt.axis('off')
plt.show()

# Os "targets" referem-se às classes ou rótulos associados a cada amostra no 
# conjunto de dados. No contexto do conjunto de dados load_digits, cada imagem 
# de dígito (0 a 9) tem um rótulo correspondente que indica qual número a imagem 
# representa.

# The target
target = dataset.target
print("\nTarget: ", target)

# Unique targets
unique_target = np.unique(dataset.target)
print("\nUnique Targets: ", unique_target)

# Showing the 10th image from the reshaped data
print("\n9 image showed.")
reshaped_first_sample = dataset.data[9].reshape(8,8)
plt.figure(figsize=(4,4))
plt.imshow(reshaped_first_sample, cmap='gray', interpolation='nearest')
plt.title(f'Dígito: {dataset.target[9]}')
plt.axis('off')
plt.show()

# target from the 10th number
print(dataset.target[9])

# Data Frame
# O DataFrame é uma estrutura de dados bidimensional fornecida pela biblioteca Pandas, 
# que permite armazenar e manipular dados de forma tabular. Neste caso, estamos criando 
# um DataFrame a partir dos dados do conjunto de dígitos (dataset.data), onde cada linha 
# representa uma amostra (imagem de dígito) e cada coluna representa uma característica 
# (pixel da imagem). Os nomes das colunas são fornecidos por dataset.feature_names, 
# que contém os rótulos correspondentes a cada pixel. Isso facilita a análise e 
# manipulação dos dados, permitindo operações como filtragem, agregação e visualização.
data_frame = pd.DataFrame(dataset.data, columns=dataset.feature_names)
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
x = data_frame
y = dataset.target

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