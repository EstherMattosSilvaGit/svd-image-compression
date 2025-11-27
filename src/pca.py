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
model = LogisticRegression()
model.fit(X_train, y_train)

# Avaliamos a precisão do modelo usando o conjunto de teste. A precisão nos diz 
# quão bem o modelo está se saindo em prever os rótulos corretos para os dados 
# que não viu durante o treinamento.
accuracy = model.score(X_test, y_test)

print("\nAccuracy: ", accuracy)

#Aplicando pca manualmente
# Passos para Calcular PCA Manualmente
# O PCA manual segue estes passos principais: centralizar os dados 
# (subtrair a média), calcular a matriz de covariância, encontrar autovalores e 
# autovetores, selecionar os principais componentes e projetar os dados.​

# 1. Centralizar os dados (média = 0)
scaler = StandardScaler()
X_centered = scaler.fit_transform(x)

# 2. Matriz de covariância
cov_matrix = np.cov(X_centered.T)

# 3. Autovalores e autovetores
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# 4. Ordenar por autovalores (maior primeiro)
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# 5. Selecionar componentes que mantêm 95% da variância
total_var = np.sum(eigenvalues)
cum_var = np.cumsum(eigenvalues / total_var)
n_components = np.argmax(cum_var >= 0.95) + 1

# 6. Matriz de projeção (primeiros n_components)
projection_matrix = eigenvectors[:, :n_components]

# 7. Transformar dados (equivalente a pca.fit_transform)
X_pca = X_centered.dot(projection_matrix)

print("Shape:", X_pca.shape)
print("Explained variance ratio:", eigenvalues[:n_components] / total_var)
print("Components:", n_components)


# Aplicando pca com a biblioteca para comparação
# Por fim, aplicamos PCA (Análise de Componentes Principais) para reduzir a 
# dimensionalidade dos dados, mantendo 95% da informação original. Isso ajuda 
# a simplificar o modelo e pode melhorar o desempenho.
pca = PCA(0.95)
X_pca = pca.fit_transform(x)

# Aqui, verificamos a nova forma dos dados após a redução de dimensionalidade, 
# que agora tem menos colunas (características).
print("\nShape: ", X_pca.shape)

# Calcula colunas
print("\nX_pca: ", X_pca)

# Explained varianca ratio: porcentagem de variacao para cada item
# A razão da variância explicada é uma métrica que indica a quantidade 
# de variância nos dados originais que é capturada por cada componente 
# principal após a aplicação da Análise de Componentes Principais (PCA).
print("\nExplained variance ratio: ", pca.explained_variance_ratio_)

#Components number
print("\nComponents: ", pca.n_components_)


# Comparar resultados
print("Sklearn components:", pca_sklearn.n_components_)
print("Shapes iguais?", X_pca.shape == X_pca_sklearn.shape)

# Podemos usar esse novo DataFrame reduzido (X_pca) para treinar nosso modelo.
# Aqui, dividimos os dados reduzidos em conjuntos de treinamento e teste, 
# onde 80% dos dados serão usados para treinar o modelo e 20% para testá-lo.
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca, y, test_size=0.2, random_state=30)

# Criamos um modelo de Regressão Logística com um número máximo de iterações definido.
model = LogisticRegression(max_iter=1000)

# Treinamos o modelo usando os dados de treinamento reduzidos.
model.fit(X_train_pca, y_train_pca)

# Avaliamos a precisão do modelo usando o conjunto de teste reduzido.
# O score nos diz quão bem o modelo está se saindo em prever os rótulos corretos.
print("\nScore Pca: ", model.score(X_test_pca, y_test_pca))

# A seguir, criamos um novo PCA especificando o número de componentes que queremos.
# Neste caso, estamos reduzindo os dados para 2 componentes principais.
pca = PCA(n_components=2)
X_pca = pca.fit_transform(x)

# Mostramos a nova forma dos dados após a redução de dimensionalidade.
print("\nShape pca: ", X_pca.shape)

# Exibimos os dados transformados pelo PCA.
print("\nDataframe pca: ", X_pca)

# Mostramos a proporção da variância explicada por cada componente principal.
print("\nExplained variance ratio pca: ", pca.explained_variance_ratio_)

# Após treinar o modelo com os dados reduzidos, dividimos novamente os dados em 
# conjuntos de treinamento e teste para avaliar o desempenho do modelo.
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca, y, test_size=0.2, random_state=30)

# Criamos e treinamos novamente o modelo de Regressão Logística.
model = LogisticRegression(max_iter=1000)
model.fit(X_train_pca, y_train_pca)

# Avaliamos a precisão do modelo com os dados reduzidos.
print("\nScore Pca: ", model.score(X_test_pca, y_test_pca))