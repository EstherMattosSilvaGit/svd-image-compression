import pandas as pd 
from sklearn.datasets import load_digits
from matplotlib import pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

#Essa parte agora vamos ter que fazer na mao, mas esta aqui pra sabermos como funciona mais ou menos
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
reshaped_data = dataset.data[0].reshape(8,8)
print("\nReshaped data: \n", reshaped_data)

# Showing the first image from the reshaped data
# print("\nFirst image showed.")
# reshaped_first_sample = first_sample.reshape(8,8)
# plt.figure(figsize=(4,4))
# plt.imshow(reshaped_first_sample, cmap='gray', interpolation='nearest')
# plt.title(f'Dígito: {dataset.target[0]}')
# plt.axis('off')
# plt.show()

# # Showing the second image from the reshaped data
# print("\nFirst image showed.")
# reshaped_first_sample = dataset.data[1].reshape(8,8)
# plt.figure(figsize=(4,4))
# plt.imshow(reshaped_first_sample, cmap='gray', interpolation='nearest')
# plt.title(f'Dígito: {dataset.target[1]}')
# plt.axis('off')
# plt.show()


# The target
target = dataset.target
print("\nTarget: ", target)


# Unique targets
unique_target = np.unique(dataset.target)
print("\nUnique Targets: ", unique_target)

# Showing the 10th image from the reshaped data
# print("\n9 image showed.")
# reshaped_first_sample = dataset.data[9].reshape(8,8)
# plt.figure(figsize=(4,4))
# plt.imshow(reshaped_first_sample, cmap='gray', interpolation='nearest')
# plt.title(f'Dígito: {dataset.target[9]}')
# plt.axis('off')
# plt.show()

# target from the 10th number
print(dataset.target[9])

# Data Frame
data_frame = pd.DataFrame(dataset.data, columns=dataset.feature_names)
print("\nData Frame: ", data_frame)


# Data frame object
print("\nData Frame object: ", data_frame.head())

# Describe
print("\nDescribe: ", data_frame.describe())

# Store as x and y
x = data_frame
y = dataset.target

print("\nX and Y: ", x, y)


# Using the sklearn library before building my machine learn model
scaler = StandardScaler()
X_scaled = scaler.fit_transform(x)

print("\nX Scaled: ", X_scaled)

# train the split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=30)

model = LogisticRegression()
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)

print("\nAccuracy: ", accuracy)

# Give me retain 95% of information
pca = PCA(0.95)
X_pca = pca.fit_transform(x)
# Diminuiu a quantidade de colunas para 29
print("\nShape: ", X_pca.shape)

# Calcula colunas
print("\nX_pca: ", X_pca)

# Explained varianca ratio: porcentagem de variacao para cada item
print("\nExplained variance ratio: ", pca.explained_variance_ratio_)

#Components number
print("\nComponents: ", pca.n_components_)

#Podemos usar esse novo data frame para treinar nosso model
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca, y, test_size=0.2, random_state=30)
model = LogisticRegression(max_iter=1000)
model.fit(X_train_pca, y_train_pca)
print("\nScore Pca: ", model.score(X_test_pca, y_test_pca))

#Create PCA by supplying the components explicity

pca = PCA(n_components=2)
X_pca = pca.fit_transform(x)
print("\nShape pca: ", X_pca.shape)
print("\nDataframe pca: ", X_pca)
print("\nExplained variance ratio pca: ", pca.explained_variance_ratio_)

# Usando apos treinar o codigo
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca, y, test_size=0.2, random_state=30)
model = LogisticRegression(max_iter=1000)
model.fit(X_train_pca, y_train_pca)
# Como podemos ver reduziu bastante
print("\nScore Pca: ", model.score(X_test_pca, y_test_pca))