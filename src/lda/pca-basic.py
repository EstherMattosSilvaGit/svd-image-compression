import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

# 1. Carrega o MNIST do OpenML
mnist = fetch_openml('mnist_784', version=1, as_frame=False)

X = mnist.data.astype(np.float32)  # (70000, 784)
y = mnist.target.astype(int)       # rótulos 0..9

# 2. Normaliza para [0, 1]
X /= 255.0

# 3. Divide em treino e teste (para organizar)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(X_train.shape, X_test.shape)

from sklearn.decomposition import PCA

# 4. Para ficar leve, foi utilizado apenas 3000 amostras
n_samples = 3000
X_subset = X_train[:n_samples]
y_subset = y_train[:n_samples]

# 5. PCA com 2 componentes
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_subset)

print("Formato após PCA:", X_pca.shape)

plt.figure(figsize=(6, 5))

scatter = plt.scatter(
    X_pca[:, 0], X_pca[:, 1],
    c=y_subset, cmap='tab10', s=10, alpha=0.7
)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("MNIST projetado em 2D com PCA")
plt.colorbar(scatter, label="Dígito")
plt.tight_layout()
plt.show()