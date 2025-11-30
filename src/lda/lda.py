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

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# 4. LDA com 2 componentes (no MNIST, max é 9, pois são 10 classes)
lda = LDA(n_components=2)

n_samples_lda = 3000
X_sub_lda = X_train[:n_samples_lda]
y_sub_lda = y_train[:n_samples_lda]

X_lda = lda.fit_transform(X_sub_lda, y_sub_lda)

print("Formato após LDA:", X_lda.shape)

plt.figure(figsize=(6, 5))

scatter = plt.scatter(
    X_lda[:, 0], X_lda[:, 1],
    c=y_sub_lda, cmap='tab10', s=10, alpha=0.7
)

plt.xlabel("LD1")
plt.ylabel("LD2")
plt.title("MNIST projetado em 2D com LDA")
plt.colorbar(scatter, label="Dígito")
plt.tight_layout()
plt.show()