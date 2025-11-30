import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

# 1. Carrega o MNIST do OpenML
mnist = fetch_openml('mnist_784', version=1, as_frame=False)

X = mnist.data.astype(np.float32)  # (70000, 784)
y = mnist.target.astype(int)       # rotulos 0..9

# 2. Normaliza para [0, 1]
X /= 255.0

# 3. Divide em treino e teste (para organizar)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(X_train.shape, X_test.shape)

### SVD no MNIST (decompor e reconstruir uma imagem) ###

# 4. Escolhe uma imagem qualquer do conjunto de treino
idx = 0
img = X_train[idx].reshape(28, 28)

plt.imshow(img, cmap='gray')
plt.title(f"Imagem original - dígito {y_train[idx]}")
plt.axis('off')
plt.show()

# 5. Aplica SVD na matriz 28x28
U, Sigma, VT = np.linalg.svd(img, full_matrices=False)

print("Formatos:", U.shape, Sigma.shape, VT.shape)

def reconstrucao_svd(U, Sigma, VT, k):
    """
    Reconstrói a imagem usando apenas os k maiores valores singulares.
    """
    Uk = U[:, :k]
    Sk = np.diag(Sigma[:k])
    VTk = VT[:k, :]
    return Uk @ Sk @ VTk

ks = [5, 10, 20, 50]

plt.figure(figsize=(10, 3))

for i, k in enumerate(ks, 1):
    img_rec = reconstrucao_svd(U, Sigma, VT, k)
    
    plt.subplot(1, len(ks), i)
    plt.imshow(img_rec, cmap='gray')
    plt.title(f"k = {k}")
    plt.axis('off')

plt.suptitle("Reconstruções da imagem com diferentes k (SVD)")
plt.tight_layout()
plt.show()