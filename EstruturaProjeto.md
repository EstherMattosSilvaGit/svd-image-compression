
# 🧮 Estrutura e Implementação do Projeto – PAC SVD Compressão de Imagens

## 🧱 Estrutura recomendada do projeto

```
pac-svd-compressao-de-imagens/
│
├── src/
│   ├── algebra_utils.py        # Funções manuais de álgebra linear
│   ├── svd_manual.py           # Cálculo e reconstrução da imagem (usando as funções acima)
│   ├── image_utils.py          # Leitura, conversão e exibição de imagens
│   └── main.py                 # Arquivo principal que executa o projeto
│
├── imagens/
│   └── imagem_teste.jpg        # Imagem usada no experimento
│
├── resultados/
│   └── (as imagens e gráficos gerados podem ser salvos aqui)
│
├── README.md                   # Descrição do projeto
└── requirements.txt            # Lista de bibliotecas usadas (Pillow, Matplotlib)
```

---

## 🐍 **1. `algebra_utils.py`**

Contém as **funções manuais de Álgebra Linear**, que substituem o NumPy.

```python
import math

def matmul(A, B):
    """Multiplica duas matrizes A e B manualmente."""
    m, n = len(A), len(A[0])
    n2, p = len(B), len(B[0])
    assert n == n2, "Dimensões incompatíveis para multiplicação"
    C = [[0 for _ in range(p)] for _ in range(m)]
    for i in range(m):
        for j in range(p):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]
    return C


def transpose(A):
    """Retorna a transposta de uma matriz A."""
    return [[A[j][i] for j in range(len(A))] for i in range(len(A[0]))]


def diag(v):
    """Cria uma matriz diagonal a partir de um vetor."""
    n = len(v)
    D = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        D[i][i] = v[i]
    return D


def norma(A):
    """Calcula a norma de Frobenius de uma matriz."""
    soma = 0
    for i in range(len(A)):
        for j in range(len(A[0])):
            soma += A[i][j] ** 2
    return math.sqrt(soma)
```

---

## 🧮 **2. `svd_manual.py`**

Implementa a **parte conceitual da SVD**, reconstruindo a imagem com os *k* maiores valores singulares.

Você não precisa fazer a decomposição completa (autovalores/autovetores), apenas **simular o comportamento** da SVD explicando que ela viria de ( A^T A ).

```python
from algebra_utils import matmul, transpose, diag, norma

def reconstruir_imagem(U, S, Vt, k):
    """Reconstrói a imagem com os k maiores valores singulares."""
    U_k = [linha[:k] for linha in U]
    Vt_k = [Vt[i] for i in range(k)]
    S_k = diag(S[:k])
    return matmul(matmul(U_k, S_k), Vt_k)


def calcular_erro(A_original, A_reconstruida):
    """Calcula o erro relativo entre a imagem original e a reconstruída."""
    numerador = norma(subtrair(A_original, A_reconstruida))
    denominador = norma(A_original)
    return numerador / denominador


def subtrair(A, B):
    """Subtrai duas matrizes A e B."""
    m, n = len(A), len(A[0])
    C = [[A[i][j] - B[i][j] for j in range(n)] for i in range(m)]
    return C
```

---

## 🖼️ **3. `image_utils.py`**

Gerencia a **leitura, conversão e visualização das imagens**.

```python
from PIL import Image
import matplotlib.pyplot as plt

def carregar_imagem(caminho):
    """Abre uma imagem e converte para tons de cinza (matriz)."""
    img = Image.open(caminho).convert('L')
    largura, altura = img.size
    matriz = [[img.getpixel((x, y)) for x in range(largura)] for y in range(altura)]
    return matriz

def exibir_imagem(matriz, titulo="Imagem"):
    """Exibe uma matriz como imagem em tons de cinza."""
    plt.imshow(matriz, cmap='gray')
    plt.title(titulo)
    plt.axis('off')
    plt.show()
```

---

## 🚀 **4. `main.py`**

Arquivo principal que une tudo e roda o projeto.

```python
from image_utils import carregar_imagem, exibir_imagem
from algebra_utils import matmul, norma
from svd_manual import reconstruir_imagem, calcular_erro

def main():
    # 1. Carregar imagem
    A = carregar_imagem("imagens/imagem_teste.jpg")
    print(f"Dimensões da imagem: {len(A)}x{len(A[0])}")

    # 2. Simular decomposição SVD (você pode carregar U, S, Vt de um exemplo ou gerar fictícios)
    U = [[1, 0], [0, 1]]
    S = [200, 100]
    Vt = [[1, 0], [0, 1]]

    # 3. Reconstrução (simulação com dados pequenos)
    A_recon = reconstruir_imagem(U, S, Vt, k=2)
    exibir_imagem(A, "Imagem Original")
    exibir_imagem(A_recon, "Imagem Reconstruída (k=2)")

if __name__ == "__main__":
    main()
```

---

## 📁 **requirements.txt**

Coloque apenas as bibliotecas externas:

```
Pillow
matplotlib
```

---

## 📘 **Funções principais do projeto**

| Arquivo            | Funções principais                                | Descrição                                     |
| ------------------ | ------------------------------------------------- | --------------------------------------------- |
| `algebra_utils.py` | `matmul`, `transpose`, `diag`, `norma`            | Operações básicas de Álgebra Linear (manuais) |
| `svd_manual.py`    | `reconstruir_imagem`, `subtrair`, `calcular_erro` | Parte teórica da SVD e reconstrução           |
| `image_utils.py`   | `carregar_imagem`, `exibir_imagem`                | Manipulação e exibição de imagens             |
| `main.py`          | `main()`                                          | Coordena a execução geral do projeto          |

---

## 🧠 Dica extra (pra deixar perfeito)

No seu **relatório**, você pode mostrar o diagrama de dependência entre os arquivos, tipo assim:

```
main.py
 ├── image_utils.py → leitura e exibição
 ├── algebra_utils.py → operações matemáticas
 └── svd_manual.py → reconstrução e análise
```
