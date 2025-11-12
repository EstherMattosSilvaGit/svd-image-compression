# 🧮 PAC – Image Compression using SVD / Compressão de Imagens usando SVD

This repository contains a small project for image compression using Singular Value Decomposition (SVD).

Use the links below to jump directly to the English or Portuguese version of this README.

- [English](#english)
- [Português (BR)](#portuguese)

---

## English

### Overview

This project demonstrates how Singular Value Decomposition (SVD) can be used to compress grayscale images by representing an image as a matrix and reconstructing it using only the k largest singular values. The implementation emphasizes manual linear algebra operations to match the goals of a Computational Linear Algebra course (PAC).

### Goal

Implement SVD-based compression to trade off between compression ratio and visual quality by reconstructing an image using only the top-k singular values.

### Steps

1. Read and convert an image to a numeric matrix (grayscale).
2. Implement linear algebra operations manually (matrix multiplication, diagonal matrix creation, norm calculation).
3. Compute SVD and reconstruct the image with different k values (e.g. 5, 20, 50, 100).
4. Display the original and reconstructed images and a plot of reconstruction error using Matplotlib.

### Libraries

• Pillow (PIL) — image I/O and conversion
• NumPy — image array handling (limited use; core linear algebra implemented manually)
• Matplotlib — visualization
• math / builtins — basic numerical operations

> Note: Core linear algebra routines (multiplication, diag creation, norms) are implemented by hand for learning purposes.

### Expected Results

• Side-by-side images showing the original and reconstructions for different k values.
• A plot showing reconstruction error vs k.

---

<a id="portuguese"></a>
## Português (BR)

### Visão geral

Este projeto demonstra como a Decomposição em Valores Singulares (SVD) pode ser usada para comprimir imagens em escala de cinza, representando a imagem como uma matriz e reconstruindo-a usando apenas os k maiores valores singulares. A implementação enfatiza operações de álgebra linear feitas manualmente, alinhadas ao objetivo da disciplina de Processamento de Álgebra Computacional (PAC).

### Objetivo

Implementar a compressão baseada em SVD para demonstrar o trade-off entre razão de compressão e qualidade visual, reconstruindo a imagem com apenas os top-k valores singulares.

### Etapas

1. Leitura e conversão da imagem para matriz numérica (tons de cinza).
2. Implementação manual das operações de Álgebra Linear (multiplicação de matrizes, criação de matriz diagonal, cálculo de norma).
3. Cálculo da SVD e reconstrução da imagem com diferentes valores de k (ex.: 5, 20, 50, 100).
4. Exibição da imagem original e das reconstruções e um gráfico de erro de reconstrução usando Matplotlib.

### Bibliotecas

• Pillow (PIL) — leitura e conversão de imagem
• NumPy — manipulação de arrays de imagem (uso limitado; as rotinas principais são manuais)
• Matplotlib — visualização
• math / builtins — operações numéricas básicas

> Observação: As rotinas principais de álgebra linear foram implementadas manualmente para fins didáticos.

### Resultados Esperados

• Imagens lado a lado: original e reconstruções para diferentes k.
• Gráfico com o erro de reconstrução em função de k.

---

## Authors / Autoria

• Esther Mattos
• Thalisson Souza

Universidade — 2025

