# Técnicas de Redução de Dimensionalidade — Relatório

Professor:
- Dr. Marcos Lage - mlage@ic.uff.br

Alunos:
- Esther Mattos - esthermattos@id.uff.br
- Talisson Souza - talisedu@gmail.com

---

I. INTRODUÇÃO

O crescimento acelerado da geração de dados em domínios como saúde, finanças, IoT e sistemas urbanos resultou em bases cada vez mais complexas e de alta dimensionalidade. Em muitos casos, os dados apresentam centenas ou milhares de atributos, frequentemente redundantes ou pouco informativos. Esse cenário impõe desafios relevantes para técnicas de Machine Learning (ML) e Visual Analytics, uma vez que o aumento da dimensionalidade tende a degradar o desempenho de algoritmos, intensificar ruídos, dificultar visualização e tornar operações computacionalmente inviáveis, fenômeno conhecido como maldição da dimensionalidade.

Para mitigar esse problema, diversas técnicas de redução de dimensionalidade foram desenvolvidas. Neste trabalho, estudamos e aplicamos o Principal Component Analysis (PCA) ao dataset MNIST, ilustrando compressão, reconstrução e projeções visuais.

II. OBJETIVO

Demonstrar e validar, de forma prática, o uso do PCA como técnica de redução de dimensionalidade aplicada a imagens (MNIST). Especificamente:
- Demonstrar como o PCA é obtido usando Álgebra Linear (SVD)
- Visualizar a variância explicada cumulativa e selecionar k componentes
- Avaliar qualidade das reconstruções (MSE) para diferentes k
- Visualizar projeção 2D (PC1 vs PC2) para inspecionar separação de classes

III. FUNDAMENTAÇÃO TEÓRICA

PCA é um método linear que projeta os dados em um subespaço de menor dimensão que preserva a máxima variância. Formalmente:
- Dados: X ∈ R^{N×D}
- Média por coluna: μ = (1/N) Σ x_i
- Centralização: X_c = X − μ
- Decomposição: X_c = U Σ V^T
- Componentes: autovetores correspondentes às colunas de V
- Projeção: scores = X_c V_k^T
- Reconstrução: X_hat = scores V_k + μ

O PCA pode ser obtido a partir da decomposição espectral da matriz de covariância (autovalores/autovetores) ou via SVD direto em X_c, que é numericamente estável.

IV. METODOLOGIA

- Implementação manual em Python, usando apenas numpy e matplotlib para visualização.
- Download e parse dos arquivos IDX do MNIST (formato original), sem bibliotecas externas de datasets.
- SVD em dados centralizados para obter componentes e valores singulares.
- Experimentos com vários valores de k e avaliação por MSE de reconstrução.
- Visualização:
  - Variância explicada cumulativa (full PCA)
  - Original x Reconstrução para cada k
  - Projeção 2D dos dados com as duas primeiras componentes

V. RESULTADOS

Os resultados são gerados pelo script `src/pca_mnist.py` e incluem figuras e CSV com MSE por k. Para reproduzir: siga as instruções do README.

Sumário de resultados (exemplo):
- MSE tipicamente diminui conforme k aumenta
- As primeiras 10–20 componentes retêm grande parte da energia/variância

VI. CONCLUSÃO

O PCA é uma técnica eficaz para reduzir dimensionalidade em imagens, preservando características relevantes para compressão e visualização. Em MNIST, poucas dezenas de componentes são suficientes para reconstruções interpretáveis, e as projeções 2D podem revelar padrões de separação entre classes.

VII. REFERÊNCIAS

- C.M. Bishop, Pattern Recognition and Machine Learning. Springer, 2006.
- L. van der Maaten, E. Postma, and J. van den Herik, Dimensionality reduction: A comparative, J. Mach. Learn. Res., 2009.

---

ANEXOS

Tabela 1 - Resumo dos Artigos Pesquisados (original do trabalho) — Tabulação resumida.

(Adicionar a tabela fornecida no enunciado, conforme solicitado.)

