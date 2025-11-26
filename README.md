# Técnicas de Redução de Dimensionalidade — PCA aplicado ao MNIST

Autores:
- Esther Mattos - esthermattos@id.uff.br
- Talisson Souza - talisedu@gmail.com

Professor:
- Dr. Marcos Lage - mlage@ic.uff.br

Resumo:
Este projeto demonstra o uso do Principal Component Analysis (PCA) como técnica de redução de dimensionalidade aplicada a imagens do dataset MNIST. Implementamos PCA manualmente usando apenas numpy e matplotlib para visualização, sem bibliotecas de alto nível (ex: sklearn), para que o professor possa inspecionar os passos de Álgebra Linear.

Palavras-chave: Visão Computacional, PCA, Imagens, Redução de Dimensionalidade

Conteúdo do repositório:
- `src/pca_mnist.py`: Script principal que baixa e carrega MNIST, executa PCA via SVD, projeta e reconstrói imagens, gera métricas (MSE) e visualizações.
- `requirements.txt`: Dependências mínimas (numpy, matplotlib).

Como usar:
1. Clone o repositório.
2. Instale dependências (recomendado em um ambiente virtual):

```powershell
python -m venv .venv; .\.venv\Scripts\activate; pip install -r requirements.txt
```

3. Execute o script:

```powershell
python src\pca_mnist.py --help
```

O script baixará automaticamente o MNIST (se necessário), calculará componentes principais, fará compressão e reconstrução para uma lista de valores k (número de componentes) e gerará gráficos:
- Gráfico da variância explicada acumulada (para todas as componentes)
- Mosaico de imagens originais e reconstruídas para comparar (k em [10, 20, 50, 100])
- Projeção 2D das imagens (primeiros 2 componentes) com cores por classe

Modo de teste rápido (sem download):
```powershell
# Executa um demo sintético para testar implementação (NumPy e Pure)
python src\pca_mnist.py --quicktest --max-samples 200 --k 5 10 20
```

Metodologia (resumo):
- Carrega MNIST dos arquivos IDX oficiais
- Converte imagens para vetores (784 dimensões)
- Centraliza dados pela média
- Calcula SVD em X_centered (numpy.linalg.svd)
- Seleciona componentes principais para projeção e reconstrução
- Avalia reconstrução por MSE e visualização

Matemática (resumo):
- Dados: X ∈ R^{N×D}. Seja μ = (1/N) ∑_i x_i a média por coluna.
- Dados centralizados: X_c = X − μ.
- SVD em X_c: X_c = U Σ V^T. As linhas de V^T (ou colunas de V) representam os autovetores (componentes principais).
- Projeção k componentes: scores = X_c V_k^T (onde V_k^T ∈ R^{k×D}). Reconstrução: X_approx = scores V_k + μ.
- Variância explicada por cada componente: λ_j = Σ_j^2 / (N − 1).

Outputs:
- `outputs/explained_variance_cumulative_full.png` — variância explicada cumulativa por todas as componentes.
- `outputs/explained_variance_k_{k}.png` — variância explicada até o k usado.
- `outputs/plots_k_{k}.png` — mosaico com originais e reconstruções para cada k.
- `outputs/proj_2d_k_{k}.png` — projeção 2D.
- `outputs/mse_results.csv` — tabela com MSE por k.

Observações:
- O foco é didático: aqui implementamos PCA via SVD apenas com numpy para mostrar cada passo.
- Se você quiser usar somente uma parte do dataset para testar (por exemplo 5000 imagens) use a flag `--max-samples`.
- Se desejar ver a implementação didática sem NumPy (muito lenta, apenas para fins educacionais), utilize a flag `--pure-python`. Recomenda-se `--max-samples 200` ou menos nesse modo.

Licença: ATENÇÃO — Trabalho acadêmico; sinta-se à vontade para adaptar e citar.

Referências:
- C.M. Bishop, Pattern Recognition and Machine Learning. Springer, 2006.
- L. van der Maaten, E. Postma, and J. van den Herik, Dimensionality reduction: A comparative, J. Mach. Learn. Res., 2009.
