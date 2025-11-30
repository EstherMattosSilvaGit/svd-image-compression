# Técnicas de Redução de Dimensionalidade — PCA aplicado ao MNIST

Autores:
- Esther Mattos - esthermattos@id.uff.br
- Talisson Souza - talisedu@gmail.com

Professor:
- Dr. Marcos Lage - mlage@ic.uff.br

Resumo:
Este projeto demonstra o uso do Principal Component Analysis (PCA) como técnica de redução de dimensionalidade aplicada a imagens do dataset MNIST. Implementamos PCA manualmente usando apenas numpy e matplotlib para visualização, sem bibliotecas de alto nível (ex: sklearn), para que o professor possa inspecionar os passos de Álgebra Linear.

Palavras-chave: Visão Computacional, PCA, Imagens, Redução de Dimensionalidade

## improved_lda/pca MNIST — Guia rápido

Este repositório contém scripts educacionais para comparar implementações manuais e de biblioteca de PCA e LDA sobre um subconjunto do MNIST (OpenML).  
Arquivos principais (exemplos):
- `improved_lda_mnist_modes.py` — LDA com modos: `manual_eig`, `manual_power`, `sklearn`.
- `improved_pca_mnist_modes.py` — PCA com modos: `manual_eigh`, `manual_power`, `sklearn`.

O objetivo é permitir comparar:
- implementação totalmente manual (método das potências + deflação),
- decomposição com funções numéricas (ex.: `np.linalg.eigh` / `np.linalg.eig`),
- implementação de alto nível da biblioteca (`sklearn`).

Saída:
- Relatórios `.txt` em `outputs/`
- Imagens das componentes em `outputs/` (PNG)
- Log com todo o stdout salvo no relatório

---

## Requisitos (ambiente)

Recomendo usar Python 3.8+ (funciona em Python 3.10/3.11/3.12). Pacotes necessários:

- numpy
- pandas
- scikit-learn
- matplotlib

Você pode instalar tudo com pip.

---

## Instalação (passo a passo)

1. Crie (opcional) e ative um ambiente virtual:

Linux / macOS:
- python3 -m venv .venv
- source .venv/bin/activate

Windows (PowerShell):
- python -m venv .venv
- .\.venv\Scripts\Activate.ps1

2. Atualize pip e instale dependências:

pip install --upgrade pip
pip install numpy pandas scikit-learn matplotlib

(Alternativamente, crie um `requirements.txt` e rode `pip install -r requirements.txt`)

Exemplo de `requirements.txt`:
```
numpy
pandas
scikit-learn
matplotlib
```

---

## Como rodar (comandos)

Os scripts baixam MNIST via OpenML caso não esteja em cache. Por padrão usam um subsample para acelerar (`MAX_SAMPLES = 5000`).

Executar LDA (modo default presente no arquivo):

python improved_lda_mnist_modes.py

Executar PCA (modo default presente no arquivo):

python improved_pca_mnist_modes.py

Observação: ambos os scripts definem variáveis no topo (`LDA_METHOD`, `PCA_METHOD`, `MAX_SAMPLES`, `RANDOM_STATE`). Para trocar o modo sem editar o arquivo, você pode importar e chamar a função principal com parâmetros:

Exemplo rodando LDA com modo `manual_eig` (sem editar arquivo — executa diretamente no interpretador):
python -c "from improved_lda_mnist_modes import run_lda; run_lda(max_samples=2000, lda_method='manual_eig')"

Exemplo rodando PCA com modo `manual_power`:
python -c "from improved_pca_mnist_modes import main as run_pca; run_pca(max_samples=2000, pca_method='manual_power')"

(Observação: alguns scripts usam nome `main` ou `run_lda` — verifique a função principal do script. Se preferir, edite `LDA_METHOD` / `PCA_METHOD` no topo do arquivo.)

---

## Modos disponíveis

LDA (`LDA_METHOD`):
- `manual_eig` : resolve o problema generalizado via `np.linalg.eig` (didático/prático);
- `manual_power`: usa power iteration + deflação (completamente manual; pode ser lento e instável numericamente);
- `sklearn` : usa `sklearn.discriminant_analysis.LinearDiscriminantAnalysis`.

PCA (`PCA_METHOD`):
- `manual_eigh` : calcula autovalores/autovetores com `np.linalg.eigh`;
- `manual_power`: power iteration + deflação para autovalores/autovetores (didático; lento);
- `sklearn` : usa `sklearn.decomposition.PCA`.

---

## Saída / Onde encontrar resultados

Após execução:
- `outputs/lda_report.txt` ou `outputs/pca_report.txt` — relatório com métricas e log completo.
- `outputs/lda_manual_comp_*.png` e `outputs/lda_sklearn_comp_*.png` — imagens das componentes LDA.
- Análogos para PCA (se gerados).

Se o `outputs/` não existir, os scripts criam automaticamente.

---

## Dicas e observações importantes

- Modo `manual_power` pode ser bem mais lento — reduza `MAX_SAMPLES` (ex.: 1000) quando for testar ou rodar em máquina com recursos limitados.
- Os cálculos “manuais” (power iteration + deflação) são didáticos; para uso prático prefira `np.linalg.eig`/`np.linalg.eigh` ou `sklearn` (mais estáveis e otimizados).
- Se o download via OpenML falhar (problema de internet), baixe manualmente MNIST e adapte o script para carregar localmente.
- Os scripts capturam o stdout em memória para incluir no relatório; em execuções longas isso pode consumir muita memória. Se for problema, comente a parte da captura (`Tee` / `_stdout_buf`) ou remova a escrita do log completo.
- Para reproduzibilidade, use `RANDOM_STATE` definido no topo dos scripts.

---

## Problemas comuns

- Erro de memória: reduza `MAX_SAMPLES`.
- Timeout/erro ao baixar MNIST: verifique conexão ou tente novamente / usar cache.
- Avisos do sklearn sobre parâmetros deprecados: os scripts removem o uso explícito de `multi_class` para evitar warnings; mantenha scikit-learn atualizado mas estável (ex.: 1.2+).

