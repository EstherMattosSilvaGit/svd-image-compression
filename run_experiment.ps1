# PowerShell script para rodar experimento PCA-MNIST
# Uso: .\run_experiment.ps1

python -m venv .venv; .\.venv\Scripts\Activate
pip install -r requirements.txt
python src\pca_mnist.py --max-samples 2000 --k 10 20 50
