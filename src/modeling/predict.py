from pathlib import Path

import joblib
import pandas as pd


# =========================================================
# CONFIGURAÇÕES
# =========================================================

TARGET = "inadimplente_2_anos"
THRESHOLD = 0.40

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = BASE_DIR / "data" / "dados_modelo.csv"
MODEL_PATH = BASE_DIR / "models" / "pipeline_modelo.pkl"


# =========================================================
# CARREGAR MODELO
# =========================================================

def carregar_modelo():

    pipeline = joblib.load(MODEL_PATH)

    return pipeline


# =========================================================
# CARREGAR DADOS
# =========================================================

def carregar_dados():

    dados = pd.read_csv(DATA_PATH)

    X = dados.drop(columns=[TARGET])

    return X


# =========================================================
# PREVISÃO
# =========================================================

def prever_risco(dados):

    print("Carregando modelo...")

    pipeline = carregar_modelo()

    print("Gerando probabilidades...")

    probabilidades = pipeline.predict_proba(dados)[:, 1]

    predicoes = (
        probabilidades >= THRESHOLD
    ).astype(int)

    resultado = dados.copy()

    resultado["probabilidade_inadimplencia"] = probabilidades
    resultado["previsao_inadimplencia"] = predicoes

    return resultado


# =========================================================
# EXECUÇÃO
# =========================================================

if __name__ == "__main__":

    print("Carregando dados...")

    X = carregar_dados()

    # Apenas 5 clientes para teste
    amostra = X.head(5)

    resultado = prever_risco(amostra)

    print("\nResultado das previsões:\n")

    print(
        resultado[
            [
                "probabilidade_inadimplencia",
                "previsao_inadimplencia"
            ]
        ]
    )