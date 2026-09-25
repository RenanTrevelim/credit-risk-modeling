from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split


# =========================================================
# CONFIGURAÇÕES
# =========================================================

TARGET = "inadimplente_2_anos"
RANDOM_STATE = 42

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = BASE_DIR / "data" / "dados_modelo.csv"
MODEL_PATH = BASE_DIR / "models" / "pipeline_modelo.pkl"


# =========================================================
# CARREGAR DADOS
# =========================================================

def carregar_dados():

    dados = pd.read_csv(DATA_PATH)

    X = dados.drop(columns=[TARGET])
    y = dados[TARGET]

    return X, y


# =========================================================
# CARREGAR PIPELINE
# =========================================================

def carregar_pipeline():

    pipeline = joblib.load(MODEL_PATH)

    return pipeline


# =========================================================
# TREINAMENTO
# =========================================================

def treinar():

    print("Carregando dados...")

    X, y = carregar_dados()

    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE
    )

    print(f"Treino: {X_treino.shape}")
    print(f"Teste: {X_teste.shape}")

    print("\nCarregando pipeline...")

    pipeline = carregar_pipeline()

    print("\nTreinando modelo...")

    pipeline.fit(
        X_treino,
        y_treino
    )

    joblib.dump(
        pipeline,
        MODEL_PATH
    )

    print("\nPipeline treinado e salvo com sucesso!")
    print(f"Modelo salvo em: {MODEL_PATH}")


if __name__ == "__main__":
    treinar()