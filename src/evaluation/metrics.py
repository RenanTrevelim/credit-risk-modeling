from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report
)


# =========================================================
# CONFIGURAÇÕES
# =========================================================

TARGET = "inadimplente_2_anos"
THRESHOLD = 0.40
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

    _, X_teste, _, y_teste = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE
    )

    return X_teste, y_teste


# =========================================================
# CARREGAR MODELO
# =========================================================

def carregar_modelo():

    pipeline = joblib.load(MODEL_PATH)

    return pipeline


# =========================================================
# CALCULAR MÉTRICAS
# =========================================================

def calcular_metricas(y_real, predicoes, probabilidades):

    metricas = {
        "accuracy": accuracy_score(
            y_real,
            predicoes
        ),

        "balanced_accuracy": balanced_accuracy_score(
            y_real,
            predicoes
        ),

        "precision_1": precision_score(
            y_real,
            predicoes,
            pos_label=1
        ),

        "recall_1": recall_score(
            y_real,
            predicoes,
            pos_label=1
        ),

        "f1_1": f1_score(
            y_real,
            predicoes,
            pos_label=1
        ),

        "roc_auc": roc_auc_score(
            y_real,
            probabilidades
        ),

        "pr_auc": average_precision_score(
            y_real,
            probabilidades
        )
    }

    return metricas


# =========================================================
# AVALIAÇÃO
# =========================================================

def avaliar():

    print("Carregando dados...")

    X_teste, y_teste = carregar_dados()

    print("Carregando modelo...")

    pipeline = carregar_modelo()

    print("Gerando previsões...")

    probabilidades = pipeline.predict_proba(
        X_teste
    )[:, 1]

    predicoes = (
        probabilidades >= THRESHOLD
    ).astype(int)

    metricas = calcular_metricas(
        y_teste,
        predicoes,
        probabilidades
    )

    print("\n==============================")
    print("MÉTRICAS DO MODELO FINAL")
    print("==============================\n")

    for nome, valor in metricas.items():

        print(
            f"{nome}: {valor:.4f}"
        )

    print("\nClassification Report:\n")

    print(
        classification_report(
            y_teste,
            predicoes,
            digits=3
        )
    )


# =========================================================
# EXECUÇÃO
# =========================================================

if __name__ == "__main__":

    avaliar()