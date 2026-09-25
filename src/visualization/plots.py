from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay
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
# MATRIZ DE CONFUSÃO
# =========================================================

def plot_matriz_confusao(y_real, predicoes):

    ConfusionMatrixDisplay.from_predictions(
        y_real,
        predicoes,
        display_labels=[
            "Não Inadimplente",
            "Inadimplente"
        ]
    )

    plt.title(
        f"Matriz de Confusão - Threshold {THRESHOLD}"
    )

    plt.tight_layout()
    plt.show()


# =========================================================
# CURVA ROC
# =========================================================

def plot_curva_roc(y_real, probabilidades):

    RocCurveDisplay.from_predictions(
        y_real,
        probabilidades
    )

    plt.title("Curva ROC - Modelo Final")

    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# =========================================================
# CURVA PRECISION-RECALL
# =========================================================

def plot_precision_recall(y_real, probabilidades):

    PrecisionRecallDisplay.from_predictions(
        y_real,
        probabilidades
    )

    plt.title("Curva Precision-Recall - Modelo Final")

    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# =========================================================
# EXECUÇÃO
# =========================================================

def gerar_graficos():

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

    print("Gerando gráficos...\n")

    plot_matriz_confusao(
        y_teste,
        predicoes
    )

    plot_curva_roc(
        y_teste,
        probabilidades
    )

    plot_precision_recall(
        y_teste,
        probabilidades
    )


if __name__ == "__main__":

    gerar_graficos()