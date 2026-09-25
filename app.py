from pathlib import Path
from textwrap import dedent

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from plotly.subplots import make_subplots

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)


# ==================================================
# CONFIGURAÇÃO DA PÁGINA
# ==================================================

st.set_page_config(
    page_title="Credit Risk Intelligence",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==================================================
# CAMINHOS
# ==================================================

ROOT = Path(__file__).resolve().parent

CAMINHO_DADOS = (
    ROOT
    / "data"
    / "dados_modelo.csv"
)

CAMINHO_MODELO = (
    ROOT
    / "models"
    / "pipeline_modelo.pkl"
)


# ==================================================
# CONFIGURAÇÕES DO PROJETO
# ==================================================

TARGET = "inadimplente_2_anos"

THRESHOLD = 0.40

RANDOM_STATE = 42

TEST_SIZE = 0.20


COLUNAS_MODELO = [
    "utilizacao_credito",
    "idade",
    "atrasos_30_59_dias",
    "razao_divida",
    "renda_mensal",
    "linhas_credito_abertas",
    "atrasos_90_dias",
    "emprestimos_imobiliarios",
    "atrasos_60_89_dias",
    "numero_dependentes",
    "total_atrasos",
    "atraso_grave",
]


# ==================================================
# PREMISSAS FINANCEIRAS
# ==================================================

PERDA_MEDIA = 5000

TAXA_RECUPERACAO = 0.30

CUSTO_ACAO = 40


# ==================================================
# HTML
# ==================================================

def renderizar_html(
    conteudo: str
) -> None:

    st.html(
        dedent(conteudo).strip()
    )


# ==================================================
# ESTILO VISUAL
# ==================================================

renderizar_html(
    """
    <style>

        .stApp,
        [data-testid="stAppViewContainer"] {
            background-color: #F4F7FB;
            color: #0F172A;
        }

        .block-container {
            max-width: 1480px;
            padding-top: 1.7rem;
            padding-bottom: 3rem;
        }


        /* SIDEBAR */

        [data-testid="stSidebar"] {

            background: linear-gradient(
                180deg,
                #0F172A 0%,
                #1E3A5F 100%
            );

        }

        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] label {

            color: #FFFFFF !important;

        }


        /* FORMULÁRIOS */

        [data-testid="stNumberInput"] label,
        [data-testid="stCheckbox"] label,
        [data-testid="stForm"] label {

            color: #334155 !important;

            font-weight: 600;

        }


        /* HERO */

        .hero {

            padding: 2.5rem 2.7rem;

            border-radius: 24px;

            background: linear-gradient(
                135deg,
                #0F172A 0%,
                #1D4ED8 60%,
                #0EA5E9 100%
            );

            margin-bottom: 1.7rem;

            box-shadow:
                0 18px 45px
                rgba(37, 99, 235, 0.16);

        }

        .hero h1 {

            margin: 0;

            color: #FFFFFF !important;

            font-size: 2.7rem;

            line-height: 1.15;

        }

        .hero p {

            max-width: 1050px;

            margin-top: 1rem;

            margin-bottom: 0;

            color: #DBEAFE !important;

            font-size: 1.05rem;

            line-height: 1.7;

        }

        .badge {

            display: inline-block;

            padding: 0.4rem 0.9rem;

            margin-bottom: 1rem;

            border-radius: 999px;

            background-color:
                rgba(255, 255, 255, 0.18);

            color: #FFFFFF !important;

            font-size: 0.8rem;

            font-weight: 800;

        }


        /* TÍTULOS */

        .section-title {

            margin-top: 1.9rem;

            margin-bottom: 0.4rem;

            color: #0F172A !important;

            font-size: 1.5rem;

            font-weight: 800;

        }

        .section-subtitle {

            margin-bottom: 1.2rem;

            color: #64748B !important;

            font-size: 0.92rem;

            line-height: 1.6;

        }


        /* INFO CARD */

        .info-card {

            min-height: 185px;

            padding: 1.55rem;

            border: 1px solid #DCE4F0;

            border-radius: 18px;

            background: #FFFFFF;

            box-shadow:
                0 8px 25px
                rgba(15, 23, 42, 0.05);

        }

        .info-card h3 {

            margin-top: 0;

            color: #1D4ED8 !important;

        }

        .info-card p {

            color: #475569 !important;

            line-height: 1.65;

        }


        /* KPI */

        .kpi-card {

            min-height: 140px;

            padding: 1.3rem 1.35rem;

            border: 1px solid #E2E8F0;

            border-radius: 18px;

            background: linear-gradient(
                145deg,
                #FFFFFF,
                #F8FAFC
            );

            box-shadow:
                0 8px 22px
                rgba(15, 23, 42, 0.05);

        }

        .kpi-label {

            color: #64748B !important;

            font-size: 0.75rem;

            font-weight: 800;

            letter-spacing: 0.04rem;

            text-transform: uppercase;

        }

        .kpi-value {

            margin-top: 0.45rem;

            color: #0F172A !important;

            font-size: 1.85rem;

            font-weight: 800;

        }

        .kpi-detail {

            margin-top: 0.35rem;

            color: #64748B !important;

            font-size: 0.8rem;

        }


        /* EXECUTIVE CARD */

        .executive-card {

            padding: 1.6rem 1.8rem;

            margin-top: 1.2rem;

            border-radius: 20px;

            background: linear-gradient(
                135deg,
                #EFF6FF,
                #FFFFFF
            );

            border: 1px solid #BFDBFE;

            box-shadow:
                0 8px 25px
                rgba(37, 99, 235, 0.06);

        }

        .executive-card h3 {

            margin-top: 0;

            color: #1D4ED8 !important;

        }

        .executive-card p {

            margin-bottom: 0.45rem;

            color: #475569 !important;

            line-height: 1.7;

            font-size: 0.98rem;

        }


        /* BUSINESS CARD */

        .business-card {

            padding: 1.6rem 1.8rem;

            border: 1px solid #BBF7D0;

            border-radius: 18px;

            background: linear-gradient(
                145deg,
                #F0FDF4,
                #FFFFFF
            );

            box-shadow:
                0 8px 24px
                rgba(22, 163, 74, 0.05);

        }

        .business-card h3 {

            margin-top: 0;

            color: #15803D !important;

        }

        .business-card h2 {

            margin-bottom: 0.4rem;

            color: #166534 !important;

            font-size: 2rem;

        }

        .business-card p {

            color: #475569 !important;

            line-height: 1.7;

        }


        /* NOTA */

        .model-note {

            padding: 1.2rem 1.4rem;

            border-left: 5px solid #2563EB;

            border-radius: 14px;

            background-color: #EFF6FF;

            color: #1E3A8A !important;

            line-height: 1.7;

        }


        /* DATAFRAME */

        [data-testid="stDataFrame"] {

            overflow: hidden;

            border: 1px solid #DCE4F0;

            border-radius: 16px;

            box-shadow:
                0 8px 22px
                rgba(15, 23, 42, 0.05);

        }


        /* DOWNLOAD */

        [data-testid="stDownloadButton"] button {

            width: 100%;

            border: none;

            border-radius: 11px;

            background: linear-gradient(
                90deg,
                #1D4ED8,
                #0EA5E9
            );

            color: #FFFFFF;

            font-weight: 700;

        }


        /* FOOTER */

        .footer {

            margin-top: 3rem;

            padding-top: 1.5rem;

            border-top:
                1px solid #DCE4F0;

            color: #64748B !important;

            font-size: 0.88rem;

            text-align: center;

        }

    </style>
    """
)


# ==================================================
# FUNÇÕES AUXILIARES
# ==================================================

def formatar_milhoes(
    valor: float
) -> str:

    return (
        f"R$ {valor / 1_000_000:.2f} Mi"
        .replace(".", ",")
    )


def formatar_inteiro(
    valor: int
) -> str:

    return (
        f"{valor:,}"
        .replace(",", ".")
    )


def converter_csv(
    dados: pd.DataFrame
) -> bytes:

    return dados.to_csv(
        index=False,
        encoding="utf-8-sig"
    ).encode(
        "utf-8-sig"
    )


def classificar_risco(
    score: float
) -> str:

    if score < 0.20:
        return "Muito Baixo"

    if score < 0.40:
        return "Baixo"

    if score < 0.60:
        return "Moderado"

    if score < 0.80:
        return "Alto"

    return "Muito Alto"


def cor_risco(
    score: float
) -> str:

    if score < 0.20:
        return "#16A34A"

    if score < 0.40:
        return "#65A30D"

    if score < 0.60:
        return "#D97706"

    if score < 0.80:
        return "#EA580C"

    return "#DC2626"


# ==================================================
# CARREGAR MODELO
# ==================================================

@st.cache_resource(
    show_spinner=False
)
def carregar_modelo():

    if not CAMINHO_MODELO.exists():

        raise FileNotFoundError(
            f"Modelo não encontrado: "
            f"{CAMINHO_MODELO}"
        )

    return joblib.load(
        CAMINHO_MODELO
    )


# ==================================================
# CARREGAR DADOS DE TESTE
# ==================================================

@st.cache_data(
    show_spinner=False
)
def carregar_dados_teste():

    if not CAMINHO_DADOS.exists():

        raise FileNotFoundError(
            f"Base não encontrada: "
            f"{CAMINHO_DADOS}"
        )


    dados = pd.read_csv(
        CAMINHO_DADOS
    )


    colunas_indice = [
        coluna
        for coluna in dados.columns
        if coluna.startswith("Unnamed")
    ]


    if colunas_indice:

        dados = dados.drop(
            columns=colunas_indice
        )


    X = dados.drop(
        columns=[TARGET]
    )


    y = dados[
        TARGET
    ]


    _, X_teste, _, y_teste = (
        train_test_split(

            X,
            y,

            test_size=TEST_SIZE,

            stratify=y,

            random_state=RANDOM_STATE
        )
    )


    return (
        X_teste,
        y_teste
    )


# ==================================================
# EXECUTAR MODELO NA CARTEIRA
# ==================================================

@st.cache_data(
    show_spinner=False
)
def executar_avaliacao_carteira():


    X_teste, y_teste = (
        carregar_dados_teste()
    )


    modelo = (
        carregar_modelo()
    )


    probabilidades = (
        modelo.predict_proba(
            X_teste
        )[:, 1]
    )


    predicoes = (
        probabilidades
        >= THRESHOLD
    ).astype(int)


    resultado = (
        X_teste
        .reset_index()
        .rename(
            columns={
                "index":
                    "indice_cliente"
            }
        )
    )


    resultado["real"] = (
        y_teste
        .reset_index(
            drop=True
        )
    )


    resultado[
        "probabilidade_inadimplencia"
    ] = probabilidades


    resultado[
        "predicao"
    ] = predicoes


    resultado[
        "faixa_risco"
    ] = pd.cut(

        probabilidades,

        bins=[
            0,
            0.20,
            0.40,
            0.60,
            0.80,
            1.00
        ],

        labels=[
            "Muito Baixo",
            "Baixo",
            "Moderado",
            "Alto",
            "Muito Alto"
        ],

        include_lowest=True
    )


    return resultado


# ==================================================
# MÉTRICAS
# ==================================================

def calcular_metricas_carteira(
    resultado: pd.DataFrame
) -> dict:


    y_real = (
        resultado["real"]
    )


    y_pred = (
        resultado["predicao"]
    )


    probabilidades = (
        resultado[
            "probabilidade_inadimplencia"
        ]
    )


    total_clientes = (
        len(resultado)
    )


    sinalizados = (
        resultado["predicao"]
        .sum()
    )


    inadimplentes = (
        resultado["real"]
        .sum()
    )


    detectados = (
        (
            (resultado["real"] == 1)
            &
            (resultado["predicao"] == 1)
        )
        .sum()
    )


    return {

        "total_clientes":
            total_clientes,

        "sinalizados":
            sinalizados,

        "percentual_sinalizado":
            sinalizados
            / total_clientes,

        "inadimplentes":
            inadimplentes,

        "detectados":
            detectados,

        "recall":
            recall_score(
                y_real,
                y_pred
            ),

        "precision":
            precision_score(
                y_real,
                y_pred,
                zero_division=0
            ),

        "f1":
            f1_score(
                y_real,
                y_pred
            ),

        "roc_auc":
            roc_auc_score(
                y_real,
                probabilidades
            ),

        "pr_auc":
            average_precision_score(
                y_real,
                probabilidades
            )
    }


# ==================================================
# IMPACTO FINANCEIRO
# ==================================================

def calcular_impacto_financeiro(
    resultado: pd.DataFrame
) -> dict:


    total = (
        len(resultado)
    )


    sinalizados = (
        resultado["predicao"]
        .sum()
    )


    inadimplentes = (
        resultado["real"]
        .sum()
    )


    detectados = (
        (
            (resultado["real"] == 1)
            &
            (resultado["predicao"] == 1)
        )
        .sum()
    )


    valor_recuperado = (
        detectados
        * PERDA_MEDIA
        * TAXA_RECUPERACAO
    )


    custo_total = (
        sinalizados
        * CUSTO_ACAO
    )


    beneficio_modelo = (
        valor_recuperado
        - custo_total
    )


    percentual_priorizado = (
        sinalizados
        / total
    )


    inadimplentes_aleatorio = (
        inadimplentes
        * percentual_priorizado
    )


    beneficio_aleatorio = (

        inadimplentes_aleatorio
        * PERDA_MEDIA
        * TAXA_RECUPERACAO

        -

        custo_total
    )


    ganho_incremental = (
        beneficio_modelo
        - beneficio_aleatorio
    )


    return {

        "modelo":
            beneficio_modelo,

        "aleatorio":
            beneficio_aleatorio,

        "incremental":
            ganho_incremental,

        "custo_total":
            custo_total,

        "valor_recuperado":
            valor_recuperado
    }


# ==================================================
# CAPTURA POR FAIXA DA CARTEIRA
# ==================================================

def calcular_captura_top(
    resultado,
    percentual
):


    ranking = (
        resultado
        .sort_values(
            "probabilidade_inadimplencia",
            ascending=False
        )
    )


    quantidade = int(
        len(ranking)
        * percentual
    )


    capturados = (
        ranking
        .iloc[:quantidade]["real"]
        .sum()
    )


    total_inadimplentes = (
        ranking["real"]
        .sum()
    )


    return (
        capturados
        / total_inadimplentes
    )


# ==================================================
# GRÁFICO - EFICIÊNCIA DA PRIORIZAÇÃO
# ==================================================

def grafico_eficiencia_priorizacao(
    metricas
):


    dados = pd.DataFrame(
        {

            "Indicador": [
                "Carteira priorizada",
                "Inadimplentes capturados"
            ],

            "Percentual": [
                metricas[
                    "percentual_sinalizado"
                ] * 100,

                metricas[
                    "recall"
                ] * 100
            ]
        }
    )


    fig = px.bar(

        dados,

        x="Indicador",

        y="Percentual",

        text="Percentual",

        title=(
            "Eficiência da Priorização"
        ),

        labels={
            "Percentual":
                "Percentual (%)"
        }
    )


    fig.update_traces(

        texttemplate="%{y:.1f}%",

        textposition="outside"
    )


    fig.update_layout(

        height=410,

        showlegend=False,

        yaxis_range=[
            0,
            100
        ],

        margin=dict(
            l=20,
            r=20,
            t=65,
            b=20
        )
    )


    return fig


# ==================================================
# GRÁFICO - FUNIL OPERACIONAL
# ==================================================

def grafico_funil(
    metricas
):


    fig = go.Figure(

        go.Funnel(

            y=[
                "Clientes avaliados",
                "Clientes priorizados",
                "Inadimplentes identificados"
            ],

            x=[
                metricas[
                    "total_clientes"
                ],

                metricas[
                    "sinalizados"
                ],

                metricas[
                    "detectados"
                ]
            ],

            textinfo=(
                "value+percent initial"
            )
        )
    )


    fig.update_layout(

        title=(
            "Funil de Priorização da Carteira"
        ),

        height=410,

        margin=dict(
            l=20,
            r=20,
            t=65,
            b=20
        )
    )


    return fig


# ==================================================
# GRÁFICO - FAIXAS DE RISCO
# ==================================================

def grafico_faixas_risco(
    resultado
):


    resumo = (
        resultado
        .groupby(
            "faixa_risco",
            observed=False
        )
        .agg(

            clientes=(
                "real",
                "count"
            ),

            inadimplentes=(
                "real",
                "sum"
            )

        )
        .reset_index()
    )


    resumo[
        "taxa_inadimplencia"
    ] = (

        resumo[
            "inadimplentes"
        ]

        /

        resumo[
            "clientes"
        ]

        * 100
    )


    fig = make_subplots(
        specs=[
            [
                {
                    "secondary_y":
                        True
                }
            ]
        ]
    )


    fig.add_trace(

        go.Bar(

            x=resumo[
                "faixa_risco"
            ],

            y=resumo[
                "clientes"
            ],

            name=(
                "Clientes"
            ),

            text=resumo[
                "clientes"
            ],

            textposition="outside"
        ),

        secondary_y=False
    )


    fig.add_trace(

        go.Scatter(

            x=resumo[
                "faixa_risco"
            ],

            y=resumo[
                "taxa_inadimplencia"
            ],

            mode="lines+markers+text",

            name=(
                "Taxa de inadimplência"
            ),

            text=[
                f"{valor:.1f}%"
                for valor
                in resumo[
                    "taxa_inadimplencia"
                ]
            ],

            textposition="top center"
        ),

        secondary_y=True
    )


    fig.update_layout(

        title=(
            "Distribuição da Carteira por Nível de Risco"
        ),

        height=480,

        legend=dict(
            orientation="h",
            y=1.12
        ),

        margin=dict(
            l=20,
            r=20,
            t=90,
            b=20
        )
    )


    fig.update_yaxes(

        title_text=(
            "Quantidade de clientes"
        ),

        secondary_y=False
    )


    fig.update_yaxes(

        title_text=(
            "Taxa de inadimplência (%)"
        ),

        secondary_y=True
    )


    return fig


# ==================================================
# GRÁFICO - CAPTURA DA CARTEIRA
# ==================================================

def grafico_captura(
    resultado
):


    ranking = (
        resultado
        .sort_values(
            "probabilidade_inadimplencia",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )


    ranking[
        "carteira_pct"
    ] = (

        (
            ranking.index
            + 1
        )

        /

        len(ranking)

        * 100
    )


    ranking[
        "captura_pct"
    ] = (

        ranking[
            "real"
        ]
        .cumsum()

        /

        ranking[
            "real"
        ]
        .sum()

        * 100
    )


    fig = go.Figure()


    fig.add_trace(

        go.Scatter(

            x=ranking[
                "carteira_pct"
            ],

            y=ranking[
                "captura_pct"
            ],

            mode="lines",

            name=(
                "Modelo de risco"
            ),

            line=dict(
                width=4
            )
        )
    )


    fig.add_trace(

        go.Scatter(

            x=[
                0,
                100
            ],

            y=[
                0,
                100
            ],

            mode="lines",

            name=(
                "Seleção aleatória"
            ),

            line=dict(
                dash="dash"
            )
        )
    )


    fig.add_vline(

        x=THRESHOLD * 0,

        visible=False
    )


    fig.update_layout(

        title=(
            "Concentração dos Inadimplentes na Carteira"
        ),

        xaxis_title=(
            "% da carteira priorizada"
        ),

        yaxis_title=(
            "% dos inadimplentes capturados"
        ),

        height=480,

        xaxis_range=[
            0,
            100
        ],

        yaxis_range=[
            0,
            100
        ],

        legend=dict(
            orientation="h",
            y=1.12
        ),

        margin=dict(
            l=20,
            r=20,
            t=90,
            b=20
        )
    )


    return fig


# ==================================================
# GRÁFICO - IMPACTO FINANCEIRO
# ==================================================

def grafico_impacto_financeiro(
    impacto
):


    dados = pd.DataFrame(
        {

            "Estratégia": [
                "Seleção Aleatória",
                "Modelo de Risco"
            ],

            "Benefício": [
                impacto[
                    "aleatorio"
                ],

                impacto[
                    "modelo"
                ]
            ]
        }
    )


    dados["Texto"] = [

        formatar_milhoes(
            valor
        )

        for valor
        in dados[
            "Benefício"
        ]

    ]


    fig = px.bar(

        dados,

        x="Estratégia",

        y="Benefício",

        text="Texto",

        title=(
            "Benefício Financeiro Potencial"
        )
    )


    fig.update_traces(

        textposition="outside"
    )


    fig.update_layout(

        height=440,

        showlegend=False,

        yaxis_title=(
            "Benefício líquido estimado (R$)"
        ),

        margin=dict(
            l=20,
            r=20,
            t=65,
            b=20
        )
    )


    return fig


# ==================================================
# PREPARAR CLIENTE INDIVIDUAL
# ==================================================

def preparar_cliente(

    utilizacao_credito,
    idade,
    atrasos_30_59,
    razao_divida,
    renda_mensal,
    linhas_credito,
    atrasos_90,
    emprestimos_imobiliarios,
    atrasos_60_89,
    numero_dependentes
):


    total_atrasos = (

        atrasos_30_59
        + atrasos_60_89
        + atrasos_90
    )


    atraso_grave = int(
        atrasos_90 > 0
    )


    cliente = pd.DataFrame(
        {

            "utilizacao_credito": [
                utilizacao_credito
            ],

            "idade": [
                idade
            ],

            "atrasos_30_59_dias": [
                atrasos_30_59
            ],

            "razao_divida": [
                razao_divida
            ],

            "renda_mensal": [
                renda_mensal
            ],

            "linhas_credito_abertas": [
                linhas_credito
            ],

            "atrasos_90_dias": [
                atrasos_90
            ],

            "emprestimos_imobiliarios": [
                emprestimos_imobiliarios
            ],

            "atrasos_60_89_dias": [
                atrasos_60_89
            ],

            "numero_dependentes": [
                numero_dependentes
            ],

            "total_atrasos": [
                total_atrasos
            ],

            "atraso_grave": [
                atraso_grave
            ]

        }
    )


    return cliente


# ==================================================
# PREVISÃO INDIVIDUAL
# ==================================================

def prever_cliente(
    modelo,
    cliente
):


    if hasattr(
        modelo,
        "feature_names_in_"
    ):

        cliente = cliente[
            list(
                modelo.feature_names_in_
            )
        ]

    else:

        cliente = cliente[
            COLUNAS_MODELO
        ]


    score = (
        modelo.predict_proba(
            cliente
        )[0, 1]
    )


    predicao = int(
        score
        >= THRESHOLD
    )


    return (
        score,
        predicao,
        cliente
    )


# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:


    st.markdown(
        "## 💳 Credit Risk"
    )


    st.caption(
        "Credit Risk Intelligence"
    )


    st.divider()


    pagina = st.radio(

        "Navegação",

        [
            "Visão geral",
            "Dashboard da carteira",
            "Avaliar cliente",
            "Sobre o projeto"
        ]
    )


    st.divider()


    st.markdown(
        "### Tecnologia"
    )


    st.markdown(
        """
**Machine Learning**

XGBoost + Optuna

**Explicabilidade**

SHAP

**Objetivo**

Detecção de Inadimplência
        """
    )


    st.divider()


    st.caption(
        f"Threshold operacional: "
        f"{THRESHOLD:.0%}"
    )


# ==================================================
# VISÃO GERAL
# ==================================================

if pagina == "Visão geral":


    resultado = (
        executar_avaliacao_carteira()
    )


    metricas = (
        calcular_metricas_carteira(
            resultado
        )
    )


    impacto = (
        calcular_impacto_financeiro(
            resultado
        )
    )


    renderizar_html(
        """
        <div class="hero">

            <span class="badge">
                Machine Learning • Credit Risk
            </span>

            <h1>
                Credit Risk Intelligence
            </h1>

            <p>
                Aplicação de Machine Learning para
                identificação e priorização de clientes
                com maior risco de inadimplência,
                transformando previsões em indicadores
                operacionais e valor potencial para
                o negócio.
            </p>

        </div>
        """
    )


    col1, col2, col3 = (
        st.columns(3)
    )


    with col1:

        renderizar_html(
            """
            <div class="info-card">

                <h3>
                    🎯 Identificação de risco
                </h3>

                <p>
                    O modelo estima o risco de
                    inadimplência de cada cliente
                    utilizando histórico de atrasos,
                    utilização de crédito e
                    características financeiras.
                </p>

            </div>
            """
        )


    with col2:

        renderizar_html(
            """
            <div class="info-card">

                <h3>
                    📊 Priorização da carteira
                </h3>

                <p>
                    Clientes com maior score são
                    priorizados para ações preventivas,
                    permitindo concentrar recursos
                    em uma parcela menor da carteira.
                </p>

            </div>
            """
        )


    with col3:

        renderizar_html(
            """
            <div class="info-card">

                <h3>
                    💰 Impacto de negócio
                </h3>

                <p>
                    O projeto simula como a utilização
                    do modelo poderia reduzir perdas
                    e aumentar a eficiência das ações
                    de prevenção à inadimplência.
                </p>

            </div>
            """
        )


    renderizar_html(
        """
        <div class="section-title">
            Principais resultados
        </div>
        """
    )


    k1, k2, k3, k4 = (
        st.columns(4)
    )


    with k1:

        renderizar_html(
            f"""
            <div class="kpi-card">

                <div class="kpi-label">
                    Clientes avaliados
                </div>

                <div class="kpi-value">
                    {formatar_inteiro(
                        metricas["total_clientes"]
                    )}
                </div>

                <div class="kpi-detail">
                    conjunto de teste
                </div>

            </div>
            """
        )


    with k2:

        renderizar_html(
            f"""
            <div class="kpi-card">

                <div class="kpi-label">
                    Inadimplentes detectados
                </div>

                <div class="kpi-value">
                    {metricas["recall"]:.1%}
                </div>

                <div class="kpi-detail">
                    recall da classe 1
                </div>

            </div>
            """
        )


    with k3:

        renderizar_html(
            f"""
            <div class="kpi-card">

                <div class="kpi-label">
                    Carteira priorizada
                </div>

                <div class="kpi-value">
                    {metricas["percentual_sinalizado"]:.1%}
                </div>

                <div class="kpi-detail">
                    clientes sinalizados
                </div>

            </div>
            """
        )


    with k4:

        renderizar_html(
            f"""
            <div class="kpi-card">

                <div class="kpi-label">
                    Ganho incremental
                </div>

                <div class="kpi-value">
                    {formatar_milhoes(
                        impacto["incremental"]
                    )}
                </div>

                <div class="kpi-detail">
                    simulação financeira
                </div>

            </div>
            """
        )


# ==================================================
# DASHBOARD EXECUTIVO
# ==================================================

elif pagina == "Dashboard da carteira":


    renderizar_html(
        """
        <div class="hero">

            <span class="badge">
                Executive Credit Risk Dashboard
            </span>

            <h1>
                Visão Executiva da Carteira
            </h1>

            <p>
                Painel gerencial para acompanhamento
                da carteira de crédito, concentração
                de clientes de risco, eficiência da
                priorização e impacto financeiro
                potencial da utilização do modelo.
            </p>

        </div>
        """
    )


    try:


        with st.spinner(
            "Analisando carteira..."
        ):


            resultado = (
                executar_avaliacao_carteira()
            )


            metricas = (
                calcular_metricas_carteira(
                    resultado
                )
            )


            impacto = (
                calcular_impacto_financeiro(
                    resultado
                )
            )


            captura_10 = (
                calcular_captura_top(
                    resultado,
                    0.10
                )
            )


            captura_20 = (
                calcular_captura_top(
                    resultado,
                    0.20
                )
            )


            captura_30 = (
                calcular_captura_top(
                    resultado,
                    0.30
                )
            )


        # ==================================================
        # KPIs EXECUTIVOS
        # ==================================================

        renderizar_html(
            """
            <div class="section-title">
                Resumo executivo
            </div>

            <div class="section-subtitle">
                Indicadores essenciais para acompanhamento
                do risco e priorização da carteira.
            </div>
            """
        )


        k1, k2, k3, k4 = (
            st.columns(4)
        )


        with k1:

            renderizar_html(
                f"""
                <div class="kpi-card">

                    <div class="kpi-label">
                        Carteira analisada
                    </div>

                    <div class="kpi-value">
                        {formatar_inteiro(
                            metricas["total_clientes"]
                        )}
                    </div>

                    <div class="kpi-detail">
                        clientes avaliados
                    </div>

                </div>
                """
            )


        with k2:

            renderizar_html(
                f"""
                <div class="kpi-card">

                    <div class="kpi-label">
                        Carteira priorizada
                    </div>

                    <div class="kpi-value">
                        {metricas["percentual_sinalizado"]:.1%}
                    </div>

                    <div class="kpi-detail">
                        {formatar_inteiro(
                            metricas["sinalizados"]
                        )}
                        clientes
                    </div>

                </div>
                """
            )


        with k3:

            renderizar_html(
                f"""
                <div class="kpi-card">

                    <div class="kpi-label">
                        Inadimplentes capturados
                    </div>

                    <div class="kpi-value">
                        {metricas["recall"]:.1%}
                    </div>

                    <div class="kpi-detail">
                        {formatar_inteiro(
                            metricas["detectados"]
                        )}
                        de
                        {formatar_inteiro(
                            metricas["inadimplentes"]
                        )}
                    </div>

                </div>
                """
            )


        with k4:

            renderizar_html(
                f"""
                <div class="kpi-card">

                    <div class="kpi-label">
                        Valor incremental
                    </div>

                    <div class="kpi-value">
                        {formatar_milhoes(
                            impacto["incremental"]
                        )}
                    </div>

                    <div class="kpi-detail">
                        vs. seleção aleatória
                    </div>

                </div>
                """
            )


        # ==================================================
        # LEITURA EXECUTIVA
        # ==================================================

        renderizar_html(
            f"""
            <div class="executive-card">

                <h3>
                    💡 Leitura Executiva
                </h3>

                <p>
                    O modelo permite concentrar as ações
                    preventivas em apenas
                    <strong>
                        {metricas["percentual_sinalizado"]:.1%}
                    </strong>
                    da carteira e, ainda assim, alcançar
                    aproximadamente
                    <strong>
                        {metricas["recall"]:.1%}
                    </strong>
                    dos clientes que efetivamente
                    apresentaram inadimplência.
                </p>

                <p>
                    Sob as premissas financeiras utilizadas
                    no projeto, essa priorização representaria
                    um benefício líquido potencial de
                    <strong>
                        {formatar_milhoes(
                            impacto["modelo"]
                        )}
                    </strong>
                    e um ganho incremental estimado de
                    <strong>
                        {formatar_milhoes(
                            impacto["incremental"]
                        )}
                    </strong>
                    em comparação com uma seleção sem
                    inteligência de risco.
                </p>

            </div>
            """
        )


        # ==================================================
        # PRIORIZAÇÃO
        # ==================================================

        renderizar_html(
            """
            <div class="section-title">
                Eficiência da estratégia de priorização
            </div>

            <div class="section-subtitle">
                Relação entre o volume da carteira que
                precisa ser abordado e a quantidade de
                inadimplentes alcançada.
            </div>
            """
        )


        col1, col2 = (
            st.columns(2)
        )


        with col1:

            st.plotly_chart(
                grafico_eficiencia_priorizacao(
                    metricas
                ),
                use_container_width=True
            )


        with col2:

            st.plotly_chart(
                grafico_funil(
                    metricas
                ),
                use_container_width=True
            )


        # ==================================================
        # SEGMENTAÇÃO
        # ==================================================

        renderizar_html(
            """
            <div class="section-title">
                Onde está concentrado o risco?
            </div>

            <div class="section-subtitle">
                Distribuição dos clientes pelas faixas
                de score e comportamento real da
                inadimplência dentro de cada grupo.
            </div>
            """
        )


        st.plotly_chart(
            grafico_faixas_risco(
                resultado
            ),
            use_container_width=True
        )


        # ==================================================
        # CAPACIDADE OPERACIONAL
        # ==================================================

        renderizar_html(
            """
            <div class="section-title">
                Quanto da carteira precisamos abordar?
            </div>

            <div class="section-subtitle">
                A curva abaixo permite simular diferentes
                capacidades operacionais de atuação.
            </div>
            """
        )


        c1, c2, c3 = (
            st.columns(3)
        )


        with c1:

            renderizar_html(
                f"""
                <div class="kpi-card">

                    <div class="kpi-label">
                        Top 10% da carteira
                    </div>

                    <div class="kpi-value">
                        {captura_10:.1%}
                    </div>

                    <div class="kpi-detail">
                        dos inadimplentes capturados
                    </div>

                </div>
                """
            )


        with c2:

            renderizar_html(
                f"""
                <div class="kpi-card">

                    <div class="kpi-label">
                        Top 20% da carteira
                    </div>

                    <div class="kpi-value">
                        {captura_20:.1%}
                    </div>

                    <div class="kpi-detail">
                        dos inadimplentes capturados
                    </div>

                </div>
                """
            )


        with c3:

            renderizar_html(
                f"""
                <div class="kpi-card">

                    <div class="kpi-label">
                        Top 30% da carteira
                    </div>

                    <div class="kpi-value">
                        {captura_30:.1%}
                    </div>

                    <div class="kpi-detail">
                        dos inadimplentes capturados
                    </div>

                </div>
                """
            )


        st.plotly_chart(
            grafico_captura(
                resultado
            ),
            use_container_width=True
        )


        # ==================================================
        # IMPACTO FINANCEIRO
        # ==================================================

        renderizar_html(
            """
            <div class="section-title">
                Impacto financeiro potencial
            </div>

            <div class="section-subtitle">
                Comparação entre uma atuação orientada
                pelo modelo e uma estratégia sem
                priorização de risco.
            </div>
            """
        )


        col_fin1, col_fin2 = (
            st.columns(
                [1.5, 1]
            )
        )


        with col_fin1:

            st.plotly_chart(
                grafico_impacto_financeiro(
                    impacto
                ),
                use_container_width=True
            )


        with col_fin2:

            renderizar_html(
                f"""
                <div class="business-card">

                    <h3>
                        💰 Impacto Estimado
                    </h3>

                    <p>
                        Benefício líquido utilizando
                        priorização por Machine Learning:
                    </p>

                    <h2>
                        {formatar_milhoes(
                            impacto["modelo"]
                        )}
                    </h2>

                    <p>
                        Benefício esperado utilizando
                        seleção aleatória:
                    </p>

                    <h2>
                        {formatar_milhoes(
                            impacto["aleatorio"]
                        )}
                    </h2>

                    <p>
                        <strong>
                            Ganho adicional atribuído à
                            priorização do modelo:
                            {formatar_milhoes(
                                impacto["incremental"]
                            )}
                        </strong>
                    </p>

                </div>
                """
            )


        renderizar_html(
            f"""
            <div class="model-note">

                <strong>
                    Premissas da simulação:
                </strong>

                perda média de R$ {PERDA_MEDIA:,.0f}
                por inadimplente,
                efetividade de
                {TAXA_RECUPERACAO:.0%}
                nas ações preventivas
                e custo de R$ {CUSTO_ACAO}
                por cliente abordado.

                Os valores são hipotéticos e possuem
                finalidade demonstrativa para o projeto.

            </div>
            """.replace(",", ".")
        )


        # ==================================================
        # DETALHES OPCIONAIS
        # ==================================================

        with st.expander(
            "📋 Ver clientes com maior risco"
        ):


            ranking = (
                resultado
                .sort_values(
                    "probabilidade_inadimplencia",
                    ascending=False
                )
                .copy()
            )


            ranking[
                "score_percentual"
            ] = (

                ranking[
                    "probabilidade_inadimplencia"
                ]

                * 100
            )


            st.dataframe(

                ranking[
                    [
                        "indice_cliente",
                        "idade",
                        "renda_mensal",
                        "total_atrasos",
                        "utilizacao_credito",
                        "score_percentual",
                        "faixa_risco",
                        "real"
                    ]
                ]
                .head(50),

                use_container_width=True,

                hide_index=True,

                column_config={

                    "indice_cliente":
                        "Cliente",

                    "idade":
                        "Idade",

                    "renda_mensal":
                        st.column_config.NumberColumn(
                            "Renda Mensal",
                            format="R$ %.2f"
                        ),

                    "total_atrasos":
                        "Atrasos",

                    "utilizacao_credito":
                        st.column_config.NumberColumn(
                            "Utilização Crédito",
                            format="%.2f"
                        ),

                    "score_percentual":
                        st.column_config.ProgressColumn(
                            "Score de Risco",
                            min_value=0,
                            max_value=100,
                            format="%.1f%%"
                        ),

                    "faixa_risco":
                        "Faixa",

                    "real":
                        "Inadimplente"
                }
            )


            st.download_button(

                "⬇️ Exportar carteira avaliada",

                data=converter_csv(
                    resultado
                ),

                file_name=(
                    "carteira_risco_avaliada.csv"
                ),

                mime="text/csv",

                use_container_width=True
            )


    except Exception as erro:


        st.error(
            "Não foi possível executar "
            f"a análise da carteira: {erro}"
        )


# ==================================================
# AVALIAÇÃO INDIVIDUAL
# ==================================================

elif pagina == "Avaliar cliente":


    renderizar_html(
        """
        <div class="hero">

            <span class="badge">
                Credit Risk Scoring
            </span>

            <h1>
                Avaliação Individual de Risco
            </h1>

            <p>
                Informe os dados do cliente para
                gerar um score estimado de risco
                e verificar se ele seria priorizado
                pelo modelo.
            </p>

        </div>
        """
    )


    try:


        modelo = (
            carregar_modelo()
        )


        with st.form(
            "form_cliente"
        ):


            renderizar_html(
                """
                <div class="section-title">
                    Dados do cliente
                </div>

                <div class="section-subtitle">
                    Preencha as informações utilizadas
                    pelo modelo para realizar a avaliação.
                </div>
                """
            )


            col1, col2 = (
                st.columns(2)
            )


            with col1:


                idade = st.number_input(
                    "Idade",
                    min_value=18,
                    max_value=120,
                    value=40,
                    step=1
                )


                utilizacao_credito = (
                    st.number_input(
                        "Utilização do crédito",
                        min_value=0.0,
                        value=0.30,
                        step=0.05,
                        format="%.2f"
                    )
                )


                razao_divida = (
                    st.number_input(
                        "Razão da dívida",
                        min_value=0.0,
                        value=0.30,
                        step=0.05,
                        format="%.2f"
                    )
                )


                renda_mensal = (
                    st.number_input(
                        "Renda mensal (R$)",
                        min_value=0.0,
                        value=5000.0,
                        step=500.0
                    )
                )


                renda_nao_informada = (
                    st.checkbox(
                        "Renda mensal não informada"
                    )
                )


                linhas_credito = (
                    st.number_input(
                        "Linhas de crédito abertas",
                        min_value=0,
                        value=5,
                        step=1
                    )
                )


            with col2:


                atrasos_30_59 = (
                    st.number_input(
                        "Atrasos de 30–59 dias",
                        min_value=0,
                        value=0,
                        step=1
                    )
                )


                atrasos_60_89 = (
                    st.number_input(
                        "Atrasos de 60–89 dias",
                        min_value=0,
                        value=0,
                        step=1
                    )
                )


                atrasos_90 = (
                    st.number_input(
                        "Atrasos de 90+ dias",
                        min_value=0,
                        value=0,
                        step=1
                    )
                )


                emprestimos_imobiliarios = (
                    st.number_input(
                        "Empréstimos imobiliários",
                        min_value=0,
                        value=1,
                        step=1
                    )
                )


                numero_dependentes = (
                    st.number_input(
                        "Número de dependentes",
                        min_value=0,
                        value=0,
                        step=1
                    )
                )


                dependentes_nao_informado = (
                    st.checkbox(
                        "Dependentes não informado"
                    )
                )


            analisar = (
                st.form_submit_button(

                    "🔎 Avaliar risco",

                    type="primary",

                    use_container_width=True
                )
            )


        if analisar:


            renda_modelo = (
                np.nan
                if renda_nao_informada
                else renda_mensal
            )


            dependentes_modelo = (
                np.nan
                if dependentes_nao_informado
                else numero_dependentes
            )


            cliente = preparar_cliente(

                utilizacao_credito,
                idade,
                atrasos_30_59,
                razao_divida,
                renda_modelo,
                linhas_credito,
                atrasos_90,
                emprestimos_imobiliarios,
                atrasos_60_89,
                dependentes_modelo
            )


            score, predicao, _ = (
                prever_cliente(
                    modelo,
                    cliente
                )
            )


            faixa = (
                classificar_risco(
                    score
                )
            )


            cor = (
                cor_risco(
                    score
                )
            )


            renderizar_html(
                """
                <div class="section-title">
                    Resultado da avaliação
                </div>
                """
            )


            r1, r2, r3, r4 = (
                st.columns(4)
            )


            with r1:

                renderizar_html(
                    f"""
                    <div class="kpi-card">

                        <div class="kpi-label">
                            Score de risco
                        </div>

                        <div
                            class="kpi-value"
                            style="
                                color:
                                {cor}
                                !important;
                            "
                        >
                            {score:.1%}
                        </div>

                        <div class="kpi-detail">
                            estimativa do modelo
                        </div>

                    </div>
                    """
                )


            with r2:


                decisao = (
                    "Priorizar"
                    if predicao == 1
                    else "Não priorizar"
                )


                renderizar_html(
                    f"""
                    <div class="kpi-card">

                        <div class="kpi-label">
                            Decisão
                        </div>

                        <div class="kpi-value">
                            {decisao}
                        </div>

                        <div class="kpi-detail">
                            threshold {THRESHOLD:.0%}
                        </div>

                    </div>
                    """
                )


            with r3:

                renderizar_html(
                    f"""
                    <div class="kpi-card">

                        <div class="kpi-label">
                            Faixa de score
                        </div>

                        <div
                            class="kpi-value"
                            style="
                                color:
                                {cor}
                                !important;
                            "
                        >
                            {faixa}
                        </div>

                        <div class="kpi-detail">
                            segmentação operacional
                        </div>

                    </div>
                    """
                )


            with r4:


                total_atrasos = (
                    atrasos_30_59
                    + atrasos_60_89
                    + atrasos_90
                )


                renderizar_html(
                    f"""
                    <div class="kpi-card">

                        <div class="kpi-label">
                            Total de atrasos
                        </div>

                        <div class="kpi-value">
                            {total_atrasos}
                        </div>

                        <div class="kpi-detail">
                            histórico informado
                        </div>

                    </div>
                    """
                )


            st.progress(
                float(score),
                text=(
                    f"Score de risco: "
                    f"{score:.1%}"
                )
            )


            if predicao == 1:


                renderizar_html(
                    f"""
                    <div class="business-card">

                        <h3>
                            ⚠️ Cliente priorizado
                        </h3>

                        <p>
                            O modelo atribuiu score de
                            <strong>
                                {score:.1%}
                            </strong>,
                            superior ao threshold
                            operacional de
                            <strong>
                                {THRESHOLD:.0%}
                            </strong>.
                        </p>

                        <p>
                            O cliente pode ser direcionado
                            para análise preventiva,
                            revisão de crédito ou
                            acompanhamento antecipado.
                        </p>

                    </div>
                    """
                )


            else:


                renderizar_html(
                    f"""
                    <div class="model-note">

                        O cliente apresentou score de
                        <strong>
                            {score:.1%}
                        </strong>,
                        abaixo do threshold operacional
                        de
                        <strong>
                            {THRESHOLD:.0%}
                        </strong>.

                        Neste cenário, ele não seria
                        priorizado pelo modelo.

                    </div>
                    """
                )


    except Exception as erro:


        st.error(
            "Não foi possível executar "
            f"o modelo: {erro}"
        )


# ==================================================
# SOBRE O PROJETO
# ==================================================

else:


    renderizar_html(
        """
        <div class="hero">

            <span class="badge">
                End-to-End Machine Learning
            </span>

            <h1>
                Sobre o Credit Risk Intelligence
            </h1>

            <p>
                Projeto de Machine Learning desenvolvido
                para demonstrar análise exploratória,
                modelagem de risco, otimização,
                explicabilidade e transformação do
                modelo em uma aplicação analítica.
            </p>

        </div>
        """
    )


    col1, col2 = (
        st.columns(2)
    )


    with col1:


        renderizar_html(
            """
            <div class="info-card">

                <h3>
                    🤖 Machine Learning
                </h3>

                <p>
                    Feature Engineering, comparação
                    de modelos, RandomizedSearch,
                    Optuna, XGBoost, ajuste de threshold,
                    curvas de aprendizado e SHAP.
                </p>

            </div>
            """
        )


    with col2:


        renderizar_html(
            """
            <div class="info-card">

                <h3>
                    💳 Produto de Dados
                </h3>

                <p>
                    O modelo treinado foi integrado
                    a uma aplicação Streamlit capaz
                    de avaliar a carteira, priorizar
                    clientes e realizar análises
                    individuais de risco.
                </p>

            </div>
            """
        )


    renderizar_html(
        """
        <div class="section-title">
            Arquitetura da solução
        </div>
        """
    )


    st.code(
        """
dados_modelo.csv
        ↓
Conjunto de Teste
        ↓
pipeline_modelo.pkl
        ↓
Pré-processamento
        ↓
XGBoost Otimizado
        ↓
Score de Inadimplência
        ↓
Threshold 0.40
        ↓
Priorização da Carteira
        ↓
Impacto Financeiro
        ↓
Streamlit Dashboard
        """,
        language="text"
    )


    renderizar_html(
        """
        <div class="section-title">
            Inferência individual
        </div>
        """
    )


    st.code(
        """
Dados do Cliente
        ↓
Feature Engineering
        ↓
total_atrasos
atraso_grave
        ↓
pipeline_modelo.pkl
        ↓
Score de Risco
        ↓
Threshold 0.40
        ↓
Priorizar / Não Priorizar
        """,
        language="text"
    )


    renderizar_html(
        """
        <div class="model-note">

            O artefato
            <strong>pipeline_modelo.pkl</strong>
            contém o pré-processamento utilizado
            no treinamento e o modelo XGBoost final.

            As features
            <strong>total_atrasos</strong>
            e
            <strong>atraso_grave</strong>
            são calculadas automaticamente durante
            a avaliação individual.

        </div>
        """
    )


    renderizar_html(
        """
        <div class="section-title">
            Tecnologias
        </div>
        """
    )


    st.code(
        """Python
Pandas
NumPy
Scikit-learn
XGBoost
Optuna
SHAP
Plotly
Streamlit
Joblib""",
        language="text"
    )


# ==================================================
# RODAPÉ
# ==================================================

renderizar_html(
    """
    <div class="footer">

        Credit Risk Intelligence •
        Machine Learning + Credit Risk Analytics

    </div>
    """
)