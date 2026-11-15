"""Enumeracoes e catalogos do schema.

Aqui vive a unica fonte de verdade sobre o que e combinacao valida entre
tipo de variavel e distribuicao, e quais sao os parametros default de cada
distribuicao. A UI deriva as cascatas daqui; o motor valida contra aqui.
"""

from __future__ import annotations

from enum import Enum


class DType(str, Enum):
    CONTINUA = "numerica_continua"
    DISCRETA = "numerica_discreta"
    ORDINAL = "ordinal"
    CATEGORICA = "categorica"
    BOOLEANA = "booleana"
    DATA_HORA = "data_hora"
    TEXTO = "texto"
    DERIVADA = "derivada"


class Distribuicao(str, Enum):
    NORMAL = "normal"
    UNIFORME = "uniforme"
    LOG_NORMAL = "log_normal"
    EXPONENCIAL = "exponencial"
    GAMA = "gama"
    BETA = "beta"
    TRIANGULAR = "triangular"
    POISSON = "poisson"
    BINOMIAL = "binomial"
    BINOMIAL_NEGATIVA = "binomial_negativa"
    GEOMETRICA = "geometrica"
    EMPIRICA = "empirica"
    CATEGORIAS = "categorias"
    BERNOULLI = "bernoulli"
    INTERVALO_DATAS = "intervalo_datas"
    ID = "id"


class MecanismoNulo(str, Enum):
    MCAR = "mcar"
    MAR = "mar"


class MetodoOutlier(str, Enum):
    NENHUM = "nenhum"
    CAUDA = "cauda"
    IQR = "iqr"
    FIXO = "fixo"


class Transformacao(str, Enum):
    NENHUMA = "nenhuma"
    LOG = "log"
    RAIZ = "raiz"
    ZSCORE = "zscore"
    MINMAX = "minmax"


class Medida(str, Enum):
    SPEARMAN = "spearman"
    PEARSON = "pearson"


class TargetTipo(str, Enum):
    BINARIA = "binaria"
    MULTICLASSE = "multiclasse"
    REGRESSAO = "regressao"
    CONTAGEM = "contagem"
    ORDINAL = "ordinal"


class TargetModo(str, Enum):
    GUIADO = "guiado"
    REGRA = "regra"


class TextoModo(str, Enum):
    SEQUENCIAL = "sequencial"
    FORMATADO = "formatado"
    ALEATORIO = "aleatorio"


class Alfabeto(str, Enum):
    NUMERIC = "numeric"
    ALPHA = "alpha"
    ALNUM = "alnum"


#: Distribuicoes legais por tipo de variavel.
DISTRIBUICOES_POR_TIPO: dict[DType, list[Distribuicao]] = {
    DType.CONTINUA: [
        Distribuicao.NORMAL,
        Distribuicao.UNIFORME,
        Distribuicao.LOG_NORMAL,
        Distribuicao.EXPONENCIAL,
        Distribuicao.GAMA,
        Distribuicao.BETA,
        Distribuicao.TRIANGULAR,
        Distribuicao.EMPIRICA,
    ],
    DType.DISCRETA: [
        Distribuicao.NORMAL,
        Distribuicao.UNIFORME,
        Distribuicao.LOG_NORMAL,
        Distribuicao.EXPONENCIAL,
        Distribuicao.TRIANGULAR,
        Distribuicao.POISSON,
        Distribuicao.BINOMIAL,
        Distribuicao.BINOMIAL_NEGATIVA,
        Distribuicao.GEOMETRICA,
        Distribuicao.EMPIRICA,
    ],
    DType.ORDINAL: [Distribuicao.CATEGORIAS, Distribuicao.EMPIRICA],
    DType.CATEGORICA: [Distribuicao.CATEGORIAS, Distribuicao.EMPIRICA],
    DType.BOOLEANA: [Distribuicao.BERNOULLI],
    DType.DATA_HORA: [Distribuicao.INTERVALO_DATAS],
    DType.TEXTO: [Distribuicao.ID],
    #: Derivadas nao tem distribuicao: o valor vem de uma expressao.
    DType.DERIVADA: [],
}


class ModoEvolucao(str, Enum):
    """Como a variavel se comporta ao longo dos meses do painel."""

    ESTATICA = "estatica"
    AR1 = "ar1"
    TENDENCIA = "tendencia"


ROTULOS_EVOLUCAO = {
    ModoEvolucao.ESTATICA: "Constante nos meses",
    ModoEvolucao.AR1: "Persistente (autocorrelacao)",
    ModoEvolucao.TENDENCIA: "Cresce a cada mes",
}

#: Parametros default de cada distribuicao. Copia rasa usada ao trocar de tipo.
PARAMS_DEFAULT: dict[Distribuicao, dict] = {
    Distribuicao.NORMAL: {"mu": 0.0, "sigma": 1.0},
    Distribuicao.UNIFORME: {"a": 0.0, "b": 1.0},
    Distribuicao.LOG_NORMAL: {"mu_log": 2.0, "sigma_log": 0.8},
    Distribuicao.EXPONENCIAL: {"lam": 1.0},
    Distribuicao.GAMA: {"shape": 2.0, "scale": 1.0},
    Distribuicao.BETA: {"alpha": 2.0, "beta": 2.0},
    Distribuicao.TRIANGULAR: {"min": 0.0, "moda": 5.0, "max": 10.0},
    Distribuicao.POISSON: {"lam": 2.0},
    Distribuicao.BINOMIAL: {"n": 10, "p": 0.5},
    Distribuicao.BINOMIAL_NEGATIVA: {"r": 3.0, "p": 0.5},
    Distribuicao.GEOMETRICA: {"p": 0.3},
    Distribuicao.EMPIRICA: {
        "itens": [
            {"valor": 1.0, "peso": 1.0},
            {"valor": 2.0, "peso": 1.0},
            {"valor": 3.0, "peso": 1.0},
        ]
    },
    Distribuicao.CATEGORIAS: {
        "itens": [
            {"rotulo": "A", "peso": 1.0},
            {"rotulo": "B", "peso": 1.0},
        ]
    },
    Distribuicao.BERNOULLI: {"p": 0.5},
    Distribuicao.INTERVALO_DATAS: {
        "inicio": "2024-01-01",
        "fim": "2025-12-31",
        "com_hora": False,
        "tendencia": "nenhuma",
    },
    Distribuicao.ID: {
        "modo": "formatado",
        "prefixo": "ID-",
        "n_digitos": 5,
        "comprimento": 8,
        "alfabeto": "alnum",
        "unico": True,
    },
}

#: Tipos que passam pela copula (texto nao tem ordinalidade).
TIPOS_COPULAVEIS = {
    DType.CONTINUA,
    DType.DISCRETA,
    DType.ORDINAL,
    DType.CATEGORICA,
    DType.BOOLEANA,
    DType.DATA_HORA,
}

#: Tipos numericos (aceitam escala, outliers, influencia no target).
TIPOS_NUMERICOS = {DType.CONTINUA, DType.DISCRETA}

ROTULOS_DTYPE = {
    DType.CONTINUA: "Numerica continua",
    DType.DISCRETA: "Numerica discreta (contagem)",
    DType.ORDINAL: "Ordinal",
    DType.CATEGORICA: "Categorica nominal",
    DType.BOOLEANA: "Booleana",
    DType.DATA_HORA: "Data / hora",
    DType.TEXTO: "Texto (ID)",
    DType.DERIVADA: "Derivada (expressao)",
}

ROTULOS_DISTRIBUICAO = {
    Distribuicao.NORMAL: "Normal",
    Distribuicao.UNIFORME: "Uniforme",
    Distribuicao.LOG_NORMAL: "Log-normal",
    Distribuicao.EXPONENCIAL: "Exponencial",
    Distribuicao.GAMA: "Gama",
    Distribuicao.BETA: "Beta",
    Distribuicao.TRIANGULAR: "Triangular",
    Distribuicao.POISSON: "Poisson",
    Distribuicao.BINOMIAL: "Binomial",
    Distribuicao.BINOMIAL_NEGATIVA: "Binomial negativa",
    Distribuicao.GEOMETRICA: "Geometrica",
    Distribuicao.EMPIRICA: "Empirica (valor + peso)",
    Distribuicao.CATEGORIAS: "Categorias com pesos",
    Distribuicao.BERNOULLI: "Bernoulli",
    Distribuicao.INTERVALO_DATAS: "Intervalo de datas",
    Distribuicao.ID: "Identificador",
}
