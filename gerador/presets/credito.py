"""Preset: base de credito mensal (painel de 12 meses, sem target).

A base entrega **somente o estado em t**: uma linha por cliente por mes, sem
lags, sem janelas moveis e sem agregados. A engenharia de features e o rotulo
ficam com quem consome o arquivo — de proposito, para nao vazar informacao do
futuro para o passado.
"""

from __future__ import annotations

from gerador.schema.dataset import DatasetConfig
from gerador.schema.painel import PainelConfig
from gerador.schema.relations import Relacao, RelacoesConfig
from gerador.schema.target import TargetConfig
from gerador.schema.types import DType, Distribuicao, ModoEvolucao
from gerador.schema.variable import Escala, Evolucao, VariableConfig

N_CLIENTES = 6000
N_MESES = 12
DATA_INICIO = "2025-12-01"


def _escala(
    *,
    casas: int = 2,
    minimo: float | None = None,
    maximo: float | None = None,
    negativo: bool = False,
) -> Escala:
    return Escala(
        casas_decimais=casas,
        minimo=minimo,
        maximo=maximo,
        permitir_negativo=negativo,
    )


def _v(
    nome: str,
    dtype: DType,
    distribuicao: Distribuicao,
    params: dict,
    *,
    descricao: str = "",
    escala: Escala | None = None,
    phi: float | None = None,
    modo: ModoEvolucao = ModoEvolucao.ESTATICA,
    passo: float = 1.0,
) -> VariableConfig:
    return VariableConfig(
        nome=nome,
        dtype=dtype.value,
        distribuicao=distribuicao.value,
        params=params,
        descricao=descricao,
        escala=escala or Escala(),
        evolucao=Evolucao(modo=modo.value, phi=phi if phi is not None else 0.9, passo=passo),
    )


def _derivada(nome: str, expressao: str, descricao: str) -> VariableConfig:
    return VariableConfig(
        nome=nome,
        dtype=DType.DERIVADA.value,
        expressao=expressao,
        descricao=descricao,
        escala=_escala(),
    )


def _categorias(itens: list[tuple[str, float]]) -> dict:
    return {"itens": [{"rotulo": rotulo, "peso": peso} for rotulo, peso in itens]}


def _empirica(itens: list[tuple[float, float]]) -> dict:
    return {"itens": [{"valor": valor, "peso": peso} for valor, peso in itens]}


#: UFs na ordem declarada do menor para o maior perfil medio de renda. A ordem
#: e o que a copula enxerga: nao reordene alfabeticamente.
UF = [
    ("MA", 6776),
    ("PI", 3271),
    ("AL", 3127),
    ("PA", 8121),
    ("PB", 3974),
    ("CE", 8794),
    ("BA", 14141),
    ("PE", 9058),
    ("AM", 3941),
    ("RN", 3303),
    ("SE", 2210),
    ("RO", 1581),
    ("TO", 1511),
    ("AC", 830),
    ("AP", 733),
    ("RR", 636),
    ("MG", 20539),
    ("ES", 3833),
    ("PR", 11444),
    ("GO", 7056),
    ("MT", 3658),
    ("MS", 2757),
    ("RS", 10882),
    ("SC", 7610),
    ("RJ", 16055),
    ("DF", 2817),
    ("SP", 44411),
]

OCUPACAO = [
    ("desempregado", 6),
    ("estudante", 4),
    ("autonomo", 25),
    ("aposentado", 15),
    ("clt", 40),
    ("servidor_publico", 10),
]

ESCOLARIDADE = [
    ("fundamental", 32),
    ("medio", 44),
    ("superior", 19),
    ("pos_graduacao", 5),
]

SISTEMA_CELULAR = [
    ("Android", 70),
    ("outro", 2),
    ("iOS", 28),
]

#: (a, b, forca) — Spearman pedido. Derivadas ficam de fora: a correlacao delas
#: vem da formula, nao da copula.
RELACOES = [
    ("renda", "valor_entradas", 0.85),
    ("renda", "valor_saidas", 0.60),
    ("renda", "limite", 0.65),
    ("renda", "patrimonio", 0.55),
    ("renda", "ocupacao", 0.50),
    ("renda", "escolaridade", 0.45),
    ("renda", "score_bureau_A", 0.45),
    ("renda", "score_bureau_B", 0.40),
    ("renda", "idade", 0.35),
    ("renda", "UF", 0.35),
    ("renda", "sistema_celular", 0.25),
    ("renda", "tempo_de_conta", 0.30),
    ("renda", "valor_a_vencer", 0.40),
    ("renda", "atraso_em_dias", -0.30),
    ("renda", "qtd_instituicoes", 0.10),
    ("score_bureau_A", "score_bureau_B", 0.75),
    ("score_bureau_A", "atraso_em_dias", -0.55),
    ("score_bureau_B", "atraso_em_dias", -0.50),
    ("score_bureau_A", "qtd_instituicoes", -0.35),
    ("score_bureau_A", "contratos_ativos", -0.20),
    ("score_bureau_A", "idade", 0.20),
    ("score_bureau_A", "escolaridade", 0.30),
    ("score_bureau_B", "idade", 0.15),
    ("atraso_em_dias", "valor_a_vencer", 0.15),
    ("qtd_instituicoes", "contratos_ativos", 0.60),
    ("qtd_instituicoes", "limite", 0.35),
    ("qtd_instituicoes", "valor_a_vencer", 0.20),
    ("contratos_ativos", "valor_a_vencer", 0.35),
    ("limite", "valor_a_vencer", 0.55),
    ("limite", "patrimonio", 0.45),
    ("valor_a_vencer", "valor_entradas", 0.30),
    ("valor_entradas", "valor_saidas", 0.80),
    ("valor_entradas", "patrimonio", 0.50),
    ("valor_saidas", "patrimonio", 0.40),
    ("idade", "patrimonio", 0.40),
    ("idade", "tempo_de_conta", 0.35),
    ("idade", "escolaridade", 0.30),
    ("tempo_de_conta", "contratos_ativos", 0.20),
    ("UF", "escolaridade", 0.20),
    ("UF", "ocupacao", 0.20),
    ("ocupacao", "escolaridade", 0.35),
    ("sistema_celular", "escolaridade", 0.20),
]


def credito_mensal() -> DatasetConfig:
    """Painel mensal de credito: 6.000 clientes x 12 meses, sem target."""
    variaveis = [
        _v(
            "id_cliente",
            DType.TEXTO,
            Distribuicao.ID,
            {
                "modo": "formatado",
                "prefixo": "CLI-",
                "n_digitos": 6,
                "comprimento": 8,
                "alfabeto": "alnum",
                "unico": True,
            },
            descricao="Identificador do cliente (chave do painel)",
        ),
        _v(
            "idade",
            DType.CONTINUA,
            Distribuicao.NORMAL,
            {"mu": 39.0, "sigma": 12.0},
            descricao="Idade do cliente em anos (constante no painel)",
            escala=_escala(casas=0, minimo=18, maximo=90),
        ),
        _v(
            "sistema_celular",
            DType.CATEGORICA,
            Distribuicao.CATEGORIAS,
            _categorias(SISTEMA_CELULAR),
            descricao="Sistema operacional do celular (constante no painel)",
        ),
        _v(
            "limite",
            DType.CONTINUA,
            Distribuicao.LOG_NORMAL,
            {"mu_log": 9.0, "sigma_log": 0.8},
            descricao="Limite de credito individual (constante no painel)",
            escala=_escala(minimo=500, maximo=200_000),
        ),
        _v(
            "renda",
            DType.CONTINUA,
            Distribuicao.LOG_NORMAL,
            {"mu_log": 7.6, "sigma_log": 0.6},
            descricao="Renda mensal declarada",
            escala=_escala(minimo=780),
            phi=0.90,
            modo=ModoEvolucao.AR1,
        ),
        _v(
            "score_bureau_A",
            DType.CONTINUA,
            Distribuicao.NORMAL,
            {"mu": 600.0, "sigma": 110.0},
            descricao="Score de bureau A (0-1000)",
            escala=_escala(casas=0, minimo=0, maximo=1000),
            phi=0.95,
            modo=ModoEvolucao.AR1,
        ),
        _v(
            "qtd_instituicoes",
            DType.DISCRETA,
            Distribuicao.POISSON,
            {"lam": 2.4},
            descricao="Quantidade de instituicoes com relacionamento no mes",
            escala=_escala(casas=0, minimo=0, maximo=15),
            phi=0.95,
            modo=ModoEvolucao.AR1,
        ),
        _v(
            "contratos_ativos",
            DType.DISCRETA,
            Distribuicao.POISSON,
            {"lam": 1.8},
            descricao="Contratos de credito ativos no mes",
            escala=_escala(casas=0, minimo=0, maximo=12),
            phi=0.92,
            modo=ModoEvolucao.AR1,
        ),
        _v(
            "valor_a_vencer",
            DType.CONTINUA,
            Distribuicao.LOG_NORMAL,
            {"mu_log": 6.8, "sigma_log": 0.9},
            descricao="Valor a vencer no mes (saldo devedor futuro)",
            escala=_escala(),
            phi=0.85,
            modo=ModoEvolucao.AR1,
        ),
        _v(
            "atraso_em_dias",
            DType.DISCRETA,
            Distribuicao.EMPIRICA,
            _empirica(
                [
                    (0.0, 70.0),
                    (5.0, 2.0),
                    (15.0, 4.0),
                    (30.0, 5.0),
                    (45.0, 3.0),
                    (60.0, 3.0),
                    (90.0, 2.5),
                    (120.0, 1.5),
                    (180.0, 1.0),
                    (240.0, 0.5),
                    (360.0, 0.5),
                ]
            ),
            descricao="Atraso em dias na data de referencia (0 = em dia)",
            escala=_escala(casas=0, minimo=0),
            phi=0.97,
            modo=ModoEvolucao.AR1,
        ),
        _v(
            "valor_entradas",
            DType.CONTINUA,
            Distribuicao.LOG_NORMAL,
            {"mu_log": 7.75, "sigma_log": 0.55},
            descricao="Soma das entradas financeiras no mes",
            escala=_escala(),
            phi=0.80,
            modo=ModoEvolucao.AR1,
        ),
        _v(
            "valor_saidas",
            DType.CONTINUA,
            Distribuicao.LOG_NORMAL,
            {"mu_log": 7.70, "sigma_log": 0.55},
            descricao="Soma das saidas financeiras no mes",
            escala=_escala(),
            phi=0.80,
            modo=ModoEvolucao.AR1,
        ),
        _v(
            "patrimonio",
            DType.CONTINUA,
            Distribuicao.LOG_NORMAL,
            {"mu_log": 11.0, "sigma_log": 1.2},
            descricao="Patrimonio declarado",
            escala=_escala(maximo=20_000_000),
            phi=0.98,
            modo=ModoEvolucao.AR1,
        ),
        _v(
            "UF",
            DType.CATEGORICA,
            Distribuicao.CATEGORIAS,
            _categorias(UF),
            descricao="Unidade federativa (constante no painel)",
        ),
        _v(
            "ocupacao",
            DType.CATEGORICA,
            Distribuicao.CATEGORIAS,
            _categorias(OCUPACAO),
            descricao="Ocupacao principal",
            phi=0.97,
            modo=ModoEvolucao.AR1,
        ),
        _v(
            "escolaridade",
            DType.ORDINAL,
            Distribuicao.CATEGORIAS,
            _categorias(ESCOLARIDADE),
            descricao="Escolaridade (constante no painel)",
        ),
        _v(
            "score_bureau_B",
            DType.CONTINUA,
            Distribuicao.NORMAL,
            {"mu": 550.0, "sigma": 130.0},
            descricao="Score de bureau B (0-1000)",
            escala=_escala(casas=0, minimo=0, maximo=1000),
            phi=0.95,
            modo=ModoEvolucao.AR1,
        ),
        _v(
            "tempo_de_conta",
            DType.DISCRETA,
            Distribuicao.UNIFORME,
            {"a": 1.0, "b": 240.0},
            descricao="Tempo de relacionamento em meses (cresce 1 por mes)",
            escala=_escala(casas=0, minimo=1),
            modo=ModoEvolucao.TENDENCIA,
            passo=1.0,
        ),
        _derivada(
            "valor_vencido",
            "(atraso_em_dias > 0) * valor_a_vencer * ruido_uniforme(0.1, 0.9)",
            "Parcela vencida: fracao do valor a vencer quando ha atraso",
        ),
        _derivada(
            "valor_prejuizo",
            "(atraso_em_dias >= 180) * valor_vencido * ruido_uniforme(0.5, 0.8)",
            "Parcela em prejuizo: fracao da divida vencida com atraso >= 180 dias",
        ),
        _derivada(
            "valor_refinanciado",
            "(atraso_em_dias >= 90) * (ruido_uniforme() < 0.20) * valor_vencido "
            "* ruido_uniforme(0.9, 1.3)",
            "Valor refinanciado no mes (20% dos casos com atraso >= 90 dias)",
        ),
        _derivada(
            "limite_total_instituicoes",
            "maximum(limite, limite * (1 + ruido_uniforme(0.2, 2.5)))",
            "Soma dos limites em todas as instituicoes",
        ),
    ]

    relacoes = RelacoesConfig(
        lista=[Relacao(a, b, forca) for a, b, forca in RELACOES],
        calibrar=True,
        tolerancia=0.04,
        max_iter=10,
        completar_psd=True,
    )

    return DatasetConfig(
        n_linhas=N_CLIENTES,
        seed=42,
        variaveis=variaveis,
        relacoes=relacoes,
        target=TargetConfig(habilitado=False),
        painel=PainelConfig(
            habilitado=True,
            coluna_id="id_cliente",
            coluna_tempo="data_mes",
            data_inicio=DATA_INICIO,
            n_meses=N_MESES,
        ),
    )
