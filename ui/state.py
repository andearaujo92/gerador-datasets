"""Estado da aplicacao (session_state) e configuracao default."""

from __future__ import annotations

import copy
import hashlib

import streamlit as st

from gerador.io.profile import carregar_schema, salvar_schema
from gerador.schema.dataset import DatasetConfig
from gerador.schema.relations import Relacao, RelacoesConfig
from gerador.schema.target import Influencia, TargetConfig
from gerador.schema.types import DType, Distribuicao, MetodoOutlier
from gerador.schema.variable import Qualidade, VariableConfig

CHAVE_CONFIG = "gd_config"
CHAVE_RESULTADO = "gd_resultado"
CHAVE_HASH = "gd_hash_gerado"


def _v(nome, dtype, dist, params, **kwargs) -> VariableConfig:
    var = VariableConfig(nome=nome, dtype=dtype, distribuicao=dist, params=params)
    for chave, valor in kwargs.items():
        setattr(var, chave, valor)
    return var


def config_padrao() -> DatasetConfig:
    """Configuracao de exemplo: o app abre util em vez de vazio."""
    idade = _v(
        "idade",
        DType.CONTINUA.value,
        Distribuicao.NORMAL.value,
        {"mu": 38.0, "sigma": 12.0},
    )
    idade.escala.minimo = 18.0
    idade.escala.maximo = 90.0
    idade.escala.casas_decimais = 0
    idade.escala.permitir_negativo = False
    idade.descricao = "Idade do cliente em anos"

    renda = _v(
        "renda",
        DType.CONTINUA.value,
        Distribuicao.LOG_NORMAL.value,
        {"mu_log": 7.6, "sigma_log": 0.6},
    )
    renda.escala.multiplicador = 1.0
    renda.escala.casas_decimais = 2
    renda.escala.permitir_negativo = False
    renda.descricao = "Renda mensal em reais"

    compras = _v(
        "compras_12m",
        DType.DISCRETA.value,
        Distribuicao.POISSON.value,
        {"lam": 3.0},
    )
    compras.escala.permitir_negativo = False
    compras.descricao = "Numero de compras nos ultimos 12 meses"

    plano = _v(
        "plano",
        DType.CATEGORICA.value,
        Distribuicao.CATEGORIAS.value,
        {
            "itens": [
                {"rotulo": "basico", "peso": 5.0},
                {"rotulo": "padrao", "peso": 3.0},
                {"rotulo": "premium", "peso": 2.0},
            ]
        },
    )
    plano.descricao = "Plano contratado"

    id_cliente = _v(
        "id_cliente",
        DType.TEXTO.value,
        Distribuicao.ID.value,
        {
            "modo": "formatado",
            "prefixo": "CLI-",
            "n_digitos": 6,
            "comprimento": 8,
            "alfabeto": "alnum",
            "unico": True,
        },
    )
    id_cliente.descricao = "Chave de juncao do cliente"

    return DatasetConfig(
        n_linhas=1000,
        seed=42,
        variaveis=[id_cliente, idade, renda, compras, plano],
        relacoes=RelacoesConfig(
            lista=[Relacao("idade", "renda", 0.45), Relacao("renda", "compras_12m", 0.35)]
        ),
        target=TargetConfig(
            habilitado=True,
            nome="churn",
            tipo="binaria",
            params={"prevalencia_base": 0.25, "ruido": 0.05, "forcar_prevalencia": True},
            influencias=[
                Influencia("compras_12m", -1, 0.8),
                Influencia("idade", 1, 0.3),
            ],
        ),
    )


def obter_config() -> DatasetConfig:
    if CHAVE_CONFIG not in st.session_state:
        st.session_state[CHAVE_CONFIG] = config_padrao()
    return st.session_state[CHAVE_CONFIG]


def _limpar_widgets() -> None:
    """Remove chaves de widget (todas contem '.') para recarregar valores do schema."""
    for chave in [k for k in st.session_state.keys() if isinstance(k, str) and "." in k]:
        del st.session_state[chave]


def definir_config(config: DatasetConfig, *, limpar_widgets: bool = True) -> None:
    st.session_state[CHAVE_CONFIG] = config
    st.session_state.pop(CHAVE_RESULTADO, None)
    st.session_state.pop(CHAVE_HASH, None)
    if limpar_widgets:
        _limpar_widgets()


def carregar_de_json(texto: str | bytes) -> DatasetConfig:
    return carregar_schema(texto)


def json_atual(config: DatasetConfig) -> str:
    return salvar_schema(config)


def hash_config(config: DatasetConfig) -> str:
    return hashlib.sha256(json_atual(config).encode("utf-8")).hexdigest()


def resultado():
    return st.session_state.get(CHAVE_RESULTADO)


def guardar_resultado(res, chave_hash: str) -> None:
    st.session_state[CHAVE_RESULTADO] = res
    st.session_state[CHAVE_HASH] = chave_hash


def resultado_desatualizado(config: DatasetConfig) -> bool:
    atual = st.session_state.get(CHAVE_HASH)
    if atual is None:
        return False
    return atual != hash_config(config)


def nome_livre(config: DatasetConfig) -> str:
    i = len(config.variaveis) + 1
    while f"var_{i}" in config.nomes:
        i += 1
    return f"var_{i}"


def remover_variavel(config: DatasetConfig, var_id: str) -> None:
    alvo = next((v for v in config.variaveis if v.id == var_id), None)
    if alvo is None:
        return
    config.variaveis = [v for v in config.variaveis if v.id != var_id]
    config.relacoes.lista = [
        r for r in config.relacoes.lista if alvo.nome not in (r.a, r.b)
    ]
    config.target.influencias = [
        i for i in config.target.influencias if i.variavel != alvo.nome
    ]
