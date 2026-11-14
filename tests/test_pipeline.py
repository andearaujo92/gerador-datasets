"""Pipeline: ordem, reprodutibilidade, limites e nulos."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gerador.engine.pipeline import gerar_dataset
from gerador.schema.dataset import DatasetConfig
from gerador.schema.relations import Relacao, RelacoesConfig
from gerador.schema.target import Influencia, TargetConfig
from gerador.schema.variable import VariableConfig

from tests.conftest import var


def test_gera_dataset_com_colunas_na_ordem(config_basica):
    res = gerar_dataset(config_basica)
    assert res.erros == []
    assert list(res.df.columns) == ["renda", "saldo", "num_tvs"]
    assert len(res.df) == config_basica.n_linhas
    assert res.df["num_tvs"].dtype == "int64"


def test_reprodutivel_com_mesma_seed():
    a = DatasetConfig(n_linhas=500, seed=99, variaveis=[var("x", params={"mu": 0, "sigma": 1})])
    b = DatasetConfig(n_linhas=500, seed=99, variaveis=[var("x", params={"mu": 0, "sigma": 1})])
    assert gerar_dataset(a).df.equals(gerar_dataset(b).df)


def test_seed_diferente_muda_resultado():
    a = DatasetConfig(n_linhas=500, seed=1, variaveis=[var("x", params={"mu": 0, "sigma": 1})])
    b = DatasetConfig(n_linhas=500, seed=2, variaveis=[var("x", params={"mu": 0, "sigma": 1})])
    assert not gerar_dataset(a).df.equals(gerar_dataset(b).df)


def test_escala_multiplicador_e_limites():
    v = var("valor", params={"mu": 0, "sigma": 1})
    v.escala.multiplicador = 1000
    v.escala.offset = 500
    v.escala.minimo = 0
    v.escala.maximo = 2000
    v.escala.permitir_negativo = False
    res = gerar_dataset(DatasetConfig(n_linhas=2000, seed=3, variaveis=[v]))
    assert res.df["valor"].min() >= 0
    assert res.df["valor"].max() <= 2000


def test_target_vem_por_ultimo_e_enxerga_features_integras():
    alvo = var("sinal", params={"mu": 0, "sigma": 1})
    alvo.qualidade.nulos.pct = 60.0
    cfg = DatasetConfig(
        n_linhas=4000,
        seed=5,
        variaveis=[alvo, var("outra", params={"mu": 0, "sigma": 1})],
        target=TargetConfig(
            habilitado=True,
            nome="y",
            tipo="binaria",
            params={"prevalencia_base": 0.5, "ruido": 0.0, "forcar_prevalencia": False},
            influencias=[Influencia("sinal", 1, 1.0)],
        ),
    )
    res = gerar_dataset(cfg)
    assert list(res.df.columns) == ["sinal", "outra", "y"]
    assert res.df["y"].isna().sum() == 0
    assert res.df["sinal"].isna().mean() > 0.5
    # o sinal continua apreensivel onde a feature existe
    validos = res.df.dropna(subset=["sinal"])
    assert validos.groupby("y")["sinal"].mean().diff().iloc[-1] > 0.5


def test_nulos_mcar_aproximam_percentual():
    v = var("x", params={"mu": 0, "sigma": 1})
    v.qualidade.nulos.pct = 20.0
    res = gerar_dataset(DatasetConfig(n_linhas=10_000, seed=6, variaveis=[v]))
    assert abs(res.df["x"].isna().mean() - 0.2) < 0.02


def test_nulos_mar_dependem_da_outra_variavel():
    a = var("a", params={"mu": 0, "sigma": 1})
    b = var("b", params={"mu": 0, "sigma": 1})
    b.qualidade.nulos.pct = 30.0
    b.qualidade.nulos.mecanismo = "mar"
    b.qualidade.nulos.depende_de = "a"
    res = gerar_dataset(DatasetConfig(n_linhas=20_000, seed=7, variaveis=[a, b]))
    df = res.df
    assert df["b"].isna().mean() == pytest.approx(0.3, abs=0.04)
    # quem tem nulo em b tem 'a' maior em media (propensao crescente)
    assert df.loc[df["b"].isna(), "a"].mean() > df.loc[df["b"].notna(), "a"].mean()


def test_representacao_de_nulo_em_texto():
    v = var("x", params={"mu": 0, "sigma": 1})
    v.qualidade.nulos.pct = 10.0
    v.qualidade.nulos.representacao = "desconhecido"
    res = gerar_dataset(DatasetConfig(n_linhas=2000, seed=8, variaveis=[v]))
    assert (res.df["x"] == "desconhecido").any()


def test_outliers_aumentam_amplitude():
    v = var("x", params={"mu": 0, "sigma": 1})
    base = gerar_dataset(DatasetConfig(n_linhas=4000, seed=9, variaveis=[v])).df["x"]
    v.qualidade.outliers.pct = 5.0
    v.qualidade.outliers.metodo = "iqr"
    v.qualidade.outliers.intensidade = 3.0
    com = gerar_dataset(DatasetConfig(n_linhas=4000, seed=9, variaveis=[v])).df["x"]
    assert com.max() > base.max()


def test_correlacao_realizada_aparece_no_comparativo(config_basica):
    config_basica.relacoes = RelacoesConfig(lista=[Relacao("renda", "saldo", 0.7)])
    res = gerar_dataset(config_basica)
    tabela = res.comparativo
    assert len(tabela) == 1
    assert abs(tabela.iloc[0]["delta"]) < 0.05


def test_texto_id_no_pipeline():
    v = VariableConfig(
        nome="id_cliente",
        dtype="texto",
        distribuicao="id",
        params={"modo": "formatado", "prefixo": "C-", "n_digitos": 6},
    )
    res = gerar_dataset(DatasetConfig(n_linhas=100, seed=10, variaveis=[v]))
    assert res.df["id_cliente"].iloc[0] == "C-000001"
    assert res.df["id_cliente"].is_unique


def test_erro_de_validacao_nao_gera_dataframe():
    cfg = DatasetConfig(n_linhas=10, variaveis=[])
    res = gerar_dataset(cfg)
    assert res.df is None
    assert res.erros


def test_booleana_e_ordinal_preservam_tipo_e_ordem():
    ativo = VariableConfig(
        nome="ativo", dtype="booleana", distribuicao="bernoulli", params={"p": 0.3}
    )
    nivel = VariableConfig(
        nome="nivel",
        dtype="ordinal",
        distribuicao="categorias",
        params={
            "itens": [
                {"rotulo": "alto", "peso": 1.0},
                {"rotulo": "baixo", "peso": 1.0},
                {"rotulo": "medio", "peso": 1.0},
            ]
        },
    )
    res = gerar_dataset(DatasetConfig(n_linhas=3000, seed=12, variaveis=[ativo, nivel]))
    assert res.erros == []
    assert res.df["ativo"].dtype == bool
    assert abs(res.df["ativo"].mean() - 0.3) < 0.03

    serie = res.df["nivel"]
    assert isinstance(serie.dtype, pd.CategoricalDtype)
    # ordem DECLARADA, nao alfabetica
    assert list(serie.cat.categories) == ["alto", "baixo", "medio"]
    assert serie.cat.ordered is True


def test_empirica_no_pipeline_produz_so_os_valores_listados():
    v = VariableConfig(
        nome="score",
        dtype="numerica_discreta",
        distribuicao="empirica",
        params={
            "itens": [
                {"valor": 10.0, "peso": 3.0},
                {"valor": 20.0, "peso": 1.0},
            ]
        },
    )
    res = gerar_dataset(DatasetConfig(n_linhas=4000, seed=13, variaveis=[v]))
    assert set(res.df["score"].unique()) == {10, 20}
    assert abs((res.df["score"] == 10).mean() - 0.75) < 0.03


def test_data_hora_vira_datetime():
    v = VariableConfig(
        nome="data",
        dtype="data_hora",
        distribuicao="intervalo_datas",
        params={"inicio": "2024-01-01", "fim": "2024-06-30", "tendencia": "crescente"},
    )
    res = gerar_dataset(DatasetConfig(n_linhas=500, seed=11, variaveis=[v]))
    assert str(res.df["data"].dtype).startswith("datetime64")
    assert res.df["data"].min().year == 2024
