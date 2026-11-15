"""Painel temporal: uma linha por entidade x mes, com estado do mes."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gerador.engine.pipeline import gerar_dataset
from gerador.schema.dataset import DatasetConfig
from gerador.schema.painel import PainelConfig, meses_do_painel, normalizar_inicio
from gerador.schema.relations import Relacao, RelacoesConfig
from gerador.schema.types import DType, Distribuicao, ModoEvolucao
from gerador.schema.variable import Evolucao, Escala, VariableConfig

from tests.conftest import var


def _id(nome: str = "id_cliente") -> VariableConfig:
    return VariableConfig(
        nome=nome,
        dtype=DType.TEXTO.value,
        distribuicao=Distribuicao.ID.value,
        params={"modo": "sequencial", "n_digitos": 6},
    )


def _config(n: int = 400, meses: int = 5, seed: int = 3) -> DatasetConfig:
    renda = var("renda", params={"mu": 1000.0, "sigma": 200.0})
    renda.escala = Escala(minimo=100)
    renda.evolucao = Evolucao(modo=ModoEvolucao.AR1.value, phi=0.85)

    fixo = var("fixo", params={"mu": 10.0, "sigma": 2.0})

    qtd = var(
        "qtd",
        dtype=DType.DISCRETA.value,
        distribuicao=Distribuicao.POISSON.value,
        params={"lam": 2.5},
    )
    qtd.escala = Escala(casas_decimais=0, minimo=0)
    qtd.evolucao = Evolucao(modo=ModoEvolucao.AR1.value, phi=0.9)

    categoria = var(
        "categoria",
        dtype=DType.CATEGORICA.value,
        distribuicao=Distribuicao.CATEGORIAS.value,
        params={
            "itens": [
                {"rotulo": "baixo", "peso": 5},
                {"rotulo": "medio", "peso": 3},
                {"rotulo": "alto", "peso": 2},
            ]
        },
    )
    categoria.evolucao = Evolucao(modo=ModoEvolucao.AR1.value, phi=0.8)

    tempo = var(
        "tempo",
        dtype=DType.DISCRETA.value,
        distribuicao=Distribuicao.UNIFORME.value,
        params={"a": 1.0, "b": 60.0},
    )
    tempo.escala = Escala(casas_decimais=0, minimo=1)
    tempo.evolucao = Evolucao(modo=ModoEvolucao.TENDENCIA.value, passo=1.0)

    dobro = VariableConfig(
        nome="dobro",
        dtype=DType.DERIVADA.value,
        expressao="renda * 2",
    )
    flag = VariableConfig(
        nome="flag",
        dtype=DType.DERIVADA.value,
        expressao="(renda > 1000) * 1",
    )

    return DatasetConfig(
        n_linhas=n,
        seed=seed,
        variaveis=[_id(), renda, fixo, qtd, categoria, tempo, dobro, flag],
        relacoes=RelacoesConfig(
            lista=[Relacao("renda", "fixo", 0.6), Relacao("renda", "qtd", 0.35)]
        ),
        painel=PainelConfig(
            habilitado=True,
            coluna_id="id_cliente",
            coluna_tempo="data_mes",
            data_inicio="2025-12-01",
            n_meses=meses,
        ),
    )


# ---------------------------------------------------------------------
# configuracao do painel
# ---------------------------------------------------------------------


def test_meses_do_painel_atravessa_ano():
    meses = meses_do_painel("2025-12-01", 4)
    assert [m.strftime("%Y-%m") for m in meses] == ["2025-12", "2026-01", "2026-02", "2026-03"]
    assert all(m.day == 1 for m in meses)


def test_normalizar_inicio_usa_primeiro_dia():
    assert normalizar_inicio("2026-03-17") == "2026-03-01"


def test_total_linhas_do_painel():
    p = PainelConfig(habilitado=True, n_meses=12)
    assert p.total_linhas(6000) == 72_000
    assert PainelConfig(habilitado=False).total_linhas(6000) == 6000


# ---------------------------------------------------------------------
# geracao
# ---------------------------------------------------------------------


def test_painel_tem_uma_linha_por_cliente_por_mes():
    config = _config(n=300, meses=6)
    res = gerar_dataset(config)
    assert not res.erros, res.erros
    df = res.df

    assert len(df) == 300 * 6
    assert df["id_cliente"].nunique() == 300
    assert df["data_mes"].nunique() == 6
    assert (df.groupby("id_cliente").size() == 6).all()
    assert (df.groupby("id_cliente")["data_mes"].nunique() == 6).all()


def test_colunas_e_ordem_do_painel():
    res = gerar_dataset(_config())
    assert list(res.df.columns) == [
        "id_cliente",
        "data_mes",
        "renda",
        "fixo",
        "qtd",
        "categoria",
        "tempo",
        "dobro",
        "flag",
    ]
    assert res.df["data_mes"].iloc[0] == pd.Timestamp("2025-12-01")


def test_sem_target_por_padrao():
    res = gerar_dataset(_config())
    assert "target" not in res.df.columns
    assert res.config.target.habilitado is False


def test_estado_do_mes_sem_lags_na_base():
    res = gerar_dataset(_config())
    colunas = set(res.df.columns)
    assert not any(c.endswith(("_12m", "_90d", "_lag", "_anterior")) for c in colunas)
    assert "tempo_de_conta" not in colunas


def test_estatica_repete_e_ar1_varia():
    res = gerar_dataset(_config(n=400, meses=8))
    df = res.df
    desvio_fixo = df.groupby("id_cliente")["fixo"].std(ddof=1)
    desvio_renda = df.groupby("id_cliente")["renda"].std(ddof=1)
    assert desvio_fixo.fillna(0).eq(0).all()
    assert (desvio_renda > 0).mean() > 0.9
    desvio_qtd = df.groupby("id_cliente")["qtd"].std(ddof=1)
    assert (desvio_qtd.fillna(0) > 0).mean() > 0.5


def test_tendencia_cresce_um_por_mes():
    res = gerar_dataset(_config(n=50, meses=6))
    df = res.df.sort_values(["id_cliente", "data_mes"])
    primeira = df.groupby("id_cliente")["tempo"].first()
    ultima = df.groupby("id_cliente")["tempo"].last()
    assert ((ultima - primeira) == 5).all()


def test_derivadas_rodam_depois_e_seguem_a_formula():
    res = gerar_dataset(_config(n=300))
    df = res.df
    assert np.allclose(df["dobro"], df["renda"] * 2, atol=1e-6)
    assert set(df["flag"].unique()).issubset({0, 1})
    assert (df.loc[df["renda"] <= 1000, "flag"] == 0).all()


def test_escala_valida_na_derivada():
    config = _config(n=300)
    config.variavel("dobro").escala = Escala(casas_decimais=0, minimo=1000, maximo=1500)
    df = gerar_dataset(config).df
    assert df["dobro"].between(1000, 1500).all()
    assert (df["dobro"] == df["dobro"].round(0)).all()
    assert not np.allclose(df["dobro"], df["renda"] * 2, atol=1e-6)


def test_autocorrelacao_realizada_proxima_do_phi():
    res = gerar_dataset(_config(n=1200, meses=10))
    auto = res.painel["autocorrelacao"]
    assert auto["fixo"] == pytest.approx(1.0, abs=0.01)
    assert 0.6 <= auto["renda"] <= 0.95
    assert auto["qtd"] >= 0.55


def test_info_do_painel():
    res = gerar_dataset(_config(n=120, meses=4))
    info = res.painel
    assert info["entidades"] == 120
    assert info["meses"] == 4
    assert info["linhas"] == 480
    assert info["inicio"] == "2025-12-01"
    assert info["fim"] == "2026-03-01"
    assert set(info["autocorrelacao"]) == {"renda", "fixo", "qtd", "categoria", "tempo"}


def test_comparativo_mede_o_primeiro_mes():
    res = gerar_dataset(_config(n=400))
    assert len(res.comparativo) == 2
    assert res.comparativo["realizado"].notna().all()


def test_painel_reprodutivel_com_mesma_seed():
    a = gerar_dataset(_config(n=150, seed=77))
    b = gerar_dataset(_config(n=150, seed=77))
    assert a.df.equals(b.df)


def test_seed_diferente_muda_o_painel():
    a = gerar_dataset(_config(n=150, seed=1))
    b = gerar_dataset(_config(n=150, seed=2))
    assert not a.df.equals(b.df)


# ---------------------------------------------------------------------
# validacao
# ---------------------------------------------------------------------


def test_painel_exige_coluna_de_id_existente():
    config = _config()
    config.painel.coluna_id = "inexistente"
    erros, _ = config.validar()
    assert any("nao existe" in e for e in erros)


def test_painel_exige_id_do_tipo_texto():
    config = _config()
    config.painel.coluna_id = "renda"
    erros, _ = config.validar()
    assert any("tipo texto" in e for e in erros)


def test_painel_rejeita_coluna_de_tempo_que_colide():
    config = _config()
    config.painel.coluna_tempo = "renda"
    erros, _ = config.validar()
    assert any("colide" in e for e in erros)


def test_painel_limita_meses():
    config = _config()
    config.painel.n_meses = 1
    assert any("2 meses" in e for e in config.validar()[0])
    config.painel.n_meses = 121
    assert any("120 meses" in e for e in config.validar()[0])


def test_painel_rejeita_data_invalida():
    config = _config()
    config.painel.data_inicio = "nao-e-data"
    erros, _ = config.validar()
    assert any("data de inicio invalida" in e for e in erros)


def test_painel_respeita_limite_de_linhas():
    config = _config(n=100_000, meses=12)
    erros, _ = config.validar()
    assert any("excedem o limite" in e for e in erros)


def test_painel_avisa_volume_alto():
    config = _config(n=50_000, meses=11)
    erros, avisos = config.validar()
    assert not any("excedem" in e for e in erros)
    assert any("memoria" in a for a in avisos)


def test_derivada_so_ve_colunas_anteriores():
    config = _config()
    config.variaveis[-1].expressao = "dobro + 1"
    erros, _ = config.validar()
    assert not erros  # dobro vem antes de flag

    config.variaveis.insert(1, VariableConfig(nome="cedo", dtype="derivada", expressao="renda + 1"))
    erros, _ = config.validar()
    assert any("cedo" in e and "expressao" in e for e in erros)


def test_derivada_nao_entra_na_matriz_de_correlacao():
    config = _config()
    config.relacoes.lista = [Relacao("renda", "dobro", 0.5)]
    erros, _ = config.validar()
    assert any("derivada" in e for e in erros)
