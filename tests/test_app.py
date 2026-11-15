"""Smoke test da aplicacao Streamlit (renderizacao e fluxo de geracao)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def _abrir() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    return at


def test_app_renderiza_sem_excecao():
    at = _abrir()
    assert not at.exception
    assert len(at.tabs) == 5
    assert any(t.value == "Gerador de datasets sinteticos" for t in at.title)


def test_botao_gerar_produz_dataset_com_target_por_ultimo():
    at = _abrir()
    at.button(key="gd.gerar").click().run()
    assert not at.exception

    res = at.session_state["gd_resultado"]
    assert res.ok, res.erros
    assert list(res.df.columns)[-1] == "churn"
    assert len(res.df) == at.session_state["gd_config"].n_linhas
    assert res.df["id_cliente"].is_unique


def test_trocar_tipo_de_variavel_nao_quebra_a_cascata():
    at = _abrir()
    config = at.session_state["gd_config"]
    var_id = config.variaveis[0].id

    at.selectbox(key=f"{var_id}.dtype").set_value("numerica_continua").run()
    assert not at.exception

    config = at.session_state["gd_config"]
    assert config.variaveis[0].dtype == "numerica_continua"
    # distribuicao precisou ser reescolhida para um tipo valido
    assert config.variaveis[0].distribuicao == "normal"

    at.button(key="gd.gerar").click().run()
    assert not at.exception
    assert at.session_state["gd_resultado"].ok


def test_config_invalida_aparece_como_erro_e_nao_gera():
    at = _abrir()
    at.text_input(key="gd.t.nome").set_value("renda").run()
    assert not at.exception

    at.button(key="gd.gerar").click().run()
    res = at.session_state["gd_resultado"]
    assert res.df is None
    assert any("colide" in e for e in res.erros)


def test_adicionar_variavel_e_relacao():
    at = _abrir()
    antes = len(at.session_state["gd_config"].variaveis)
    at.button(key="gd.add_var").click().run()
    assert not at.exception
    assert len(at.session_state["gd_config"].variaveis) == antes + 1

    relacoes_antes = len(at.session_state["gd_config"].relacoes.lista)
    at.button(key="gd.add_rel").click().run()
    assert not at.exception
    assert len(at.session_state["gd_config"].relacoes.lista) == relacoes_antes + 1


def test_carregar_preset_de_credito_mensal():
    at = _abrir()
    at.button(key="gd.carregar_preset").click().run()
    assert not at.exception

    config = at.session_state["gd_config"]
    assert config.painel.habilitado
    assert config.n_linhas == 6000
    assert len(config.variaveis) == 22
    assert config.target.habilitado is False
    assert any(v.dtype == "derivada" for v in config.variaveis)
    # controles do painel aparecem na barra lateral
    assert at.number_input(key="gd.painel_meses").value == 12
    assert at.checkbox(key="gd.painel_on").value is True


def test_preset_gera_painel_sem_target():
    at = _abrir()
    at.button(key="gd.carregar_preset").click().run()
    at.number_input(key="gd.n_linhas").set_value(60).run()
    at.number_input(key="gd.painel_meses").set_value(3).run()
    at.button(key="gd.gerar").click().run()
    assert not at.exception

    res = at.session_state["gd_resultado"]
    assert res.ok, res.erros
    assert len(res.df) == 60 * 3
    assert "data_mes" in res.df.columns
    assert "id_cliente" in res.df.columns
    assert all(c not in res.df.columns for c in ("churn", "target"))
    assert res.df["id_cliente"].nunique() == 60
