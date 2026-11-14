"""Barra lateral: configuracao global, schema e geracao."""

from __future__ import annotations

import numpy as np
import streamlit as st

from gerador.engine.pipeline import gerar_dataset
from gerador.io.export import dicionario_md, exportar
from gerador.io.profile import carregar_schema
from ui.state import (
    definir_config,
    guardar_resultado,
    hash_config,
    json_atual,
    resultado,
    resultado_desatualizado,
)

CHAVE_ASSINATURA = "gd_assinatura_arquivo"


@st.cache_data(show_spinner=False, max_entries=6)
def _gerar(schema_json: str):
    """Gera a partir do JSON do schema (cache por conteudo)."""
    return gerar_dataset(carregar_schema(schema_json))


def _sortear_seed() -> None:
    st.session_state["gd.seed"] = int(np.random.default_rng().integers(0, 2**31 - 1))


def render(config) -> None:
    with st.sidebar:
        st.title("Gerador de dados")
        st.caption("Datasets sinteticos para machine learning")

        st.markdown("#### Configuracao global")
        config.n_linhas = int(
            st.number_input(
                "Linhas",
                min_value=1,
                max_value=1_000_000,
                value=int(config.n_linhas),
                step=100,
                key="gd.n_linhas",
            )
        )

        col_seed, col_botao = st.columns([3, 1])
        with col_seed:
            config.seed = int(
                st.number_input(
                    "Seed",
                    min_value=0,
                    max_value=2**31 - 1,
                    value=int(config.seed),
                    step=1,
                    key="gd.seed",
                )
            )
        with col_botao:
            st.write("")
            st.write("")
            st.button("Sortear", on_click=_sortear_seed, key="gd.sortear_seed")

        formatos = ["csv", "parquet", "xlsx"]
        config.formato_export = st.selectbox(
            "Formato",
            formatos,
            index=formatos.index(config.formato_export)
            if config.formato_export in formatos
            else 0,
            key="gd.formato",
        )
        config.incluir_dicionario = st.checkbox(
            "Incluir dicionario de dados",
            value=bool(config.incluir_dicionario),
            key="gd.incluir_dicionario",
        )

        st.divider()
        st.markdown("#### Schema")
        st.download_button(
            "Salvar schema (JSON)",
            data=json_atual(config),
            file_name="schema-gerador.json",
            mime="application/json",
            key="gd.salvar_json",
        )
        arquivo = st.file_uploader(
            "Carregar schema", type=["json"], key="gd.carregar_json"
        )
        if arquivo is not None:
            assinatura = f"{arquivo.name}:{arquivo.size}"
            if st.session_state.get(CHAVE_ASSINATURA) != assinatura:
                try:
                    definir_config(carregar_schema(arquivo.getvalue()))
                except Exception as exc:  # schema malformado
                    st.error(f"Schema invalido: {exc}")
                else:
                    st.session_state[CHAVE_ASSINATURA] = assinatura
                    st.rerun()

        st.divider()
        if st.button("Gerar dataset", type="primary", key="gd.gerar"):
            _executar(config)

        _render_resultado(config)


def _executar(config) -> None:
    schema = json_atual(config)
    chave = hash_config(config)
    with st.spinner("Gerando dataset..."):
        try:
            res = _gerar(schema)
        except Exception as exc:  # falha inesperada do motor
            st.error(f"Falha na geracao: {exc}")
            return
    guardar_resultado(res, chave)
    st.rerun()


def _render_resultado(config) -> None:
    res = resultado()
    if res is None:
        return

    if res.erros:
        st.error(f"{len(res.erros)} erro(s) — detalhes na aba Diagnostico.")
    else:
        dados, nome, mime = exportar(res.df, config.formato_export)
        st.download_button(
            f"Baixar {nome}",
            data=dados,
            file_name=nome,
            mime=mime,
            key="gd.baixar_dataset",
        )
        if config.incluir_dicionario:
            st.download_button(
                "Baixar dicionario.md",
                data=dicionario_md(res.config).encode("utf-8"),
                file_name="dicionario.md",
                mime="text/markdown",
                key="gd.baixar_dicionario",
            )
        linhas = f"{len(res.df):,}".replace(",", ".")
        st.caption(f"{linhas} linhas · {res.df.shape[1]} colunas · {res.tempo_ms:.0f} ms")

    if resultado_desatualizado(config):
        st.info("A configuracao mudou desde a ultima geracao.")
