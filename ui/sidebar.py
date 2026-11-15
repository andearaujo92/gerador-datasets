"""Barra lateral: configuracao global, schema e geracao."""

from __future__ import annotations

from datetime import date

import numpy as np
import streamlit as st

from gerador.engine.pipeline import gerar_dataset
from gerador.io.export import dicionario_md, exportar
from gerador.io.profile import carregar_schema
from gerador.presets import listar_presets, obter_preset
from gerador.schema.painel import AVISO_LINHAS, LIMITE_LINHAS, normalizar_inicio
from gerador.schema.types import DType
from ui.state import (
    definir_config,
    guardar_resultado,
    hash_config,
    json_atual,
    resultado,
    resultado_desatualizado,
)

CHAVE_ASSINATURA = "gd_assinatura_arquivo"


def _numero_br(valor: int) -> str:
    return f"{int(valor):,}".replace(",", ".")


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

        st.markdown("#### Modelos prontos")
        presets = listar_presets()
        chave_preset = st.selectbox(
            "Base",
            list(presets),
            format_func=lambda k: presets[k],
            key="gd.preset_escolha",
        )
        if st.button("Carregar modelo", key="gd.carregar_preset"):
            definir_config(obter_preset(chave_preset))
            st.rerun()
        st.caption("Carregar substitui o schema atual. Salve o seu antes, se precisar.")

        st.divider()
        st.markdown("#### Configuracao global")
        config.painel.habilitado = st.checkbox(
            "Painel temporal (1 linha por cliente por mes)",
            value=bool(config.painel.habilitado),
            key="gd.painel_on",
            help=(
                "Transforma a base em um painel balanceado: cada cliente aparece "
                "em todos os meses, com o estado daquele mes (sem lags nem somas moveis)."
            ),
        )

        config.n_linhas = int(
            st.number_input(
                "Clientes" if config.painel.habilitado else "Linhas",
                min_value=1,
                max_value=1_000_000,
                value=int(config.n_linhas),
                step=100,
                key="gd.n_linhas",
            )
        )

        if config.painel.habilitado:
            _render_painel(config)

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


def _render_painel(config) -> None:
    """Controles do painel: janela, chave e volume total."""
    p = config.painel
    col_meses, col_inicio = st.columns(2)
    with col_meses:
        p.n_meses = int(
            st.number_input(
                "Meses",
                min_value=2,
                max_value=120,
                value=int(p.n_meses),
                step=1,
                key="gd.painel_meses",
            )
        )
    with col_inicio:
        try:
            inicial = date.fromisoformat(normalizar_inicio(p.data_inicio))
        except (ValueError, TypeError):
            inicial = date(2025, 12, 1)
        escolhida = st.date_input(
            "Primeiro mes",
            value=inicial,
            min_value=date(2000, 1, 1),
            max_value=date(2100, 1, 1),
            key="gd.painel_inicio",
        )
        p.data_inicio = escolhida.replace(day=1).isoformat()

    textos = [v.nome for v in config.variaveis if v.dtype == DType.TEXTO.value]
    if textos:
        p.coluna_id = st.selectbox(
            "Coluna do cliente",
            textos,
            index=textos.index(p.coluna_id) if p.coluna_id in textos else 0,
            key="gd.painel_id",
        )
    else:
        st.caption("Adicione uma variavel de texto (ID) para ser a chave do painel.")

    p.coluna_tempo = st.text_input(
        "Coluna de tempo",
        value=p.coluna_tempo,
        key="gd.painel_tempo",
    )

    total = int(config.n_linhas) * int(p.n_meses)
    st.caption(
        f"{_numero_br(config.n_linhas)} clientes x {p.n_meses} meses = "
        f"{_numero_br(total)} linhas"
    )
    if total > LIMITE_LINHAS:
        st.error("Acima do limite de 1.000.000 de linhas — reduza clientes ou meses.")
    elif total > AVISO_LINHAS:
        st.warning("Base grande: a geracao pode demorar alguns segundos.")


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
        linhas = _numero_br(len(res.df))
        if res.painel:
            st.caption(
                f"{linhas} linhas · {_numero_br(res.painel['entidades'])} clientes · "
                f"{res.painel['meses']} meses · {res.df.shape[1]} colunas · "
                f"{res.tempo_ms:.0f} ms"
            )
        else:
            st.caption(f"{linhas} linhas · {res.df.shape[1]} colunas · {res.tempo_ms:.0f} ms")

    if resultado_desatualizado(config):
        st.info("A configuracao mudou desde a ultima geracao.")
