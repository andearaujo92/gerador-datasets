"""Aplicacao Streamlit: gerador de datasets sinteticos para ML."""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="Gerador de dados sinteticos",
    page_icon="🧪",
    layout="wide",
)

from ui import diagnostics, relations, sidebar, target_card, variable_card  # noqa: E402
from ui.state import obter_config, resultado  # noqa: E402


def main() -> None:
    config = obter_config()

    st.title("Gerador de datasets sinteticos")
    st.caption(
        "Defina cada variavel, as relacoes entre elas e o alvo derivado. "
        "A exportacao sai pronta para treinar modelos — com dicionario de dados."
    )

    aba_variaveis, aba_relacoes, aba_alvo, aba_diagnostico, aba_previa = st.tabs(
        ["Variaveis", "Relacoes", "Alvo", "Diagnostico", "Previa"]
    )

    with aba_variaveis:
        variable_card.render_todas(config)

    with aba_relacoes:
        relations.render(config)

    with aba_alvo:
        target_card.render(config)

    res = resultado()

    with aba_diagnostico:
        diagnostics.render_diagnostico(config, res)

    with aba_previa:
        diagnostics.render_previa(config, res)

    # Sidebar por ultimo de proposito: o botao "Gerar" enxerga o estado final
    # de todos os widgets renderizados acima.
    sidebar.render(config)


main()
