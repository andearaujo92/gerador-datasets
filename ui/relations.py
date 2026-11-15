"""Editor de correlacoes entre variaveis."""

from __future__ import annotations

import streamlit as st

from gerador.schema.relations import Relacao
from gerador.schema.types import DType


def _correlacionaveis(config) -> list[str]:
    """Variaveis que participam da copula (texto/ID e derivadas ficam de fora)."""
    return [
        v.nome
        for v in config.variaveis
        if v.dtype not in (DType.TEXTO.value, DType.DERIVADA.value)
    ]


def render(config) -> None:
    st.subheader("Relacoes entre variaveis")
    st.caption(
        "A forca e declarada em **Spearman**: vale para qualquer tipo, porque mede "
        "associacao monotonica. A matriz e convertida para o espaco latente e "
        "validada (autovalores) antes de gerar."
    )

    nomes = _correlacionaveis(config)
    if len(config.nomes) > len(nomes):
        st.caption(
            "Variaveis de texto (ID) e derivadas ficam fora: IDs sao chaves e a "
            "correlacao das derivadas vem da formula."
        )
    if len(nomes) < 2:
        st.info("Adicione ao menos duas variaveis numericas/categoricas para correlacionar.")
        return

    lista = config.relacoes.lista
    if not lista:
        st.caption("Nenhuma relacao definida: as variaveis saem independentes.")

    remover = None
    for rel in list(lista):
        col_a, col_b, col_forca, col_remover = st.columns([3, 3, 5, 1])

        with col_a:
            indice_a = nomes.index(rel.a) if rel.a in nomes else 0
            rel.a = st.selectbox(
                "Variavel A", nomes, index=indice_a, key=f"{rel.uid}.a"
            )

        opcoes_b = [n for n in nomes if n != rel.a]
        with col_b:
            indice_b = opcoes_b.index(rel.b) if rel.b in opcoes_b else 0
            rel.b = st.selectbox(
                "Variavel B", opcoes_b, index=indice_b, key=f"{rel.uid}.b"
            )

        with col_forca:
            rel.forca = float(
                st.slider(
                    f"Forca ({rel.a} x {rel.b})",
                    min_value=-1.0,
                    max_value=1.0,
                    value=float(rel.forca),
                    step=0.05,
                    key=f"{rel.uid}.forca",
                )
            )

        with col_remover:
            st.write("")
            st.write("")
            if st.button("Remover", key=f"{rel.uid}.remover"):
                remover = rel.uid

    if remover is not None:
        config.relacoes.lista = [r for r in lista if r.uid != remover]
        st.rerun()

    acoes = st.columns([1, 1, 3])
    with acoes[0]:
        if st.button("Adicionar relacao", key="gd.add_rel"):
            config.relacoes.lista.append(Relacao(nomes[0], nomes[1], 0.5))
            st.rerun()
    with acoes[1]:
        if lista and st.button("Limpar todas", key="gd.limpar_rel"):
            config.relacoes.lista = []
            st.rerun()

    if lista:
        st.divider()
        esquerda, direita = st.columns([1, 3])
        with esquerda:
            config.relacoes.calibrar = st.checkbox(
                "Calibrar iterativamente",
                value=bool(config.relacoes.calibrar),
                key="gd.calibrar",
                help=(
                    "Variaveis discretas e categoricas atenuam a correlacao realizada "
                    "(empates). A calibracao ajusta a matriz para compensar e repete "
                    "ate a tolerancia."
                ),
            )
            config.relacoes.completar_psd = st.checkbox(
                "Completar matriz automaticamente",
                value=bool(config.relacoes.completar_psd),
                key="gd.completar_psd",
                help=(
                    "Quando as correlacoes informadas nao fecham sozinhas, os pares "
                    "nao declarados sao preenchidos com o valor que torna a matriz "
                    "valida, em vez de recusar a geracao."
                ),
            )
        with direita:
            if config.relacoes.calibrar:
                config.relacoes.tolerancia = float(
                    st.slider(
                        "Tolerancia",
                        min_value=0.01,
                        max_value=0.10,
                        value=float(config.relacoes.tolerancia),
                        step=0.01,
                        key="gd.tolerancia",
                    )
                )
                config.relacoes.max_iter = int(
                    st.number_input(
                        "Maximo de iteracoes",
                        min_value=1,
                        max_value=100,
                        value=int(config.relacoes.max_iter),
                        step=1,
                        key="gd.max_iter",
                    )
                )
            else:
                st.caption(
                    "Sem calibracao, variaveis categoricas tendem a sair com "
                    "correlacao menor que a pedida. O diagnostico mostra pedido vs. realizado."
                )
