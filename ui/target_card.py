"""Editor do target (variavel alvo derivada)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from gerador.schema.target import Influencia
from gerador.schema.types import TargetModo, TargetTipo

ROTULOS_TIPO = {
    TargetTipo.BINARIA: "Binaria (0/1)",
    TargetTipo.MULTICLASSE: "Multiclasse",
    TargetTipo.REGRESSAO: "Regressao (continua)",
    TargetTipo.CONTAGEM: "Contagem (Poisson)",
    TargetTipo.ORDINAL: "Ordinal",
}

ROTULOS_MODO = {
    TargetModo.GUIADO: "Guiado (features influentes)",
    TargetModo.REGRA: "Regra (expressao)",
}

ROTULOS_DIRECAO = {1: "Aumenta", -1: "Diminui", 0: "Sem direcao"}

FUNCOES_REGRA = "log, exp, sqrt, abs, clip, sin, cos, round, minimum, maximum, ruido_normal, ruido_uniforme"


def render(config) -> None:
    t = config.target
    st.subheader("Variavel alvo")
    st.caption(
        "O target e **derivado** das features: ele nao entra na matriz de correlacao. "
        "E gerado depois das features e antes dos nulos."
    )

    t.habilitado = st.checkbox("Gerar variavel alvo", value=bool(t.habilitado), key="gd.t.on")
    if not t.habilitado:
        st.caption("Sem target: o arquivo sai apenas com as features.")
        return

    col_nome, col_tipo = st.columns(2)
    with col_nome:
        t.nome = st.text_input("Nome do target", value=t.nome, key="gd.t.nome")

    with col_tipo:
        tipos = list(TargetTipo)
        try:
            atual = TargetTipo(t.tipo)
        except ValueError:
            atual = TargetTipo.BINARIA
        escolhido = st.selectbox(
            "Tipo",
            tipos,
            index=tipos.index(atual),
            format_func=lambda k: ROTULOS_TIPO[k],
            key="gd.t.tipo",
        )
        if escolhido.value != t.tipo:
            t.tipo = escolhido.value
            t.resetar_params()

    modos = list(TargetModo)
    try:
        modo_atual = TargetModo(t.modo)
    except ValueError:
        modo_atual = TargetModo.GUIADO
    escolhido_modo = st.radio(
        "Modo",
        modos,
        index=modos.index(modo_atual),
        format_func=lambda k: ROTULOS_MODO[k],
        horizontal=True,
        key="gd.t.modo",
    )
    t.modo = escolhido_modo.value

    if escolhido_modo is TargetModo.REGRA:
        _render_regra(config, t)
    else:
        _render_guiado(config, t)

    st.divider()
    _render_params(config, t)
    _render_avisos(config, t)


def _render_guiado(config, t) -> None:
    nomes = config.nomes
    if not nomes:
        st.info("Crie variaveis antes de definir influencias.")
        return

    st.markdown("**Features que influenciam o alvo**")
    st.caption("Sem nenhuma influencia o alvo vira ruido puro (AUC ~0.5).")

    remover = None
    for inf in list(t.influencias):
        c1, c2, c3, c4 = st.columns([3, 2, 4, 1])
        with c1:
            indice = nomes.index(inf.variavel) if inf.variavel in nomes else 0
            inf.variavel = st.selectbox(
                "Feature", nomes, index=indice, key=f"{inf.uid}.inf_var"
            )
        with c2:
            opcoes = list(ROTULOS_DIRECAO)
            indice_d = opcoes.index(inf.direcao) if inf.direcao in opcoes else 0
            inf.direcao = int(
                st.selectbox(
                    "Direcao",
                    opcoes,
                    index=indice_d,
                    format_func=lambda k: ROTULOS_DIRECAO[k],
                    key=f"{inf.uid}.inf_dir",
                )
            )
        with c3:
            inf.forca = float(
                st.slider(
                    f"Forca de {inf.variavel}",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(inf.forca),
                    step=0.05,
                    key=f"{inf.uid}.inf_forca",
                )
            )
        with c4:
            st.write("")
            st.write("")
            if st.button("Remover", key=f"{inf.uid}.inf_remover"):
                remover = inf.uid

    if remover is not None:
        t.influencias = [i for i in t.influencias if i.uid != remover]
        st.rerun()

    if st.button("Adicionar feature influente", key="gd.t.add_inf"):
        usada = 0
        for nome in nomes:
            if all(i.variavel != nome for i in t.influencias):
                usada = nome
                break
        t.influencias.append(Influencia(usada, 1, 0.5))
        st.rerun()


def _render_regra(config, t) -> None:
    st.markdown("**Regra do target**")
    st.caption(
        "Expressao booleana/numérica em Python restrito. Colunas: "
        + (", ".join(f"`{n}`" for n in config.nomes) if config.nomes else "(nenhuma)")
        + f". Funcoes: {FUNCOES_REGRA}."
    )
    t.regra = st.text_area(
        "Expressao",
        value=t.regra or "",
        placeholder="ex.: (renda > 3000) and (compras_12m <= 2)",
        key="gd.t.regra",
    )

    from gerador.expressions import validar_expressao

    if t.regra:
        erro = validar_expressao(t.regra, config.nomes)
        if erro:
            st.error(f"Expressao invalida: {erro}")
        else:
            st.success("Expressao valida.")


def _render_params(config, t) -> None:
    tipo = t.tipo
    chave = f"gd.t.{tipo}"

    if tipo == TargetTipo.BINARIA.value:
        t.params["prevalencia_base"] = float(
            st.slider(
                "Prevalencia desejada (classe 1)",
                min_value=0.01,
                max_value=0.99,
                value=float(t.params.get("prevalencia_base", 0.3)),
                step=0.01,
                key=f"{chave}.prev",
            )
        )
        t.params["ruido"] = float(
            st.slider(
                "Ruido nos rotulos (fracao invertida)",
                min_value=0.0,
                max_value=1.0,
                value=float(t.params.get("ruido", 0.1)),
                step=0.01,
                key=f"{chave}.ruido",
                help=(
                    "Cada rotulo tem essa chance de ser invertido. E destrutivo: "
                    "0,10 ja reduz bastante o AUC; 0,50 zera o sinal."
                ),
            )
        )
        t.params["forcar_prevalencia"] = st.checkbox(
            "Forcar prevalencia exata",
            value=bool(t.params.get("forcar_prevalencia", True)),
            key=f"{chave}.forcar",
            help="Garante exatamente a contagem pedida, em vez de apenas aproximar.",
        )

    elif tipo == TargetTipo.REGRESSAO.value:
        t.params["media_base"] = float(
            st.number_input(
                "Media base",
                value=float(t.params.get("media_base", 0.0)),
                step=1.0,
                key=f"{chave}.media",
            )
        )
        t.params["ruido"] = float(
            st.slider(
                "Ruido (0 = determinístico, 1 = quase aleatorio)",
                min_value=0.0,
                max_value=1.0,
                value=float(t.params.get("ruido", 0.4)),
                step=0.05,
                key=f"{chave}.ruido",
            )
        )

    elif tipo == TargetTipo.CONTAGEM.value:
        t.params["lam_base"] = float(
            st.number_input(
                "Media esperada (lambda)",
                value=float(t.params.get("lam_base", 5.0)),
                min_value=0.1,
                step=0.5,
                key=f"{chave}.lam",
            )
        )
        t.params["ruido"] = float(
            st.slider(
                "Ruido",
                min_value=0.0,
                max_value=1.0,
                value=float(t.params.get("ruido", 0.3)),
                step=0.05,
                key=f"{chave}.ruido",
            )
        )

    elif tipo == TargetTipo.MULTICLASSE.value:
        classes = t.params.get("classes") or [
            {"rotulo": "A", "proporcao": 0.5},
            {"rotulo": "B", "proporcao": 0.5},
        ]
        base = pd.DataFrame(classes)
        editado = st.data_editor(
            base,
            num_rows="dynamic",
            key=f"{chave}.classes",
            column_config={
                "rotulo": st.column_config.TextColumn("rotulo", required=True),
                "proporcao": st.column_config.NumberColumn(
                    "proporcao", min_value=0.01, max_value=1.0, step=0.05, default=0.5
                ),
            },
        )
        novas: list[dict] = []
        for _, linha in editado.iterrows():
            rotulo = linha.get("rotulo")
            if rotulo is None or (isinstance(rotulo, float) and np.isnan(rotulo)):
                continue
            if not str(rotulo).strip():
                continue
            proporcao = linha.get("proporcao", 0.5)
            if proporcao is None or (isinstance(proporcao, float) and np.isnan(proporcao)):
                proporcao = 0.5
            novas.append({"rotulo": str(rotulo).strip(), "proporcao": float(proporcao)})
        if novas:
            t.params["classes"] = novas

        t.params["ruido"] = float(
            st.slider(
                "Ruido",
                min_value=0.0,
                max_value=1.0,
                value=float(t.params.get("ruido", 0.2)),
                step=0.05,
                key=f"{chave}.ruido",
            )
        )

    elif tipo == TargetTipo.ORDINAL.value:
        t.params["rotulos"] = st.text_input(
            "Rotulos, em ordem crescente",
            value=str(t.params.get("rotulos", "baixo, medio, alto")),
            key=f"{chave}.rotulos",
        )
        t.params["proporcoes"] = st.text_input(
            "Proporcoes (mesma quantidade de rotulos)",
            value=str(t.params.get("proporcoes", "")),
            placeholder="0.5, 0.3, 0.2",
            key=f"{chave}.proporcoes",
        )
        rotulos = [r.strip() for r in str(t.params["rotulos"]).split(",") if r.strip()]
        if t.params["proporcoes"]:
            valores = [v.strip() for v in str(t.params["proporcoes"]).split(",") if v.strip()]
            if len(valores) != len(rotulos):
                st.warning(
                    f"{len(valores)} proporcoes para {len(rotulos)} rotulos; "
                    "vai usar proporcoes iguais."
                )
        t.params["ruido"] = float(
            st.slider(
                "Ruido",
                min_value=0.0,
                max_value=1.0,
                value=float(t.params.get("ruido", 0.2)),
                step=0.05,
                key=f"{chave}.ruido",
            )
        )


def _render_avisos(config, t) -> None:
    if not t.habilitado:
        return

    if t.nome in config.nomes:
        st.error(f"O nome '{t.nome}' colide com uma variavel existente.")

    if t.modo == TargetModo.GUIADO.value:
        for inf in t.influencias:
            var = config.variavel(inf.variavel)
            if var is None:
                st.warning(f"Influencia aponta para '{inf.variavel}', que nao existe mais.")
            elif inf.forca >= 0.5 and var.qualidade.nulos.pct >= 30:
                st.warning(
                    f"'{inf.variavel}' tem {var.qualidade.nulos.pct:g}% de nulos e forca "
                    f"{inf.forca:.2f}: o sinal aprendivel vai degradar."
                )
