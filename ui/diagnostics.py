"""Diagnostico da geracao e pre-visualizacao do dataset."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import streamlit as st

from gerador.engine import diagnostics as diag

# ---------------------------------------------------------------------
# heatmap sem dependencias (nada de matplotlib/plotly)
# ---------------------------------------------------------------------


def _celula(valor: float) -> tuple[str, str]:
    if valor is None or (isinstance(valor, float) and math.isnan(valor)):
        return "rgba(0,0,0,0.04)", "#666"
    v = max(-1.0, min(1.0, float(valor)))
    intensidade = abs(v)
    base = (33, 102, 172) if v >= 0 else (178, 24, 43)
    r, g, b = base
    cor_texto = "#fff" if intensidade > 0.55 else "#1a1a1a"
    return f"rgba({r},{g},{b},{0.08 + 0.85 * intensidade:.2f})", cor_texto


def _heatmap_html(mat: pd.DataFrame) -> str:
    if mat.empty:
        return ""
    colunas = list(mat.columns)
    fragmentos = [
        "<div style='overflow-x:auto'><table style='border-collapse:collapse;"
        "font-size:12px;font-family:ui-sans-serif,system-ui'>",
        "<thead><tr><th></th>"
        + "".join(
            f"<th style='padding:4px 8px;text-align:left;font-weight:600'>{c}</th>"
            for c in colunas
        )
        + "</tr></thead><tbody>",
    ]
    for i, linha in enumerate(mat.index):
        fragmentos.append(
            f"<tr><th style='padding:4px 8px;text-align:left;font-weight:600'>{linha}</th>"
        )
        for j, coluna in enumerate(colunas):
            valor = mat.iloc[i, j]
            fundo, texto = _celula(valor)
            conteudo = "—" if valor is None or (isinstance(valor, float) and math.isnan(valor)) else f"{float(valor):.2f}"
            fragmentos.append(
                f"<td style='padding:4px 8px;text-align:right;background:{fundo};"
                f"color:{texto};border:1px solid rgba(0,0,0,0.06)'>{conteudo}</td>"
            )
        fragmentos.append("</tr>")
    fragmentos.append("</tbody></table></div>")
    return "".join(fragmentos)


# ---------------------------------------------------------------------
# diagnostico
# ---------------------------------------------------------------------


def render_diagnostico(config, res) -> None:
    st.subheader("Diagnostico")

    if res is None:
        st.info("Gere o dataset para ver o diagnostico.")
        return

    if res.erros:
        st.error("Corrija os problemas abaixo antes de gerar:")
        for erro in res.erros:
            st.write(f"- {erro}")
        return

    df = res.df
    metricas = st.columns(4)
    metricas[0].metric("Linhas", f"{len(df):,}".replace(",", "."))
    metricas[1].metric("Colunas", df.shape[1])
    metricas[2].metric("Tempo", f"{res.tempo_ms:.0f} ms")
    metricas[3].metric("Nulos", f"{df.isna().mean().mean() * 100:.1f}%")

    _render_target(res, df)
    _render_comparativo(res)
    _render_correlacoes(df)
    _render_alertas(config, res, df)
    _render_resumos(df)


def _render_target(res, df) -> None:
    info = res.target_info or {}
    if not info:
        return
    st.markdown("### Alvo")
    colunas = st.columns(4)
    tipo = info.get("tipo", "")
    colunas[0].metric("Tipo", str(tipo))

    if tipo == "binaria":
        colunas[1].metric("Prevalencia", f"{info.get('prevalencia', float('nan')):.1%}")
        colunas[2].metric("Prevalencia pedida", f"{info.get('prevalencia_pedida', float('nan')):.1%}")
        auc = info.get("auc", float("nan"))
        colunas[3].metric("AUC", "—" if auc != auc else f"{auc:.3f}")
        st.caption(
            "AUC mede o quanto as features realmente explicam o alvo. "
            "Perto de 0,5 = sinal fraco; acima de 0,8 = sinal forte."
        )
    elif tipo == "regressao":
        colunas[1].metric("Media", f"{info.get('media', float('nan')):.3f}")
        r2 = info.get("r2", float("nan"))
        colunas[2].metric("R2 linear", "—" if r2 != r2 else f"{r2:.3f}")
        st.caption("R2 mede a fracao da variancia do alvo explicada pelos sinais configurados.")
    elif tipo == "contagem":
        colunas[1].metric("Media realizada", f"{info.get('media', float('nan')):.2f}")
        st.caption("A media realizada deve ficar proxima do lambda base configurado.")
    elif tipo in ("multiclasse", "ordinal"):
        rotulos = info.get("rotulos") or []
        pedidas = info.get("proporcoes") or []
        realizadas = info.get("proporcoes_realizadas") or []
        tabela = pd.DataFrame(
            {
                "classe": rotulos,
                "pedida": [f"{p:.1%}" for p in pedidas],
                "realizada": [f"{p:.1%}" for p in realizadas],
            }
        )
        st.dataframe(tabela, hide_index=True)
    elif tipo == "binaria" or info.get("features"):
        st.caption("Features usadas: " + ", ".join(info.get("features", [])))


def _render_comparativo(res) -> None:
    tabela = res.comparativo
    if tabela is None or tabela.empty:
        return
    st.markdown("### Correlacoes: pedido vs. realizado")
    st.dataframe(tabela, hide_index=True)

    divergentes = tabela[tabela["status"] == "divergente"]
    if not divergentes.empty:
        st.warning(
            f"{len(divergentes)} relacao(oes) divergiram mais de 0,12. Causas comuns: "
            "muitos empates (categoricas), clipping de escala ou nulos altos. "
            "Ative a calibracao na aba Relacoes."
        )
    elif (tabela["status"] == "atencao").any():
        st.info(
            "Algumas relacoes ficaram na faixa de atencao (0,05 a 0,12). "
            "Costuma ser atenuacao por empates em variaveis discretas."
        )


def _render_correlacoes(df: pd.DataFrame) -> None:
    matriz = diag.matriz(df, "spearman")
    if matriz.empty:
        return
    st.markdown("### Matriz de Spearman realizada")
    st.markdown(_heatmap_html(matriz), unsafe_allow_html=True)


def _render_alertas(config, res, df: pd.DataFrame) -> None:
    alertas = diag.alertas_qualidade(df, config)
    if alertas:
        st.markdown("### Alertas de qualidade")
        for alerta in alertas:
            st.warning(alerta)


def _render_resumos(df: pd.DataFrame) -> None:
    st.markdown("### Resumo por variavel")
    for coluna in df.columns:
        serie = df[coluna]
        resumo = diag.resumo(serie)
        partes = [f"**{coluna}**", f"nulos {resumo['pct_nulos']:.1f}%", f"unicos {resumo['unicos']}"]
        if "media" in resumo:
            partes.append(f"media {resumo['media']}")
            partes.append(f"desvio {resumo['desvio']}")
            partes.append(f"min {resumo['min']} / max {resumo['max']}")
        elif "moda" in resumo:
            partes.append(f"moda {resumo['moda']}")
        elif "min" in resumo:
            partes.append(f"{resumo['min']} -> {resumo['max']}")
        st.caption("  ·  ".join(partes))


# ---------------------------------------------------------------------
# previa
# ---------------------------------------------------------------------


def render_previa(config, res) -> None:
    st.subheader("Previa")
    if res is None or res.df is None:
        st.info("Gere o dataset para ver a previa.")
        return
    if res.erros:
        st.error("A geracao nao concluiu por causa dos erros na configuracao.")
        return

    df = res.df
    st.caption(f"{len(df):,} linhas · {df.shape[1]} colunas · mostrando as 100 primeiras".replace(",", "."))
    st.dataframe(df.head(100), hide_index=True)

    st.markdown("### Distribuicao")
    coluna = st.selectbox("Variavel", list(df.columns), key="gd.previa.coluna")
    serie = df[coluna]

    if serie.dtype.kind in "iufb" and serie.notna().any():
        histograma = diag.histograma(serie, bins=24)
        if not histograma.empty:
            st.bar_chart(histograma.set_index("faixa")["contagem"])
    elif serie.dtype.kind == "M" and serie.notna().any():
        por_mes = serie.dt.to_period("M").value_counts().sort_index()
        st.bar_chart(pd.DataFrame({"contagem": por_mes.to_numpy()}, index=[str(i) for i in por_mes.index]))
    else:
        contagem = diag.contagens(serie, limite=20)
        if not contagem.empty:
            st.bar_chart(contagem.set_index("categoria")["contagem"])
        else:
            st.caption("Sem valores para plotar.")

    st.markdown("### Numeros")
    st.json(diag.resumo(serie), expanded=False)
