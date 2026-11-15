"""Exportacao do dataset e geracao do dicionario de dados."""

from __future__ import annotations

import io

import pandas as pd

from gerador.schema.dataset import DatasetConfig
from gerador.schema.painel import meses_do_painel
from gerador.schema.types import DType, ModoEvolucao

EXTENSOES = {"csv": "csv", "parquet": "parquet", "xlsx": "xlsx"}


def exportar(df: pd.DataFrame, formato: str = "csv") -> tuple[bytes, str, str]:
    """Retorna (bytes, nome_sugerido, mime)."""
    if formato == "csv":
        return (
            df.to_csv(index=False).encode("utf-8"),
            "dataset.csv",
            "text/csv",
        )
    if formato == "parquet":
        buffer = io.BytesIO()
        df.to_parquet(buffer, index=False)
        return (
            buffer.getvalue(),
            "dataset.parquet",
            "application/octet-stream",
        )
    if formato == "xlsx":
        buffer = io.BytesIO()
        df.to_excel(buffer, index=False, engine="openpyxl")
        return (
            buffer.getvalue(),
            "dataset.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    raise ValueError(f"formato desconhecido: {formato}")


def _evolucao_md(v) -> str:
    modo = v.evolucao.modo
    if v.tipo in (DType.TEXTO, DType.DERIVADA) or modo == ModoEvolucao.ESTATICA.value:
        return "-" if v.tipo in (DType.TEXTO, DType.DERIVADA) else "estatica"
    if modo == ModoEvolucao.AR1.value:
        return f"ar1 (phi={v.evolucao.phi:g})"
    if modo == ModoEvolucao.TENDENCIA.value:
        return f"tendencia ({v.evolucao.passo:+g}/mes)"
    return modo


def dicionario_md(config: DatasetConfig) -> str:
    painel = config.painel
    linhas = [f"# Dicionario de dados — {len(config.variaveis)} variaveis", ""]
    if painel.habilitado:
        meses = meses_do_painel(painel.data_inicio, painel.n_meses)
        linhas += [
            f"- Clientes (entidades): {config.n_linhas}",
            f"- Meses: {painel.n_meses}",
            f"- Linhas totais: {config.total_linhas}",
            f"- Janela: {meses[0].date()} a {meses[-1].date()}",
            f"- Coluna de tempo: `{painel.coluna_tempo}`",
            "- Uma linha por cliente por mes, com o estado daquele mes: a base nao "
            "traz defasagens nem agregados — isso fica por conta do modelo.",
        ]
    else:
        linhas.append(f"- Linhas: {config.n_linhas}")
    linhas += [
        f"- Seed: {config.seed}",
        f"- Relacoes: {len(config.relacoes.lista)}",
        f"- Target: {config.target.nome if config.target.habilitado else 'nenhum'}",
        "",
        "| Variavel | Tipo | Distribuicao | Parametros | Escala | Evolucao | Nulos | Outliers | Descricao |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for v in config.variaveis:
        if v.tipo is DType.DERIVADA:
            distribuicao = "expressao"
            params = f"`{v.expressao}`"
            escala = "-"
        else:
            distribuicao = v.distribuicao
            params = ", ".join(f"{k}={val}" for k, val in v.params.items() if k != "itens")
            if "itens" in v.params:
                params = f"{len(v.params['itens'])} itens"
            escala = (
                f"x{v.escala.multiplicador:g} +{v.escala.offset:g}"
                if v.escala.multiplicador != 1.0 or v.escala.offset != 0.0
                else v.escala.transformacao
            )
        nulos = f"{v.qualidade.nulos.pct:g}% {v.qualidade.nulos.mecanismo}"
        outliers = (
            f"{v.qualidade.outliers.pct:g}% {v.qualidade.outliers.metodo}"
            if v.qualidade.outliers.metodo != "nenhum"
            else "-"
        )
        linhas.append(
            f"| `{v.nome}` | {v.dtype} | {distribuicao} | {params or '-'} | "
            f"{escala} | {_evolucao_md(v)} | {nulos} | {outliers} | {v.descricao or '-'} |"
        )

    if config.target.habilitado:
        linhas += [
            "",
            f"## Target: `{config.target.nome}`",
            "",
            f"- Tipo: {config.target.tipo}",
            f"- Modo: {config.target.modo}",
        ]
        if config.target.modo == "regra":
            linhas.append(f"- Regra: `{config.target.regra}`")
        else:
            for inf in config.target.influencias:
                seta = "+" if inf.direcao >= 0 else "-"
                linhas.append(f"- {seta} `{inf.variavel}` forca {inf.forca:.2f}")

    linhas += ["", "---", "*Gerado por gerador-dados.*", ""]
    return "\n".join(linhas)
