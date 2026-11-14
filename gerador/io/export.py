"""Exportacao do dataset e geracao do dicionario de dados."""

from __future__ import annotations

import io

import pandas as pd

from gerador.schema.dataset import DatasetConfig

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


def dicionario_md(config: DatasetConfig) -> str:
    linhas = [
        f"# Dicionario de dados — {len(config.variaveis)} variaveis",
        "",
        f"- Linhas: {config.n_linhas}",
        f"- Seed: {config.seed}",
        f"- Relacoes: {len(config.relacoes.lista)}",
        f"- Target: {config.target.nome if config.target.habilitado else 'nenhum'}",
        "",
        "| Variavel | Tipo | Distribuicao | Parametros | Escala | Nulos | Outliers | Descricao |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for v in config.variaveis:
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
            f"| `{v.nome}` | {v.dtype} | {v.distribuicao} | {params or '-'} | "
            f"{escala} | {nulos} | {outliers} | {v.descricao or '-'} |"
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
