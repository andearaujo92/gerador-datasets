"""Diagnostico: pedido vs. realizado, AUC, R2 e resumos."""

from __future__ import annotations

import numpy as np
import pandas as pd

from gerador.engine.copula import spearman
from gerador.schema.relations import Relacao


def _numerizavel(serie: pd.Series) -> pd.Series:
    """Converte para numero preservando a ordem declarada das categorias."""
    if isinstance(serie.dtype, pd.CategoricalDtype):
        return serie.cat.codes.astype(float)
    if serie.dtype.kind in "iufb":
        return pd.to_numeric(serie, errors="coerce")
    if serie.dtype.kind == "M":
        return serie.view("int64").astype(float)
    return serie.astype("category").cat.codes.astype(float)


def pearson(a: pd.Series, b: pd.Series) -> float:
    x = _numerizavel(a)
    y = _numerizavel(b)
    par = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(par) < 3 or par["x"].std() == 0 or par["y"].std() == 0:
        return float("nan")
    return float(par["x"].corr(par["y"]))


def comparar(df: pd.DataFrame, relacoes: list[Relacao]) -> pd.DataFrame:
    """Tabela pedido vs. realizado para cada relacao."""
    linhas = []
    for rel in relacoes:
        if rel.a not in df.columns or rel.b not in df.columns:
            continue
        if df[rel.a].dtype == object or df[rel.b].dtype == object:
            # texto/ID nao participa da matriz: nao faz sentido medir associacao
            continue
        realizado = spearman(df[rel.a].to_numpy(), df[rel.b].to_numpy())
        delta = realizado - rel.forca if not np.isnan(realizado) else float("nan")
        if np.isnan(realizado):
            status = "sem dados"
        elif abs(delta) <= 0.05:
            status = "ok"
        elif abs(delta) <= 0.12:
            status = "atencao"
        else:
            status = "divergente"
        linhas.append(
            {
                "A": rel.a,
                "B": rel.b,
                "medida": rel.medida,
                "pedido": round(rel.forca, 3),
                "realizado": None if np.isnan(realizado) else round(realizado, 3),
                "delta": None if np.isnan(delta) else round(delta, 3),
                "status": status,
                "pearson": None
                if np.isnan(p := pearson(df[rel.a], df[rel.b]))
                else round(p, 3),
            }
        )
    return pd.DataFrame(linhas)


def matriz(df: pd.DataFrame, metodo: str = "spearman") -> pd.DataFrame:
    colunas = [c for c in df.columns if df[c].dtype.kind in "iufb"]
    if len(colunas) < 2:
        return pd.DataFrame()
    if metodo == "spearman":
        return df[colunas].corr(method="spearman")
    return df[colunas].corr(method="pearson")


def resumo(serie: pd.Series) -> dict:
    limpa = serie.dropna()
    dados = {
        "n": int(len(serie)),
        "nulos": int(serie.isna().sum()),
        "pct_nulos": round(float(serie.isna().mean() * 100), 2),
        "unicos": int(limpa.nunique()),
    }
    if serie.dtype.kind in "iufb" and len(limpa) > 0:
        dados.update(
            {
                "media": round(float(limpa.mean()), 4),
                "desvio": round(float(limpa.std()), 4) if len(limpa) > 1 else 0.0,
                "min": float(limpa.min()),
                "max": float(limpa.max()),
            }
        )
    elif serie.dtype.kind == "M":
        dados.update({"min": limpa.min(), "max": limpa.max()})
    else:
        moda = limpa.mode()
        dados["moda"] = moda.iloc[0] if len(moda) else None
    return dados


def histograma(serie: pd.Series, bins: int = 24) -> pd.DataFrame:
    limpa = pd.to_numeric(serie, errors="coerce").dropna()
    if len(limpa) == 0:
        return pd.DataFrame()
    contagens, bordas = np.histogram(limpa.to_numpy(), bins=bins)
    centros = (bordas[:-1] + bordas[1:]) / 2.0
    return pd.DataFrame({"faixa": np.round(centros, 4), "contagem": contagens})


def contagens(serie: pd.Series, limite: int = 15) -> pd.DataFrame:
    valores = serie.dropna().astype(str).value_counts().head(limite)
    return pd.DataFrame({"categoria": valores.index, "contagem": valores.to_numpy()})


def alertas_qualidade(df: pd.DataFrame, config) -> list[str]:
    avisos: list[str] = []
    for v in config.variaveis:
        if v.nome not in df.columns:
            continue
        serie = df[v.nome]
        if v.qualidade.nulos.pct > 0:
            realizado = float(serie.isna().mean() * 100)
            if abs(realizado - v.qualidade.nulos.pct) > 2.0:
                avisos.append(
                    f"'{v.nome}': nulos pedidos {v.qualidade.nulos.pct:.1f}% "
                    f"mas realizados {realizado:.1f}%"
                )
        if v.dtype == "numerica_discreta" and serie.notna().any():
            nao_inteiros = int((serie.dropna() % 1 != 0).sum())
            if nao_inteiros:
                avisos.append(f"'{v.nome}': {nao_inteiros} valores nao inteiros em variavel discreta")
    return avisos
