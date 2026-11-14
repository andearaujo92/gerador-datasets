"""Valores ausentes: MCAR e MAR, com representacao configuravel."""

from __future__ import annotations

import numpy as np
import pandas as pd

from gerador.schema.types import MecanismoNulo
from gerador.schema.variable import VariableConfig


def _mascara(
    serie: pd.Series,
    var: VariableConfig,
    df: pd.DataFrame,
    rng: np.random.Generator,
) -> np.ndarray:
    n = len(serie)
    config_nulos = var.qualidade.nulos
    pct = float(config_nulos.pct) / 100.0

    if pct <= 0:
        return np.zeros(n, dtype=bool)

    if config_nulos.mecanismo == MecanismoNulo.MAR.value and config_nulos.depende_de:
        dependente = config_nulos.depende_de
        if dependente in df.columns:
            base = df[dependente]
            postos = base.rank(method="average").to_numpy(dtype=float)
            intervalo = np.nanmax(postos) - np.nanmin(postos)
            postos = (postos - np.nanmin(postos)) / max(intervalo, 1e-12)
            postos = np.nan_to_num(postos, nan=0.5)
            propensao = np.clip(pct * postos / max(postos.mean(), 1e-12), 0.0, 1.0)
            return rng.random(n) < propensao

    return rng.random(n) < pct


def aplicar_nulos(
    serie: pd.Series,
    var: VariableConfig,
    df: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.Series:
    mascara = _mascara(serie, var, df, rng)
    if not mascara.any():
        return serie

    representacao = var.qualidade.nulos.representacao
    if representacao == "nan":
        resultado = serie.astype(float) if serie.dtype.kind in "iub" else serie.copy()
        resultado = serie.copy()
        resultado[mascara] = np.nan
        return resultado

    mapa = {"": "", "-1": -1, "desconhecido": "desconhecido"}
    valor = mapa.get(representacao, representacao)
    resultado = serie.astype(object).copy()
    resultado[mascara] = valor
    return resultado
