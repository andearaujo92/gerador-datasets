"""Transformacoes, escala, arredondamento e limites.

Ordem fixa: transformar -> escalar -> arredondar -> clipar.
"""

from __future__ import annotations

import numpy as np

from gerador.schema.types import DType, Transformacao
from gerador.schema.variable import Escala


def aplicar_escala(valores: np.ndarray, escala: Escala, dtype: str) -> np.ndarray:
    if dtype not in (DType.CONTINUA.value, DType.DISCRETA.value):
        return valores

    x = np.asarray(valores, dtype=float)

    t = escala.transformacao
    if t == Transformacao.LOG.value:
        x = np.log1p(np.clip(x, 0, None))
    elif t == Transformacao.RAIZ.value:
        x = np.sqrt(np.clip(x, 0, None))
    elif t == Transformacao.ZSCORE.value:
        desvio = float(np.std(x))
        if desvio > 1e-12:
            x = (x - float(np.mean(x))) / desvio
    elif t == Transformacao.MINMAX.value:
        lo, hi = float(np.min(x)), float(np.max(x))
        if hi - lo > 1e-12:
            x = (x - lo) / (hi - lo)

    if escala.multiplicador != 1.0:
        x = x * float(escala.multiplicador)
    if escala.offset != 0.0:
        x = x + float(escala.offset)

    if dtype == DType.DISCRETA.value:
        x = np.rint(x)
    else:
        x = np.round(x, int(escala.casas_decimais))

    if not escala.permitir_negativo:
        x = np.clip(x, 0, None)
    if escala.minimo is not None:
        x = np.clip(x, float(escala.minimo), None)
    if escala.maximo is not None:
        x = np.clip(x, None, float(escala.maximo))

    return x
