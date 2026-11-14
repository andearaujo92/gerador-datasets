"""Outliers configuraveis: cauda da distribuicao, IQR ou valor fixo."""

from __future__ import annotations

import numpy as np

from gerador.schema.types import MetodoOutlier


def aplicar_outliers(
    valores: np.ndarray,
    pct: float,
    metodo: str,
    intensidade: float,
    rng: np.random.Generator,
    ppf_fn=None,
) -> np.ndarray:
    """Aplica outliers a uma coluna numerica. `ppf_fn(u)` amostra a cauda."""
    if metodo == MetodoOutlier.NENHUM.value or pct <= 0:
        return valores

    x = np.asarray(valores, dtype=float).copy()
    n = len(x)
    if n == 0:
        return x
    k = int(round(n * float(pct) / 100.0))
    if k <= 0:
        return x
    idx = rng.choice(n, size=min(k, n), replace=False)

    if metodo == MetodoOutlier.CAUDA.value:
        if ppf_fn is not None:
            cauda = ppf_fn(1.0 - rng.uniform(1e-6, 5e-3, size=len(idx)))
            x[idx] = np.asarray(cauda, dtype=float)
        else:
            desvio = float(np.std(x)) or 1.0
            x[idx] = x[idx] + intensidade * desvio
    elif metodo == MetodoOutlier.IQR.value:
        q1, q3 = np.percentile(x, [25, 75])
        iqr = q3 - q1
        if iqr <= 0:
            iqr = float(np.std(x)) or 1.0
        escolha = rng.random(len(idx)) < 0.5
        altos = q3 + float(intensidade) * iqr
        baixos = q1 - float(intensidade) * iqr
        x[idx] = np.where(escolha, altos, baixos)
    elif metodo == MetodoOutlier.FIXO.value:
        x[idx] = float(intensidade)

    return x
