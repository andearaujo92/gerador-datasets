"""Marginais: uma unica interface `ppf(u)` para todas as distribuicoes.

Essa e a abstracao central do motor. Como toda distribuicao expoe apenas o
inverso da CDF, a copula gaussiana funciona para continuas, contagens,
categoricas e datas sem nenhum caso especial.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from gerador.schema.types import Distribuicao

EPS = 1e-9


def _u(u: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(u, dtype=float), EPS, 1.0 - EPS)


def _itens(itens: list[dict], campo: str) -> tuple[np.ndarray, np.ndarray]:
    """Valores e pesos (nao normalizados) na ordem declarada."""
    valores = np.array([i[campo] for i in itens])
    pesos = np.array([float(i.get("peso", 1.0)) for i in itens], dtype=float)
    return valores, np.clip(pesos, 1e-12, None)


def _acumulado(pesos: np.ndarray) -> np.ndarray:
    acumulado = np.cumsum(pesos) / pesos.sum()
    acumulado[-1] = 1.0
    return acumulado


def ppf(distribuicao: str | Distribuicao, params: dict, u: np.ndarray) -> np.ndarray:
    """Inverso da CDF. Recebe u em [0,1] e devolve o valor da variavel."""
    d = Distribuicao(distribuicao)
    u = _u(u)
    p = params or {}

    if d is Distribuicao.NORMAL:
        return stats.norm(loc=float(p["mu"]), scale=max(float(p["sigma"]), 1e-9)).ppf(u)

    if d is Distribuicao.UNIFORME:
        a, b = float(p["a"]), float(p["b"])
        return stats.uniform(loc=a, scale=max(b - a, 1e-9)).ppf(u)

    if d is Distribuicao.LOG_NORMAL:
        return stats.lognorm(
            s=max(float(p["sigma_log"]), 1e-9), scale=float(np.exp(float(p["mu_log"])))
        ).ppf(u)

    if d is Distribuicao.EXPONENCIAL:
        return stats.expon(scale=1.0 / max(float(p["lam"]), 1e-9)).ppf(u)

    if d is Distribuicao.GAMA:
        return stats.gamma(a=max(float(p["shape"]), 1e-9), scale=max(float(p["scale"]), 1e-9)).ppf(u)

    if d is Distribuicao.BETA:
        return stats.beta(
            a=max(float(p["alpha"]), 1e-9), b=max(float(p["beta"]), 1e-9)
        ).ppf(u)

    if d is Distribuicao.TRIANGULAR:
        lo, mo, hi = float(p["min"]), float(p["moda"]), float(p["max"])
        span = max(hi - lo, 1e-9)
        c = float(np.clip((mo - lo) / span, 0.0, 1.0))
        return stats.triang(c=c, loc=lo, scale=span).ppf(u)

    if d is Distribuicao.POISSON:
        return stats.poisson(mu=max(float(p["lam"]), 1e-9)).ppf(u)

    if d is Distribuicao.BINOMIAL:
        return stats.binom(n=int(p["n"]), p=float(p["p"])).ppf(u)

    if d is Distribuicao.BINOMIAL_NEGATIVA:
        return stats.nbinom(n=float(p["r"]), p=float(p["p"])).ppf(u)

    if d is Distribuicao.GEOMETRICA:
        return stats.geom(p=float(p["p"])).ppf(u)

    if d is Distribuicao.EMPIRICA:
        # massa de probabilidade nos valores listados (funcao degrau), ordenada
        # pelo proprio valor para a quantilica continuar monotonica.
        valores, pesos = _itens(p["itens"], "valor")
        ordem = np.argsort(np.asarray(valores, dtype=float), kind="stable")
        valores = np.asarray(valores)[ordem]
        acumulado = _acumulado(pesos[ordem])
        idx = np.searchsorted(acumulado, u, side="left")
        return valores[np.clip(idx, 0, len(valores) - 1)]

    if d is Distribuicao.CATEGORIAS:
        # O motor trabalha com CODIGOS (0..k-1), nao com rotulos: assim a
        # correlacao usa a ordem declarada e nao a ordem alfabetica.
        _, pesos = _itens(p["itens"], "rotulo")
        acumulado = _acumulado(pesos)
        return np.clip(np.searchsorted(acumulado, u, side="left"), 0, len(pesos) - 1).astype(
            "int64"
        )

    if d is Distribuicao.BERNOULLI:
        prob = float(p["p"])
        return (u > (1.0 - prob)).astype(int)

    if d is Distribuicao.INTERVALO_DATAS:
        from datetime import datetime

        inicio = _para_timestamp(p.get("inicio", "2024-01-01"))
        fim = _para_timestamp(p.get("fim", "2025-12-31"))
        lo, hi = min(inicio, fim), max(inicio, fim)

        tendencia = p.get("tendencia", "nenhuma")
        if tendencia == "crescente":
            uu = np.sqrt(u)
        elif tendencia == "decrescente":
            uu = u**2
        elif tendencia == "sazonal":
            uu = np.clip(u + 0.08 * np.sin(2 * np.pi * 3 * u), 0, 1)
        else:
            uu = u

        segundos = lo + uu * max(hi - lo, 1)
        return np.floor(segundos).astype("int64")

    if d is Distribuicao.ID:
        raise ValueError("ID nao passa pela copula; use engine.text.gerar_ids")

    raise ValueError(f"distribuicao desconhecida: {distribuicao}")


def rotulos_categorias(params: dict) -> list[str]:
    """Rotulos na ordem declarada pelo usuario (nao a ordem alfabetica)."""
    return [str(i["rotulo"]) for i in (params or {}).get("itens", [])]


def _para_timestamp(valor) -> float:
    from datetime import datetime

    if isinstance(valor, (int, float)):
        return float(valor)
    if isinstance(valor, datetime):
        return valor.timestamp()
    texto = str(valor).strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%d/%m/%Y"):
        try:
            return datetime.strptime(texto, fmt).timestamp()
        except ValueError:
            continue
    raise ValueError(f"data invalida: {valor!r}")


def amostrar_independente(
    distribuicao: str | Distribuicao, params: dict, n: int, rng: np.random.Generator
) -> np.ndarray:
    """Amostra sem correlacao (usado em teste e em preview isolado)."""
    return ppf(distribuicao, params, rng.random(n))


def ajustar_discretas(valores: np.ndarray, dtype: str) -> np.ndarray:
    """Forca inteiro >= 0 para variaveis de contagem."""
    from gerador.schema.types import DType

    if dtype == DType.DISCRETA.value:
        return np.clip(np.rint(np.asarray(valores, dtype=float)), 0, None)
    return valores
