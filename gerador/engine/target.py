"""Geracao do target a partir das features ja limpas.

O target nunca entra na matriz de correlacao: ele e derivado. O intercepto das
binarias e resolvido por bisseccao **na amostra**, para a prevalencia realizada
bater a pedida.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit

from gerador.expressions import avaliar
from gerador.schema.target import TargetConfig
from gerador.schema.types import TargetModo, TargetTipo

#: Ganho aplicado ao score nas escalas logit/log (binaria, contagem, multiclasse).
#: Sem ele, "forca=1.0" produziria apenas AUC ~0.75; com ele, ~0.99.
_GAIN_LOGIT = 3.0


def _padronizar(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    desvio = float(np.std(x))
    if desvio <= 1e-12:
        return np.zeros_like(x)
    return (x - float(np.mean(x))) / desvio


def _score(df: pd.DataFrame, cfg: TargetConfig) -> tuple[np.ndarray, list[str]]:
    usadas: list[str] = []
    score = np.zeros(len(df))
    for inf in cfg.influencias:
        if inf.variavel not in df.columns:
            continue
        serie = df[inf.variavel]
        if serie.dtype.kind not in "iufb":
            serie = serie.astype("category").cat.codes
        score += float(inf.forca) * float(inf.direcao) * _padronizar(serie.to_numpy(dtype=float))
        usadas.append(inf.variavel)
    return score, usadas


def _resolver_intercepto(score: np.ndarray, alvo: float) -> float:
    lo, hi = -60.0, 60.0
    if float(expit(lo + score).mean()) > alvo:
        return lo
    if float(expit(hi + score).mean()) < alvo:
        return hi
    for _ in range(80):
        meio = (lo + hi) / 2.0
        if float(expit(meio + score).mean()) < alvo:
            lo = meio
        else:
            hi = meio
    return (lo + hi) / 2.0


def _flipar(y: np.ndarray, fracao: float, rng: np.random.Generator) -> np.ndarray:
    if fracao <= 0:
        return y
    y = y.copy()
    n = int(round(len(y) * min(fracao, 1.0)))
    if n <= 0:
        return y
    idx = rng.choice(len(y), size=n, replace=False)
    y[idx] = 1 - y[idx]
    return y


def _forcar_prevalencia(y: np.ndarray, alvo: float, rng: np.random.Generator) -> np.ndarray:
    y = y.copy()
    k = int(round(len(y) * alvo))
    atuais = int(y.sum())
    if atuais > k:
        positivos = np.flatnonzero(y == 1)
        y[rng.choice(positivos, size=atuais - k, replace=False)] = 0
    elif atuais < k:
        negativos = np.flatnonzero(y == 0)
        y[rng.choice(negativos, size=min(k - atuais, len(negativos)), replace=False)] = 1
    return y


def gerar_target(
    df: pd.DataFrame, cfg: TargetConfig, rng: np.random.Generator
) -> tuple[np.ndarray, dict]:
    """Retorna (valores, info) onde info alimenta a aba de diagnostico."""
    info: dict = {"tipo": cfg.tipo, "nome": cfg.nome}

    if cfg.modo == TargetModo.REGRA.value and cfg.regra:
        bruto = avaliar(cfg.regra, df, rng)
        y = np.asarray(bruto).astype(bool).astype(int)
        ruido = float(cfg.params.get("ruido", 0.0))
        y = _flipar(y, ruido, rng)
        info["prevalencia"] = float(y.mean())
        info["score"] = None
        return y, info

    score, usadas = _score(df, cfg)
    info["features"] = usadas
    ruido = float(cfg.params.get("ruido", 0.0))
    tipo = cfg.tipo
    score_logit = score * _GAIN_LOGIT

    if tipo == TargetTipo.BINARIA.value:
        prevalencia = float(cfg.params.get("prevalencia_base", 0.3))
        b0 = _resolver_intercepto(score_logit, prevalencia)
        p = expit(b0 + score_logit)
        y = (rng.random(len(df)) < p).astype(int)
        y = _flipar(y, ruido, rng)
        if cfg.params.get("forcar_prevalencia", True):
            y = _forcar_prevalencia(y, prevalencia, rng)
        info["prevalencia"] = float(y.mean())
        info["prevalencia_pedida"] = prevalencia
        info["score"] = p
        info["auc"] = _auc(y, p)
        return y, info

    if tipo == TargetTipo.REGRESSAO.value:
        media = float(cfg.params.get("media_base", 0.0))
        sigma = float(np.std(score)) * (0.01 + 4.99 * ruido)
        y = media + score + rng.normal(0.0, max(sigma, 1e-9), len(df))
        info["r2"] = _r2(y, media + score)
        info["media"] = float(np.mean(y))
        info["score"] = media + score
        return y, info

    if tipo == TargetTipo.CONTAGEM.value:
        lam_base = max(float(cfg.params.get("lam_base", 5.0)), 1e-6)
        efetivo = score_logit * (1.0 - ruido)
        deslocamento = np.log(lam_base) - np.log(np.mean(np.exp(efetivo)))
        lam = np.exp(deslocamento + efetivo)
        y = rng.poisson(lam).astype(int)
        info["media"] = float(np.mean(y))
        info["score"] = lam
        return y, info

    if tipo == TargetTipo.MULTICLASSE.value:
        classes = cfg.params.get("classes") or [{"rotulo": "A", "proporcao": 1.0}]
        k = len(classes)
        rotulos = [c["rotulo"] for c in classes]
        proporcoes = np.array([float(c.get("proporcao", 1 / k)) for c in classes], dtype=float)
        proporcoes = proporcoes / proporcoes.sum()
        alinhamento = np.cos(2 * np.pi * np.arange(k) / max(k, 1)) * (1.0 - ruido)
        logits = score_logit[:, None] * alinhamento[None, :] + np.log(proporcoes)[None, :]
        p = np.exp(logits - logits.max(axis=1, keepdims=True))
        p = p / p.sum(axis=1, keepdims=True)
        y = np.array([rng.choice(k, p=linha) for linha in p])
        info["rotulos"] = rotulos
        info["proporcoes"] = proporcoes.tolist()
        info["proporcoes_realizadas"] = [
            float((y == i).mean()) for i in range(k)
        ]
        info["score"] = p
        return y, info

    if tipo == TargetTipo.ORDINAL.value:
        rotulos = _lista(cfg.params.get("rotulos", "baixo, medio, alto"))
        proporcoes = _proporcoes(cfg.params.get("proporcoes", ""), len(rotulos))
        sigma = float(np.std(score)) or 1.0
        latente = score + rng.normal(0.0, sigma * (0.01 + 4.99 * ruido), len(df))
        cortes = np.quantile(latente, np.cumsum(proporcoes)[:-1])
        y = np.searchsorted(cortes, latente, side="right")
        info["rotulos"] = rotulos
        info["proporcoes"] = proporcoes
        info["proporcoes_realizadas"] = [float((y == i).mean()) for i in range(len(rotulos))]
        info["score"] = latente
        return y, info

    raise ValueError(f"tipo de target desconhecido: {tipo}")


def _lista(texto: str) -> list[str]:
    itens = [t.strip() for t in str(texto).split(",") if t.strip()]
    return itens or ["a", "b"]


def _proporcoes(texto: str, k: int) -> list[float]:
    try:
        valores = [float(t.strip().replace(",", ".")) for t in str(texto).split(",") if t.strip()]
    except ValueError:
        valores = []
    if len(valores) != k or sum(valores) <= 0:
        valores = [1.0] * k
    total = float(sum(valores))
    return [v / total for v in valores]


def _auc(y: np.ndarray, score: np.ndarray) -> float:
    from scipy import stats

    y = np.asarray(y)
    score = np.asarray(score, dtype=float)
    positivos = score[y == 1]
    negativos = score[y == 0]
    if len(positivos) == 0 or len(negativos) == 0:
        return float("nan")
    postos = stats.rankdata(np.concatenate([positivos, negativos]))
    soma_pos = float(postos[: len(positivos)].sum())
    return (soma_pos - len(positivos) * (len(positivos) + 1) / 2.0) / (
        len(positivos) * len(negativos)
    )


def _r2(y: np.ndarray, yhat: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    residuo = float(np.sum((y - yhat) ** 2))
    total = float(np.sum((y - np.mean(y)) ** 2))
    if total <= 1e-12:
        return float("nan")
    return 1.0 - residuo / total
