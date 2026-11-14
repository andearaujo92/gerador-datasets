"""Copula gaussiana: correlacao entre variaveis de tipos diferentes.

A forca e expressa em **Spearman** (invariante a transformacao monotonica).
Para continuas a conversao latente e exata:

    R = 2 * sin(pi * rho / 6)

Para discretas/categoricas os empates atenuam o resultado; a calibracao
iterativa corrige, e o diagnostico sempre mostra pedido vs. realizado.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from gerador.schema.relations import Relacao


def spearman_para_latente(rho: float | np.ndarray) -> float | np.ndarray:
    return 2.0 * np.sin(np.pi * np.asarray(rho, dtype=float) / 6.0)


def latente_para_spearman(r: float | np.ndarray) -> float | np.ndarray:
    return (6.0 / np.pi) * np.arcsin(np.clip(np.asarray(r, dtype=float) / 2.0, -1.0, 1.0))


def matriz_relacoes(
    nomes: list[str], relacoes: list[Relacao]
) -> tuple[np.ndarray, dict[str, int]]:
    """Monta a matriz latente a partir da lista de relacoes."""
    indices = {nome: i for i, nome in enumerate(nomes)}
    k = len(nomes)
    R = np.eye(k)
    for rel in relacoes:
        if rel.a not in indices or rel.b not in indices or rel.a == rel.b:
            continue
        i, j = indices[rel.a], indices[rel.b]
        valor = float(np.clip(spearman_para_latente(rel.forca), -1.0, 1.0))
        R[i, j] = R[j, i] = valor
    return R, indices


def validar_psd(R: np.ndarray, tol: float = 1e-8) -> tuple[bool, float]:
    """Uma matriz de correlacao so e valida se todos os autovalores forem >= 0."""
    R = np.atleast_2d(R)
    if R.size == 0:
        return True, 0.0
    autovalores = np.linalg.eigvalsh(R)
    menor = float(autovalores.min())
    return menor >= -tol, menor


def projetar_psd(R: np.ndarray) -> np.ndarray:
    """Matriz valida mais proxima (Higham), com diagonal renormalizada para 1."""
    R = np.asarray(R, dtype=float)
    simetrica = (R + R.T) / 2.0
    valores, vetores = np.linalg.eigh(simetrica)
    valores = np.clip(valores, 0.0, None)
    projetada = vetores @ np.diag(valores) @ vetores.T
    diag = np.sqrt(np.clip(np.diag(projetada), 1e-12, None))
    projetada = projetada / np.outer(diag, diag)
    np.fill_diagonal(projetada, 1.0)
    return (projetada + projetada.T) / 2.0


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman robusto a strings (usa postos)."""
    a = np.asarray(a)
    b = np.asarray(b)
    mascara = np.ones(len(a), dtype=bool)
    if a.dtype.kind == "f":
        mascara &= ~np.isnan(a)
    if b.dtype.kind == "f":
        mascara &= ~np.isnan(b)
    a, b = a[mascara], b[mascara]
    if len(a) < 3:
        return float("nan")
    ra = stats.rankdata(a)
    rb = stats.rankdata(b)
    if np.std(ra) == 0 or np.std(rb) == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def _amostrar_latente(n: int, R: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    k = R.shape[0]
    if k == 1:
        return rng.standard_normal((n, 1))
    # svd tolera matrizes semidefinidas (apos projecao PSD)
    return rng.multivariate_normal(np.zeros(k), R, size=n)


def amostrar(
    colunas: list[tuple[str, object]],
    R: np.ndarray,
    n: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """colunas: [(nome, callable(u) -> valores)]"""
    if not colunas:
        return {}
    Z = _amostrar_latente(n, R, rng)
    U = stats.norm.cdf(Z)
    return {nome: fn(U[:, j]) for j, (nome, fn) in enumerate(colunas)}


def amostrar_correlacionado(
    colunas: list[tuple[str, object]],
    relacoes,
    n: int,
    rng: np.random.Generator,
) -> tuple[dict[str, np.ndarray], dict]:
    """Gera as colunas respeitando as correlacoes pedidas.

    Retorna (dados, relatorio) com pedido vs. realizado e status da calibracao.
    """
    nomes = [c[0] for c in colunas]
    relatorio_base = {
        "pares": [],
        "calibrado": False,
        "convergiu": True,
        "iteracoes": 0,
        "projetada": False,
        "menor_autovalor": 0.0,
    }
    if not nomes:
        return {}, relatorio_base

    R, indices = matriz_relacoes(nomes, relacoes.lista)
    ok, min_eig = validar_psd(R)
    if not ok:
        R = projetar_psd(R)

    dados = amostrar(colunas, R, n, rng)

    relatorio = {
        "pares": [],
        "calibrado": False,
        "convergiu": True,
        "iteracoes": 0,
        "projetada": not ok,
        "menor_autovalor": min_eig,
    }

    pares = [
        (r.a, r.b, float(r.forca))
        for r in relacoes.lista
        if r.a in indices and r.b in indices and r.a != r.b
    ]
    if not pares:
        return dados, relatorio

    if not relacoes.calibrar:
        for a, b, alvo in pares:
            relatorio["pares"].append(
                {"a": a, "b": b, "alvo": alvo, "realizado": spearman(dados[a], dados[b])}
            )
        return dados, relatorio

    # calibracao: ajusta o coeficiente latente ate o Spearman realizado bater
    convergiu = False
    for iteracao in range(1, max(1, int(relacoes.max_iter)) + 1):
        maior_erro = 0.0
        for a, b, alvo in pares:
            i, j = indices[a], indices[b]
            realizado = spearman(dados[a], dados[b])
            if np.isnan(realizado):
                continue
            erro = alvo - realizado
            maior_erro = max(maior_erro, abs(erro))
            if abs(erro) <= relacoes.tolerancia:
                continue
            R[i, j] = R[j, i] = float(
                np.clip(R[i, j] + 0.5 * spearman_para_latente(erro), -0.999, 0.999)
            )
        relatorio["iteracoes"] = iteracao
        if maior_erro <= relacoes.tolerancia:
            convergiu = True
            break
        ok, _ = validar_psd(R)
        if not ok:
            R = projetar_psd(R)
        dados = amostrar(colunas, R, n, rng)

    relatorio["calibrado"] = True
    relatorio["convergiu"] = convergiu
    for a, b, alvo in pares:
        relatorio["pares"].append(
            {"a": a, "b": b, "alvo": alvo, "realizado": spearman(dados[a], dados[b])}
        )
    return dados, relatorio
