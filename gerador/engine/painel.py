"""Motor de painel mensal: cada entidade atravessa os meses com persistencia.

O desenho separa duas fontes de variacao, o que permite ter ao mesmo tempo
correlacao **entre** variaveis (na cross-section de cada mes) e autocorrelacao
**dentro** da entidade ao longo do tempo:

    Z[j, t] = sqrt(phi_j) * B[j] + sqrt(1 - phi_j) * W[j, t]

`B` e o traco fixo da entidade (o que faz `idade` ou `limite` serem constantes e
ainda assim correlacionados com `renda`), `W` e o choque do mes. Como
`Var(Z) = 1`, a correlacao lag-1 de cada variavel e exatamente `phi_j` e a
correlacao entre `j` e `k` em qualquer mes e a pedida, desde que a matriz de
`B`/`W` seja compensada por:

    R_B[j,k] = R[j,k] / (sqrt(phi_j phi_k) + sqrt((1-phi_j)(1-phi_k)))

Sem a compensacao, variaveis com phis diferentes atenuariam a correlacao
cruzada. A matriz compensada e projetada para PSD antes de amostrar.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
from scipy import stats

from gerador.engine.copula import (
    matriz_relacoes,
    projetar_psd,
    spearman,
    spearman_para_latente,
    validar_psd,
)
from gerador.engine.marginals import ppf, rotulos_categorias
from gerador.engine.text import gerar_ids
from gerador.engine.transforms import aplicar_escala
from gerador.schema.dataset import DatasetConfig
from gerador.schema.painel import meses_do_painel
from gerador.schema.types import TIPOS_COPULAVEIS, DType, ModoEvolucao
from gerador.schema.variable import VariableConfig

EPS = 1e-9


def phi_efetivo(var: VariableConfig) -> float:
    """Persistencia usada pelo motor (estatica/tendencia = traco fixo)."""
    if var.evolucao.modo == ModoEvolucao.AR1.value:
        return float(np.clip(var.evolucao.phi, 0.0, 0.999))
    return 1.0


def _compensar(R: np.ndarray, phis: np.ndarray) -> np.ndarray:
    a = np.sqrt(np.clip(phis, 0.0, 1.0))
    b = np.sqrt(np.clip(1.0 - phis, 0.0, 1.0))
    denominador = np.outer(a, a) + np.outer(b, b)
    Rb = R / np.clip(denominador, EPS, None)
    Rb = np.clip(Rb, -1.0, 1.0)
    np.fill_diagonal(Rb, 1.0)
    return Rb


def _latente(n: int, Rb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    k = Rb.shape[0]
    if k == 0:
        return np.empty((n, 0))
    if k == 1:
        return rng.standard_normal((n, 1))
    return rng.multivariate_normal(np.zeros(k), Rb, size=n)


def _combinar(B: np.ndarray, W: np.ndarray, phis: np.ndarray) -> np.ndarray:
    a = np.sqrt(np.clip(phis, 0.0, 1.0))[None, :]
    b = np.sqrt(np.clip(1.0 - phis, 0.0, 1.0))[None, :]
    return a * B + b * W


def _valores(var: VariableConfig, z: np.ndarray) -> np.ndarray:
    """Latente normal -> valores finais (marginal + escala)."""
    u = stats.norm.cdf(z)
    valores = ppf(var.distribuicao, var.params, u)
    return aplicar_escala(valores, var.escala, var.dtype)


def _limitar(var: VariableConfig, valores: np.ndarray) -> np.ndarray:
    """Reaplica apenas limites/arredondamento (sem transformacao)."""
    escala = replace(
        var.escala,
        transformacao="nenhuma",
        multiplicador=1.0,
        offset=0.0,
    )
    return aplicar_escala(valores, escala, var.dtype)


def _copula_painel(
    copulaveis: list[VariableConfig],
    config: DatasetConfig,
    n: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Calibra a cross-section do mes 1 e devolve (B, Rb, phis, Z1, relatorio)."""
    nomes = [v.nome for v in copulaveis]
    phis = np.array([phi_efetivo(v) for v in copulaveis], dtype=float)
    k = len(nomes)

    if k == 0:
        return np.empty((n, 0)), np.empty((0, 0)), phis, np.empty((n, 0)), {
            "pares": [],
            "calibrado": False,
            "convergiu": True,
            "iteracoes": 0,
            "projetada": False,
            "menor_autovalor": 0.0,
        }

    R, indices = matriz_relacoes(
        nomes, config.relacoes.lista, completar=bool(config.relacoes.completar_psd)
    )
    ok, min_eig = validar_psd(R)
    if not ok:
        R = projetar_psd(R)

    Rb = _compensar(R, phis)
    if not validar_psd(Rb)[0]:
        Rb = projetar_psd(Rb)

    pares = [
        (r.a, r.b, float(r.forca))
        for r in config.relacoes.lista
        if r.a in indices and r.b in indices and r.a != r.b
    ]

    B = _latente(n, Rb, rng)
    W = _latente(n, Rb, rng)
    Z1 = _combinar(B, W, phis)
    valores = {v.nome: _valores(v, Z1[:, j]) for j, v in enumerate(copulaveis)}

    relatorio = {
        "pares": [],
        "calibrado": False,
        "convergiu": True,
        "iteracoes": 0,
        "projetada": not ok,
        "menor_autovalor": min_eig,
    }

    if config.relacoes.calibrar and pares:
        convergiu = False
        for iteracao in range(1, max(1, int(config.relacoes.max_iter)) + 1):
            relatorio["iteracoes"] = iteracao
            maior_erro = 0.0
            for a, b, alvo in pares:
                realizado = spearman(valores[a], valores[b])
                if np.isnan(realizado):
                    continue
                erro = alvo - realizado
                maior_erro = max(maior_erro, abs(erro))
                if abs(erro) <= config.relacoes.tolerancia:
                    continue
                i, j = indices[a], indices[b]
                R[i, j] = R[j, i] = float(
                    np.clip(R[i, j] + 0.5 * spearman_para_latente(erro), -0.999, 0.999)
                )
            if maior_erro <= config.relacoes.tolerancia:
                convergiu = True
                break
            if not validar_psd(R)[0]:
                R = projetar_psd(R)
            Rb = _compensar(R, phis)
            if not validar_psd(Rb)[0]:
                Rb = projetar_psd(Rb)
            B = _latente(n, Rb, rng)
            W = _latente(n, Rb, rng)
            Z1 = _combinar(B, W, phis)
            valores = {v.nome: _valores(v, Z1[:, j]) for j, v in enumerate(copulaveis)}
        relatorio["calibrado"] = True
        relatorio["convergiu"] = convergiu

    return B, Rb, phis, Z1, relatorio


def gerar_painel(
    config: DatasetConfig, rng: np.random.Generator
) -> tuple[pd.DataFrame, dict]:
    """Gera o painel longo (entidade x mes) e o relatorio de correlacao."""
    n = int(config.n_linhas)
    T = int(config.painel.n_meses)
    meses = meses_do_painel(config.painel.data_inicio, T)
    copulaveis = [v for v in config.variaveis if v.tipo in TIPOS_COPULAVEIS]

    B, Rb, phis, Z1, relatorio = _copula_painel(copulaveis, config, n, rng)
    k = len(copulaveis)

    # -- todos os meses ------------------------------------------------
    Z = np.empty((n, T, k), dtype=float)
    if k:
        Z[:, 0, :] = Z1
        for t in range(1, T):
            Z[:, t, :] = _combinar(B, _latente(n, Rb, rng), phis)

    dados: dict[str, np.ndarray] = {}
    #: Valores do mes 1 na escala usada pela copula (codigos para categorias):
    #: o diagnostico precisa dessa escala, nao dos rotulos ja convertidos.
    mes1: dict[str, np.ndarray] = {}

    # -- copulaveis: valores por mes, achatados em ordem entidade-mes ----
    for j, var in enumerate(copulaveis):
        if var.evolucao.modo == ModoEvolucao.TENDENCIA.value:
            inicial = _valores(var, Z[:, 0, j])
            serie = inicial[:, None] + float(var.evolucao.passo) * np.arange(T)[None, :]
            serie = _limitar(var, serie)
        elif phis[j] >= 1.0:
            inicial = _valores(var, Z[:, 0, j])
            serie = np.repeat(inicial[:, None], T, axis=1)
        else:
            serie = _valores(var, Z[:, :, j])

        matriz = np.asarray(serie)
        mes1[var.nome] = matriz[:, 0]
        plano = matriz.reshape(-1)

        if var.dist.value == "categorias":
            rotulos = rotulos_categorias(var.params)
            codigos = np.clip(np.rint(plano.astype(float)).astype("int64"), 0, len(rotulos) - 1)
            dados[var.nome] = pd.Categorical.from_codes(
                codigos, categories=rotulos, ordered=(var.dtype == DType.ORDINAL.value)
            )
        else:
            dados[var.nome] = plano

    # -- texto: um id por entidade, repetido em todos os meses ----------
    for var in config.variaveis:
        if var.tipo is not DType.TEXTO:
            continue
        p = var.params
        ids = gerar_ids(
            n,
            rng,
            modo=p.get("modo", "formatado"),
            prefixo=p.get("prefixo", "ID-"),
            n_digitos=int(p.get("n_digitos", 5)),
            comprimento=int(p.get("comprimento", 8)),
            alfabeto=p.get("alfabeto", "alnum"),
            unico=bool(p.get("unico", True)),
        )
        dados[var.nome] = np.repeat(ids, T)

    # -- coluna de tempo: dia 1 de cada mes, repetida por entidade -------
    dados[config.painel.coluna_tempo] = np.tile(
        np.asarray(meses, dtype="datetime64[ns]"), n
    )

    # -- relatorio: pedido vs. realizado no mes 1 -----------------------
    nomes_cop = [v.nome for v in copulaveis]
    indices = {nome: i for i, nome in enumerate(nomes_cop)}
    pares = [
        (r.a, r.b, float(r.forca))
        for r in config.relacoes.lista
        if r.a in indices and r.b in indices and r.a != r.b
    ]
    for a, b, alvo in pares:
        relatorio["pares"].append(
            {
                "a": a,
                "b": b,
                "alvo": alvo,
                "realizado": spearman(mes1[a], mes1[b]),
            }
        )

    df = pd.DataFrame({nome: valores for nome, valores in dados.items()})
    return df, relatorio
