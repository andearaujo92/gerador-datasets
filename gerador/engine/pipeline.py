"""Orquestracao da geracao.

Ordem fixa (ela importa):
  1. marginais + copula
  2. transformar / escalar / arredondar / clipar
  3. target (enxerga as features integras)
  4. outliers
  5. nulos
  6. ordenar colunas, ajustar dtypes
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from gerador.engine import diagnostics
from gerador.engine.copula import amostrar_correlacionado
from gerador.engine.marginals import ppf, rotulos_categorias
from gerador.engine.missing import aplicar_nulos
from gerador.engine.outliers import aplicar_outliers
from gerador.engine.target import gerar_target
from gerador.engine.text import gerar_ids
from gerador.engine.transforms import aplicar_escala
from gerador.schema.dataset import DatasetConfig
from gerador.schema.types import TIPOS_COPULAVEIS, DType


@dataclass
class Resultado:
    config: DatasetConfig
    df: pd.DataFrame | None = None
    erros: list[str] = field(default_factory=list)
    avisos: list[str] = field(default_factory=list)
    relatorio_correlacao: dict = field(default_factory=dict)
    comparativo: pd.DataFrame | None = None
    target_info: dict = field(default_factory=dict)
    tempo_ms: float = 0.0

    @property
    def ok(self) -> bool:
        return self.df is not None and not self.erros


def gerar_dataset(config: DatasetConfig) -> Resultado:
    import time

    inicio = time.perf_counter()
    resultado = Resultado(config=config)
    resultado.erros, resultado.avisos = config.validar()
    if resultado.erros:
        return resultado

    rng = np.random.default_rng(int(config.seed))
    n = int(config.n_linhas)
    variaveis = config.variaveis

    # -- 1. marginais + copula ----------------------------------------
    copulaveis = [v for v in variaveis if v.tipo in TIPOS_COPULAVEIS]
    colunas = [
        (v.nome, (lambda u, v=v: ppf(v.distribuicao, v.params, u))) for v in copulaveis
    ]
    dados, relatorio = amostrar_correlacionado(colunas, config.relacoes, n, rng)
    resultado.relatorio_correlacao = relatorio

    # -- 2. escala ----------------------------------------------------
    for v in copulaveis:
        dados[v.nome] = aplicar_escala(dados[v.nome], v.escala, v.dtype)

    # -- 2b. codigos -> rotulos (ordem declarada, nao alfabetica) ------
    for v in copulaveis:
        if v.dist.value != "categorias":
            continue
        rotulos = rotulos_categorias(v.params)
        codigos = np.clip(np.asarray(dados[v.nome]).astype("int64"), 0, len(rotulos) - 1)
        dados[v.nome] = pd.Categorical.from_codes(
            codigos, categories=rotulos, ordered=(v.dtype == DType.ORDINAL.value)
        )

    # -- texto (fora da copula) ---------------------------------------
    for v in variaveis:
        if v.tipo is not DType.TEXTO:
            continue
        p = v.params
        try:
            dados[v.nome] = gerar_ids(
                n,
                rng,
                modo=p.get("modo", "formatado"),
                prefixo=p.get("prefixo", "ID-"),
                n_digitos=int(p.get("n_digitos", 5)),
                comprimento=int(p.get("comprimento", 8)),
                alfabeto=p.get("alfabeto", "alnum"),
                unico=bool(p.get("unico", True)),
            )
        except ValueError as exc:
            resultado.erros.append(f"'{v.nome}': {exc}")
            return resultado

    df = pd.DataFrame({v.nome: dados[v.nome] for v in variaveis if v.nome in dados})

    # -- 3. target ----------------------------------------------------
    if config.target.habilitado:
        try:
            y, info = gerar_target(df, config.target, rng)
        except Exception as exc:  # erro de expressao do usuario
            resultado.erros.append(f"target: {exc}")
            return resultado
        df[config.target.nome] = y
        resultado.target_info = info

    # -- 4. outliers --------------------------------------------------
    for v in variaveis:
        if v.tipo not in (DType.CONTINUA, DType.DISCRETA):
            continue
        cfg_out = v.qualidade.outliers
        if cfg_out.metodo == "nenhum" or cfg_out.pct <= 0:
            continue
        fn = None
        if cfg_out.metodo == "cauda":
            fn = lambda u, v=v: ppf(v.distribuicao, v.params, u)  # noqa: E731
        df[v.nome] = aplicar_outliers(
            df[v.nome].to_numpy(dtype=float),
            cfg_out.pct,
            cfg_out.metodo,
            cfg_out.intensidade,
            rng,
            fn,
        )

    # -- 5. nulos -----------------------------------------------------
    for v in variaveis:
        if v.qualidade.nulos.pct <= 0:
            continue
        df[v.nome] = aplicar_nulos(df[v.nome], v, df, rng)

    # -- 6. ordem, dtypes, diagnostico --------------------------------
    ordem = config.nomes + ([config.target.nome] if config.target.habilitado else [])
    df = df[[c for c in ordem if c in df.columns]]
    df = _ajustar_dtypes(df, config)

    resultado.df = df
    resultado.comparativo = diagnostics.comparar(df, config.relacoes.lista)
    resultado.tempo_ms = (time.perf_counter() - inicio) * 1000.0
    return resultado


def _ajustar_dtypes(df: pd.DataFrame, config: DatasetConfig) -> pd.DataFrame:
    df = df.copy()
    for v in config.variaveis:
        if v.nome not in df.columns:
            continue
        serie = df[v.nome]
        try:
            if v.tipo is DType.CONTINUA:
                convertida = _numerico_seguro(serie)
                df[v.nome] = (
                    convertida.astype("float64") if convertida is not None else serie.astype(object)
                )
            elif v.tipo is DType.DISCRETA:
                convertida = _numerico_seguro(serie)
                if convertida is None:
                    df[v.nome] = serie.astype(object)
                else:
                    df[v.nome] = convertida.astype("Int64" if convertida.isna().any() else "int64")
            elif v.tipo is DType.BOOLEANA:
                if serie.isna().any():
                    df[v.nome] = serie.astype("boolean")
                else:
                    df[v.nome] = serie.astype(bool)
            elif v.tipo is DType.DATA_HORA:
                df[v.nome] = pd.to_datetime(serie, unit="s", errors="coerce")
            elif v.tipo in (DType.ORDINAL, DType.CATEGORICA):
                # preserva a ORDEM declarada das categorias (nao a alfabetica)
                if isinstance(serie.dtype, pd.CategoricalDtype):
                    df[v.nome] = serie
                else:
                    categorias = list(dict.fromkeys(serie.dropna().astype(str)))
                    df[v.nome] = pd.Categorical(
                        serie.astype(object), categories=categorias, ordered=v.tipo is DType.ORDINAL
                    )
            else:
                df[v.nome] = serie.astype(object)
        except (TypeError, ValueError):
            df[v.nome] = serie.astype(object)
    return df


def _numerico_seguro(serie: pd.Series) -> pd.Series | None:
    """Converte para numero apenas se nao destruir representacoes nao numericas.

    Uma coluna com nulos representados por 'desconhecido' precisa continuar
    legivel no arquivo exportado; coagi-la para NaN apagaria essa informacao.
    """
    convertida = pd.to_numeric(serie, errors="coerce")
    if int(convertida.isna().sum()) <= int(serie.isna().sum()):
        return convertida
    return None
