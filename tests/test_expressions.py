"""Avaliador de expressoes: seguranca e semantica."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gerador.expressions import avaliar, validar_expressao

COLUNAS = ["renda", "idade"]


@pytest.fixture
def df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "renda": np.array([1000.0, 3000.0, 6000.0, 9000.0]),
            "idade": np.array([20.0, 30.0, 40.0, 50.0]),
        }
    )


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


# -- semantica ---------------------------------------------------------


def test_comparacao_simples(df, rng):
    saida = avaliar("renda > 3000", df, rng)
    assert list(saida) == [False, False, True, True]


def test_aritmetica_e_booleanos(df, rng):
    saida = avaliar("(renda / 1000) + idade > 40", df, rng)
    assert list(saida) == [False, False, True, True]


def test_funcoes_permitidas(df, rng):
    saida = avaliar("log(renda) > 5", df, rng)
    esperado = np.log1p(df["renda"].to_numpy()) > 5
    assert np.array_equal(saida, esperado)


def test_ruido_normal_e_reprodutivel(df, rng):
    a = avaliar("renda + ruido_normal(0, 1)", df, np.random.default_rng(7))
    b = avaliar("renda + ruido_normal(0, 1)", df, np.random.default_rng(7))
    assert np.array_equal(a, b)


def test_expressao_booleana_composta(df, rng):
    saida = avaliar("(renda > 2000) and (idade < 45)", df, rng)
    assert list(saida) == [False, True, True, False]


def test_comparacao_encadeada(df, rng):
    saida = avaliar("1000 < renda <= 6000", df, rng)
    assert list(saida) == [False, True, True, False]


def test_negacao_com_not(df, rng):
    saida = avaliar("not (renda > 3000)", df, rng)
    assert list(saida) == [True, True, False, False]


def test_or_com_arrays(df, rng):
    saida = avaliar("(renda > 8000) or (idade < 25)", df, rng)
    assert list(saida) == [True, False, False, True]


def test_ifexp_e_rejeitado():
    # ternario nao se aplica a colunas (arrays) e falharia com "truth value ambiguous"
    assert validar_expressao("1 if renda > 3000 else 0", COLUNAS) is not None


# -- seguranca ---------------------------------------------------------


@pytest.mark.parametrize(
    "expressao",
    [
        "__import__('os').system('ls')",
        "renda.__class__",
        "renda[0]",
        "open('/etc/passwd')",
        "eval('1+1')",
        "[c for c in range(3)]",
        "lambda: 1",
    ],
)
def test_expressoes_perigosas_sao_rejeitadas(expressao):
    assert validar_expressao(expressao, COLUNAS) is not None


def test_expressao_perigosa_falha_ao_avaliar(df, rng):
    with pytest.raises(ValueError):
        avaliar("renda.__class__.__mro__", df, rng)


def test_nome_desconhecido(df, rng):
    assert "desconhecido" in (validar_expressao("salario > 10", COLUNAS) or "")
    with pytest.raises(ValueError):
        avaliar("salario > 10", df, rng)


def test_funcao_desconhecida():
    assert validar_expressao("foo(1)", COLUNAS) is not None


def test_sintaxe_invalida():
    erro = validar_expressao("renda >", COLUNAS)
    assert erro is not None and "sintaxe" in erro


def test_expressao_vazia():
    assert validar_expressao("   ", COLUNAS) is not None


def test_expressao_valida_retorna_none():
    assert validar_expressao("renda > 3000", COLUNAS) is None
