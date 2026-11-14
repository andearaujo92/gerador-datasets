"""IDs: determinismo, unicidade, colisao e formatacao."""

from __future__ import annotations

import numpy as np
import pytest

from gerador.engine.text import gerar_ids, probabilidade_colisao


def test_sequencial():
    saida = gerar_ids(5, np.random.default_rng(1), modo="sequencial")
    assert list(saida) == [1, 2, 3, 4, 5]


def test_formatado_com_padding():
    saida = gerar_ids(3, np.random.default_rng(1), modo="formatado", prefixo="CLI-", n_digitos=4)
    assert list(saida) == ["CLI-0001", "CLI-0002", "CLI-0003"]


def test_formatado_expande_digitos_quando_necessario():
    saida = gerar_ids(12_000, np.random.default_rng(1), modo="formatado", n_digitos=3)
    assert len(saida[0]) == len("ID-") + 5


def test_aleatorio_deterministico_com_mesma_seed():
    a = gerar_ids(500, np.random.default_rng(42), modo="aleatorio", comprimento=10)
    b = gerar_ids(500, np.random.default_rng(42), modo="aleatorio", comprimento=10)
    assert np.array_equal(a, b)


def test_aleatorio_muda_com_seed_diferente():
    a = gerar_ids(200, np.random.default_rng(1), modo="aleatorio", comprimento=10)
    b = gerar_ids(200, np.random.default_rng(2), modo="aleatorio", comprimento=10)
    assert not np.array_equal(a, b)


def test_aleatorio_unico():
    saida = gerar_ids(5_000, np.random.default_rng(3), modo="aleatorio", comprimento=10)
    assert len(set(saida)) == 5_000
    assert all(len(c) == 10 for c in saida)


def test_aleatorio_sem_unico_permite_repeticao():
    saida = gerar_ids(
        2_000, np.random.default_rng(4), modo="aleatorio", comprimento=2, unico=False
    )
    assert len(set(saida)) < 2_000


def test_colisao_recusada_quando_espaco_insuficiente():
    with pytest.raises(ValueError, match="colidem"):
        gerar_ids(5_000, np.random.default_rng(5), modo="aleatorio", comprimento=2)


def test_alfabeto_numerico():
    saida = gerar_ids(200, np.random.default_rng(6), modo="aleatorio", comprimento=12, alfabeto="numeric")
    assert all(c.isdigit() for c in saida)


def test_probabilidade_colisao_cresce():
    assert probabilidade_colisao(10, 10**9) < probabilidade_colisao(10_000, 10**9)
    assert probabilidade_colisao(0, 10) == 0.0
