"""Marginais: momentos e quantis de cada distribuicao."""

from __future__ import annotations

import numpy as np
import pytest

from gerador.engine.marginals import amostrar_independente, ajustar_discretas, ppf
from gerador.schema.types import Distribuicao


def _amostra(dist, params, n=200_000, seed=1):
    return amostrar_independente(dist, params, n, np.random.default_rng(seed))


@pytest.mark.parametrize(
    "dist, params, media, desvio",
    [
        (Distribuicao.NORMAL, {"mu": 5.0, "sigma": 2.0}, 5.0, 2.0),
        (Distribuicao.UNIFORME, {"a": 0.0, "b": 10.0}, 5.0, 10 / np.sqrt(12)),
        (Distribuicao.EXPONENCIAL, {"lam": 0.5}, 2.0, 2.0),
        (Distribuicao.TRIANGULAR, {"min": 0.0, "moda": 5.0, "max": 10.0}, 5.0, None),
        (Distribuicao.POISSON, {"lam": 3.0}, 3.0, np.sqrt(3.0)),
        (Distribuicao.BINOMIAL, {"n": 10, "p": 0.3}, 3.0, np.sqrt(10 * 0.3 * 0.7)),
        (Distribuicao.GEOMETRICA, {"p": 0.25}, 4.0, np.sqrt(0.75) / 0.25),
    ],
)
def test_momentos(dist, params, media, desvio):
    x = _amostra(dist, params)
    assert abs(x.mean() - media) < 0.05 * max(abs(media), 1.0)
    if desvio is not None:
        assert abs(x.std() - desvio) < 0.05 * max(desvio, 1.0)


def test_log_normal_positiva_e_assimetrica():
    x = _amostra(Distribuicao.LOG_NORMAL, {"mu_log": 2.0, "sigma_log": 0.8})
    assert (x > 0).all()
    assert x.mean() > np.median(x)


def test_beta_limitada():
    x = _amostra(Distribuicao.BETA, {"alpha": 2.0, "beta": 5.0})
    assert x.min() >= 0 and x.max() <= 1


def test_gama_positiva():
    x = _amostra(Distribuicao.GAMA, {"shape": 3.0, "scale": 2.0})
    assert (x > 0).all()
    assert abs(x.mean() - 6.0) < 0.2


def test_binomial_negativa_media():
    x = _amostra(Distribuicao.BINOMIAL_NEGATIVA, {"r": 4.0, "p": 0.5})
    assert abs(x.mean() - 4.0) < 0.15


def test_discretas_retornam_inteiros():
    for dist, params in [
        (Distribuicao.POISSON, {"lam": 2.0}),
        (Distribuicao.BINOMIAL, {"n": 5, "p": 0.4}),
        (Distribuicao.GEOMETRICA, {"p": 0.5}),
    ]:
        x = _amostra(dist, params, n=2000)
        assert np.all(x == np.rint(x))
        assert x.min() >= 0


def test_empirica_respeita_pesos():
    params = {"itens": [{"valor": 10.0, "peso": 3.0}, {"valor": 20.0, "peso": 1.0}]}
    x = _amostra(Distribuicao.EMPIRICA, params, n=50_000)
    assert abs((x == 10.0).mean() - 0.75) < 0.02


def test_categorias_respeitam_pesos():
    params = {
        "itens": [
            {"rotulo": "solteiro", "peso": 4.0},
            {"rotulo": "casado", "peso": 5.0},
            {"rotulo": "divorciado", "peso": 1.0},
        ]
    }
    x = _amostra(Distribuicao.CATEGORIAS, params, n=50_000)
    # a copula trabalha com CODIGOS na ordem declarada (nao alfabetica)
    assert set(np.unique(x)) == {0, 1, 2}
    assert abs((x == 0).mean() - 0.4) < 0.02
    assert abs((x == 1).mean() - 0.5) < 0.02


def test_rotulos_categorias_preserva_ordem_declarada():
    from gerador.engine.marginals import rotulos_categorias

    params = {
        "itens": [
            {"rotulo": "alto", "peso": 1.0},
            {"rotulo": "baixo", "peso": 1.0},
            {"rotulo": "medio", "peso": 1.0},
        ]
    }
    assert rotulos_categorias(params) == ["alto", "baixo", "medio"]


def test_bernoulli_taxa():
    x = _amostra(Distribuicao.BERNOULLI, {"p": 0.3}, n=50_000)
    assert set(np.unique(x)) == {0, 1}
    assert abs(x.mean() - 0.3) < 0.01


def test_intervalo_de_datas():
    x = _amostra(
        Distribuicao.INTERVALO_DATAS,
        {"inicio": "2024-01-01", "fim": "2024-12-31", "tendencia": "nenhuma"},
        n=10_000,
    )
    assert x.dtype.kind == "i"
    assert x.min() >= 1704067200  # 2024-01-01 UTC
    assert x.max() <= 1735689600  # 2025-01-01 UTC


def test_ppf_e_monotono():
    u = np.linspace(0.001, 0.999, 500)
    x = ppf("normal", {"mu": 0.0, "sigma": 1.0}, u)
    assert np.all(np.diff(x) > 0)


def test_ajustar_discretas_arredonda_e_limita():
    x = np.array([-1.4, 2.6, 3.2])
    saida = ajustar_discretas(x, "numerica_discreta")
    assert list(saida) == [0, 3, 3]
