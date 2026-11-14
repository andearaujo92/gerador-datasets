"""Copula: correlacao realizada, PSD e calibracao."""

from __future__ import annotations

import numpy as np
import pytest

from gerador.engine.copula import (
    amostrar_correlacionado,
    latente_para_spearman,
    matriz_relacoes,
    projetar_psd,
    spearman,
    spearman_para_latente,
    validar_psd,
)
from gerador.engine.marginals import ppf
from gerador.schema.relations import Relacao, RelacoesConfig


def _colunas(nomes_params):
    return [(nome, (lambda u, d=d, p=p: ppf(d, p, u))) for nome, d, p in nomes_params]


def test_conversao_spearman_latente_e_inversa():
    for rho in (-0.9, -0.3, 0.0, 0.4, 0.95):
        assert latente_para_spearman(spearman_para_latente(rho)) == pytest.approx(rho, abs=1e-9)


def test_correlacao_continua_atinge_o_alvo():
    colunas = _colunas(
        [
            ("a", "normal", {"mu": 0, "sigma": 1}),
            ("b", "normal", {"mu": 0, "sigma": 1}),
        ]
    )
    relacoes = RelacoesConfig(lista=[Relacao("a", "b", 0.7)], calibrar=False)
    dados, relatorio = amostrar_correlacionado(
        colunas, relacoes, n=20_000, rng=np.random.default_rng(11)
    )
    realizado = spearman(dados["a"], dados["b"])
    assert realizado == pytest.approx(0.7, abs=0.03)
    assert relatorio["pares"][0]["alvo"] == 0.7


def test_correlacao_negativa():
    colunas = _colunas(
        [
            ("a", "normal", {"mu": 0, "sigma": 1}),
            ("b", "log_normal", {"mu_log": 1.0, "sigma_log": 0.5}),
        ]
    )
    relacoes = RelacoesConfig(lista=[Relacao("a", "b", -0.6)])
    dados, _ = amostrar_correlacionado(colunas, relacoes, n=20_000, rng=np.random.default_rng(12))
    assert spearman(dados["a"], dados["b"]) == pytest.approx(-0.6, abs=0.04)


def test_calibracao_melhora_ordinal():
    """Com empates a correlacao crua atenua; a calibracao deve aproximar."""
    colunas = _colunas(
        [
            ("a", "normal", {"mu": 0, "sigma": 1}),
            (
                "b",
                "categorias",
                {
                    "itens": [
                        {"rotulo": "baixo", "peso": 1},
                        {"rotulo": "medio", "peso": 1},
                        {"rotulo": "alto", "peso": 1},
                    ]
                },
            ),
        ]
    )
    alvo = 0.8
    sem_calibrar, _ = amostrar_correlacionado(
        colunas,
        RelacoesConfig(lista=[Relacao("a", "b", alvo)], calibrar=False),
        n=20_000,
        rng=np.random.default_rng(13),
    )
    calibrado, relatorio = amostrar_correlacionado(
        colunas,
        RelacoesConfig(lista=[Relacao("a", "b", alvo)], calibrar=True, max_iter=15),
        n=20_000,
        rng=np.random.default_rng(13),
    )
    bruto = abs(spearman(sem_calibrar["a"], sem_calibrar["b"]) - alvo)
    final = abs(spearman(calibrado["a"], calibrado["b"]) - alvo)
    assert final < bruto
    assert relatorio["calibrado"] is True


def test_matriz_psd_valida():
    R, indices = matriz_relacoes(
        ["a", "b", "c"], [Relacao("a", "b", 0.5), Relacao("b", "c", 0.4)]
    )
    ok, menor = validar_psd(R)
    assert ok
    assert menor > -1e-8
    assert indices["c"] == 2
    assert R[0, 1] == R[1, 0]


def test_matriz_invalida_e_projetada():
    R = np.array(
        [
            [1.0, 0.95, -0.95],
            [0.95, 1.0, 0.95],
            [-0.95, 0.95, 1.0],
        ]
    )
    ok, menor = validar_psd(R)
    assert not ok and menor < 0
    projetada = projetar_psd(R)
    ok2, menor2 = validar_psd(projetada)
    assert ok2
    assert menor2 >= -1e-8
    assert np.allclose(np.diag(projetada), 1.0)
    assert np.allclose(projetada, projetada.T)


def test_sem_relacoes_gera_independentes():
    colunas = _colunas(
        [
            ("a", "normal", {"mu": 0, "sigma": 1}),
            ("b", "normal", {"mu": 0, "sigma": 1}),
        ]
    )
    dados, relatorio = amostrar_correlacionado(
        colunas, RelacoesConfig(), n=20_000, rng=np.random.default_rng(14)
    )
    assert abs(spearman(dados["a"], dados["b"])) < 0.05
    assert relatorio["pares"] == []


def test_spearman_com_nulos_ignora_pares():
    a = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0])
    b = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    assert spearman(a, b) == pytest.approx(1.0)
