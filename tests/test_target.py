"""Target: prevalencia, AUC vs. ruido, R2, multiclasse e regra."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gerador.engine.target import gerar_target
from gerador.schema.target import Influencia, TargetConfig
from gerador.schema.types import TargetTipo


def _df(n=20_000, seed=9) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    renda = rng.normal(3000, 800, n)
    saldo = 0.6 * renda + rng.normal(0, 400, n)
    return pd.DataFrame({"renda": renda, "saldo": saldo})


def _cfg(tipo, params, influencias):
    return TargetConfig(
        habilitado=True,
        nome="y",
        tipo=tipo,
        params=params,
        influencias=[Influencia(*i) for i in influencias],
    )


def test_binaria_atinge_prevalencia_pedida():
    cfg = _cfg(
        TargetTipo.BINARIA.value,
        {"prevalencia_base": 0.25, "ruido": 0.1, "forcar_prevalencia": True},
        [("saldo", -1, 0.8), ("renda", -1, 0.4)],
    )
    y, info = gerar_target(_df(), cfg, np.random.default_rng(1))
    assert 0.24 <= y.mean() <= 0.26
    assert info["prevalencia"] == pytest.approx(y.mean())


def test_binaria_sem_forcar_fica_proxima():
    cfg = _cfg(
        TargetTipo.BINARIA.value,
        {"prevalencia_base": 0.4, "ruido": 0.0, "forcar_prevalencia": False},
        [("saldo", 1, 0.6)],
    )
    y, info = gerar_target(_df(), cfg, np.random.default_rng(2))
    assert abs(y.mean() - 0.4) < 0.05
    assert set(np.unique(y)) == {0, 1}


def test_auc_cai_conforme_ruido_sobe():
    scores = []
    for ruido in (0.0, 0.3, 0.7, 1.0):
        cfg = _cfg(
            TargetTipo.BINARIA.value,
            {"prevalencia_base": 0.3, "ruido": ruido, "forcar_prevalencia": False},
            [("saldo", 1, 0.9)],
        )
        _, info = gerar_target(_df(), cfg, np.random.default_rng(3))
        scores.append(info["auc"])
    assert scores[0] > 0.85
    assert scores[0] > scores[1] > scores[-1]
    assert scores[-1] < 0.65


def test_sem_influencia_e_ruido_puro():
    cfg = _cfg(
        TargetTipo.BINARIA.value,
        {"prevalencia_base": 0.5, "ruido": 0.0, "forcar_prevalencia": False},
        [],
    )
    _, info = gerar_target(_df(), cfg, np.random.default_rng(4))
    assert 0.45 <= info["auc"] <= 0.55


def test_regressao_r2_cai_com_ruido():
    cfg = _cfg(TargetTipo.REGRESSAO.value, {"media_base": 0.0, "ruido": 0.1}, [("saldo", 1, 1.0)])
    y, info = gerar_target(_df(), cfg, np.random.default_rng(5))
    assert info["r2"] > 0.7

    cfg.params["ruido"] = 1.0
    y2, info2 = gerar_target(_df(), cfg, np.random.default_rng(5))
    assert info2["r2"] < 0.1


def test_contagem_positiva_com_media_proxima():
    cfg = _cfg(TargetTipo.CONTAGEM.value, {"lam_base": 4.0, "ruido": 0.2}, [("saldo", 1, 0.7)])
    y, info = gerar_target(_df(), cfg, np.random.default_rng(6))
    assert (y >= 0).all()
    assert np.all(y == np.rint(y))
    assert abs(y.mean() - 4.0) < 0.6


def test_multiclasse_proporcoes():
    cfg = _cfg(
        TargetTipo.MULTICLASSE.value,
        {
            "classes": [
                {"rotulo": "A", "proporcao": 0.5},
                {"rotulo": "B", "proporcao": 0.3},
                {"rotulo": "C", "proporcao": 0.2},
            ],
            "ruido": 0.2,
        },
        [("saldo", 1, 0.8)],
    )
    y, info = gerar_target(_df(), cfg, np.random.default_rng(7))
    assert set(np.unique(y)) == {0, 1, 2}
    assert len(info["rotulos"]) == 3
    assert sum(info["proporcoes_realizadas"]) == pytest.approx(1.0, abs=1e-6)


def test_ordinal_proporcoes_exatas():
    cfg = _cfg(
        TargetTipo.ORDINAL.value,
        {"rotulos": "baixo, medio, alto", "proporcoes": "0.5, 0.3, 0.2", "ruido": 0.2},
        [("saldo", 1, 0.8)],
    )
    y, info = gerar_target(_df(), cfg, np.random.default_rng(8))
    assert info["rotulos"] == ["baixo", "medio", "alto"]
    assert set(np.unique(y)) == {0, 1, 2}
    for realizado, pedido in zip(info["proporcoes_realizadas"], info["proporcoes"]):
        assert abs(realizado - pedido) < 0.02


def test_modo_regra_limiar():
    cfg = TargetConfig(
        habilitado=True,
        nome="y",
        tipo="binaria",
        modo="regra",
        regra="renda > 3000",
        params={"ruido": 0.0},
    )
    df = _df()
    y, info = gerar_target(df, cfg, np.random.default_rng(9))
    esperado = (df["renda"] > 3000).astype(int).to_numpy()
    assert np.array_equal(y, esperado)
    assert info["prevalencia"] == pytest.approx(esperado.mean())


def test_regra_com_ruido_altera_rotulos():
    cfg = TargetConfig(
        habilitado=True, nome="y", tipo="binaria", modo="regra", regra="renda > 3000", params={"ruido": 0.2}
    )
    df = _df()
    y, _ = gerar_target(df, cfg, np.random.default_rng(10))
    esperado = (df["renda"] > 3000).astype(int).to_numpy()
    assert 0.15 < (y != esperado).mean() < 0.25


def test_regra_invalida_falha_limpo():
    cfg = TargetConfig(habilitado=True, modo="regra", regra="__import__('os').system('ls')")
    with pytest.raises(ValueError):
        gerar_target(_df(), cfg, np.random.default_rng(11))
