"""Fixtures compartilhadas."""

from __future__ import annotations

import numpy as np
import pytest

from gerador.schema.dataset import DatasetConfig
from gerador.schema.relations import Relacao, RelacoesConfig
from gerador.schema.target import Influencia, TargetConfig
from gerador.schema.variable import VariableConfig


def var(nome, dtype="numerica_continua", distribuicao="normal", params=None, **kwargs):
    v = VariableConfig(nome=nome, dtype=dtype, distribuicao=distribuicao, params=params or {})
    for chave, valor in kwargs.items():
        if chave in ("escala", "qualidade"):
            setattr(v, chave, valor)
    if not params:
        from gerador.schema.types import PARAMS_DEFAULT, Distribuicao

        v.params = dict(PARAMS_DEFAULT[Distribuicao(distribuicao)])
    return v


@pytest.fixture
def config_basica() -> DatasetConfig:
    return DatasetConfig(
        n_linhas=2000,
        seed=7,
        variaveis=[
            var("renda", params={"mu": 3000.0, "sigma": 800.0}),
            var("saldo", params={"mu": 1000.0, "sigma": 400.0}),
            var(
                "num_tvs",
                dtype="numerica_discreta",
                distribuicao="poisson",
                params={"lam": 2.0},
            ),
        ],
    )


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(123)
