"""Exportacao e dicionario de dados."""

from __future__ import annotations

import io

import pandas as pd

from gerador.engine.pipeline import gerar_dataset
from gerador.io.export import dicionario_md, exportar


def _resultado(config_basica):
    config_basica.target.habilitado = False
    return gerar_dataset(config_basica)


def test_export_csv(config_basica):
    res = _resultado(config_basica)
    dados, nome, mime = exportar(res.df, "csv")
    assert nome == "dataset.csv"
    assert mime == "text/csv"
    lido = pd.read_csv(io.BytesIO(dados))
    assert list(lido.columns) == ["renda", "saldo", "num_tvs"]
    assert len(lido) == config_basica.n_linhas


def test_export_parquet(config_basica):
    res = _resultado(config_basica)
    dados, nome, _ = exportar(res.df, "parquet")
    assert nome == "dataset.parquet"
    lido = pd.read_parquet(io.BytesIO(dados))
    assert lido.shape == res.df.shape


def test_export_xlsx(config_basica):
    res = _resultado(config_basica)
    dados, nome, _ = exportar(res.df, "xlsx")
    assert nome == "dataset.xlsx"
    lido = pd.read_excel(io.BytesIO(dados))
    assert lido.shape[0] == config_basica.n_linhas


def test_dicionario_lista_variaveis(config_basica):
    texto = dicionario_md(config_basica)
    assert "renda" in texto
    assert "saldo" in texto
    assert "num_tvs" in texto
    assert f"{config_basica.n_linhas}" in texto
    assert "Target" in texto


def test_formato_invalido():
    import pytest

    with pytest.raises(ValueError):
        exportar(pd.DataFrame({"a": [1]}), "docx")
