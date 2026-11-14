"""Salvar e carregar o schema em JSON."""

from __future__ import annotations

import json
from pathlib import Path

from gerador.schema.dataset import DatasetConfig


def salvar_schema(config: DatasetConfig) -> str:
    return json.dumps(config.to_dict(), indent=2, ensure_ascii=False)


def carregar_schema(fonte: str | bytes | Path) -> DatasetConfig:
    if isinstance(fonte, bytes):
        dados = json.loads(fonte.decode("utf-8"))
    elif isinstance(fonte, Path):
        dados = json.loads(fonte.read_text(encoding="utf-8"))
    else:
        dados = json.loads(fonte)
    return DatasetConfig.from_dict(dados)


def salvar_arquivo(config: DatasetConfig, caminho: str | Path) -> Path:
    caminho = Path(caminho)
    caminho.write_text(salvar_schema(config), encoding="utf-8")
    return caminho
