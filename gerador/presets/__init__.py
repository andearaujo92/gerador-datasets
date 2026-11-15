"""Presets prontos: configuracoes completas carregaveis pela UI."""

from __future__ import annotations

from gerador.presets.credito import credito_mensal
from gerador.schema.dataset import DatasetConfig

#: chave -> (rotulo exibido, fabrica)
PRESETS: dict[str, tuple[str, object]] = {
    "credito_mensal": (
        "Credito mensal (painel 12 meses, sem target)",
        credito_mensal,
    ),
}


def listar_presets() -> dict[str, str]:
    return {chave: rotulo for chave, (rotulo, _) in PRESETS.items()}


def obter_preset(chave: str) -> DatasetConfig:
    if chave not in PRESETS:
        raise KeyError(f"preset desconhecido: '{chave}'")
    _, fabrica = PRESETS[chave]
    return fabrica()  # type: ignore[operator]
