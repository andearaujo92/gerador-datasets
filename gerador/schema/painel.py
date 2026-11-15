"""Configuracao do painel temporal (uma linha por entidade x mes)."""

from __future__ import annotations

from dataclasses import dataclass

#: Limite duro de linhas geradas para nao estourar memoria no app.
LIMITE_LINHAS = 1_000_000
#: A partir daqui a UI avisa que a geracao pode demorar.
AVISO_LINHAS = 500_000


def normalizar_inicio(inicio: str) -> str:
    """Primeiro dia do mes de `inicio` em ISO (YYYY-MM-DD)."""
    import pandas as pd

    return pd.Timestamp(inicio).replace(day=1).date().isoformat()


def meses_do_painel(inicio: str, n_meses: int) -> list:
    """Lista de timestamps mensais, sempre no dia 1, comecando em `inicio`."""
    import pandas as pd

    base = pd.Timestamp(inicio).replace(day=1)
    return [base + pd.DateOffset(months=i) for i in range(int(n_meses))]


@dataclass
class PainelConfig:
    habilitado: bool = False
    coluna_id: str = "id_cliente"
    coluna_tempo: str = "data_mes"
    data_inicio: str = "2025-12-01"
    n_meses: int = 12

    def to_dict(self) -> dict:
        return {
            "habilitado": self.habilitado,
            "coluna_id": self.coluna_id,
            "coluna_tempo": self.coluna_tempo,
            "data_inicio": self.data_inicio,
            "n_meses": self.n_meses,
        }

    @classmethod
    def from_dict(cls, d: dict | None) -> "PainelConfig":
        d = d or {}
        return cls(
            habilitado=bool(d.get("habilitado", False)),
            coluna_id=d.get("coluna_id", "id_cliente"),
            coluna_tempo=d.get("coluna_tempo", "data_mes"),
            data_inicio=d.get("data_inicio", "2025-12-01"),
            n_meses=int(d.get("n_meses", 12)),
        )

    def total_linhas(self, n_entidades: int) -> int:
        if not self.habilitado:
            return int(n_entidades)
        return int(n_entidades) * int(self.n_meses)
