"""Relacoes (correlacao) entre variaveis."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from gerador.schema.types import Medida


@dataclass
class Relacao:
    a: str
    b: str
    forca: float = 0.5
    medida: str = Medida.SPEARMAN.value
    uid: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    def to_dict(self) -> dict:
        return {
            "a": self.a,
            "b": self.b,
            "forca": self.forca,
            "medida": self.medida,
            "uid": self.uid,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Relacao":
        return cls(
            a=d.get("a", ""),
            b=d.get("b", ""),
            forca=float(d.get("forca", 0.5)),
            medida=d.get("medida", Medida.SPEARMAN.value),
            uid=d.get("uid") or uuid.uuid4().hex[:8],
        )


@dataclass
class RelacoesConfig:
    lista: list[Relacao] = field(default_factory=list)
    calibrar: bool = False
    tolerancia: float = 0.02
    max_iter: int = 20

    def to_dict(self) -> dict:
        return {
            "lista": [r.to_dict() for r in self.lista],
            "calibrar": self.calibrar,
            "tolerancia": self.tolerancia,
            "max_iter": self.max_iter,
        }

    @classmethod
    def from_dict(cls, d: dict | None) -> "RelacoesConfig":
        d = d or {}
        return cls(
            lista=[Relacao.from_dict(r) for r in d.get("lista", [])],
            calibrar=bool(d.get("calibrar", False)),
            tolerancia=float(d.get("tolerancia", 0.02)),
            max_iter=int(d.get("max_iter", 20)),
        )
