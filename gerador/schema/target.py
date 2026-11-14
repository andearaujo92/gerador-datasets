"""Configuracao do target."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from gerador.schema.types import TargetModo, TargetTipo


PARAMS_DEFAULT_TARGET: dict[str, dict] = {
    TargetTipo.BINARIA.value: {"prevalencia_base": 0.3, "ruido": 0.1, "forcar_prevalencia": True},
    TargetTipo.MULTICLASSE.value: {
        "classes": [
            {"rotulo": "A", "proporcao": 0.5},
            {"rotulo": "B", "proporcao": 0.3},
            {"rotulo": "C", "proporcao": 0.2},
        ],
        "ruido": 0.2,
    },
    TargetTipo.REGRESSAO.value: {"media_base": 0.0, "ruido": 0.4},
    TargetTipo.CONTAGEM.value: {"lam_base": 5.0, "ruido": 0.3},
    TargetTipo.ORDINAL.value: {
        "rotulos": "baixo, medio, alto",
        "proporcoes": "0.5, 0.3, 0.2",
        "ruido": 0.2,
    },
}


@dataclass
class Influencia:
    variavel: str
    direcao: int = 1
    forca: float = 0.5
    uid: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    def to_dict(self) -> dict:
        return {
            "variavel": self.variavel,
            "direcao": self.direcao,
            "forca": self.forca,
            "uid": self.uid,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Influencia":
        return cls(
            variavel=d.get("variavel", ""),
            direcao=int(d.get("direcao", 1)),
            forca=float(d.get("forca", 0.5)),
            uid=d.get("uid") or uuid.uuid4().hex[:8],
        )


@dataclass
class TargetConfig:
    habilitado: bool = False
    nome: str = "target"
    tipo: str = TargetTipo.BINARIA.value
    modo: str = TargetModo.GUIADO.value
    params: dict = field(default_factory=lambda: dict(PARAMS_DEFAULT_TARGET[TargetTipo.BINARIA.value]))
    influencias: list[Influencia] = field(default_factory=list)
    regra: str | None = None
    expressao: str | None = None
    permitir_nulos: bool = False

    def resetar_params(self) -> None:
        self.params = dict(PARAMS_DEFAULT_TARGET[self.tipo])

    def to_dict(self) -> dict:
        return {
            "habilitado": self.habilitado,
            "nome": self.nome,
            "tipo": self.tipo,
            "modo": self.modo,
            "params": dict(self.params),
            "influencias": [i.to_dict() for i in self.influencias],
            "regra": self.regra,
            "expressao": self.expressao,
            "permitir_nulos": self.permitir_nulos,
        }

    @classmethod
    def from_dict(cls, d: dict | None) -> "TargetConfig":
        d = d or {}
        return cls(
            habilitado=bool(d.get("habilitado", False)),
            nome=d.get("nome", "target"),
            tipo=d.get("tipo", TargetTipo.BINARIA.value),
            modo=d.get("modo", TargetModo.GUIADO.value),
            params=dict(d.get("params") or {}),
            influencias=[Influencia.from_dict(i) for i in d.get("influencias", [])],
            regra=d.get("regra"),
            expressao=d.get("expressao"),
            permitir_nulos=bool(d.get("permitir_nulos", False)),
        )
