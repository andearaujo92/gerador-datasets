"""Schema do dataset: configuracao, validacao e serializacao."""

from gerador.schema.dataset import DatasetConfig
from gerador.schema.relations import Relacao, RelacoesConfig
from gerador.schema.target import Influencia, TargetConfig
from gerador.schema.variable import (
    Escala,
    Nulos,
    Outliers,
    Qualidade,
    VariableConfig,
    nova_variavel,
)

__all__ = [
    "DatasetConfig",
    "Escala",
    "Influencia",
    "Nulos",
    "Outliers",
    "Qualidade",
    "Relacao",
    "RelacoesConfig",
    "TargetConfig",
    "VariableConfig",
    "nova_variavel",
]
