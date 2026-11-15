"""Configuracao de uma variavel e sua validacao."""

from __future__ import annotations

import copy
import uuid
from dataclasses import dataclass, field
from typing import Any

from gerador.schema.types import (
    DISTRIBUICOES_POR_TIPO,
    PARAMS_DEFAULT,
    DType,
    Distribuicao,
    MecanismoNulo,
    MetodoOutlier,
    ModoEvolucao,
    Transformacao,
)


@dataclass
class Escala:
    transformacao: str = Transformacao.NENHUMA.value
    multiplicador: float = 1.0
    offset: float = 0.0
    casas_decimais: int = 2
    minimo: float | None = None
    maximo: float | None = None
    permitir_negativo: bool = True

    def to_dict(self) -> dict:
        return {
            "transformacao": self.transformacao,
            "multiplicador": self.multiplicador,
            "offset": self.offset,
            "casas_decimais": self.casas_decimais,
            "min": self.minimo,
            "max": self.maximo,
            "permitir_negativo": self.permitir_negativo,
        }

    @classmethod
    def from_dict(cls, d: dict | None) -> "Escala":
        d = d or {}
        return cls(
            transformacao=d.get("transformacao", Transformacao.NENHUMA.value),
            multiplicador=float(d.get("multiplicador", 1.0)),
            offset=float(d.get("offset", 0.0)),
            casas_decimais=int(d.get("casas_decimais", 2)),
            minimo=d.get("min"),
            maximo=d.get("max"),
            permitir_negativo=bool(d.get("permitir_negativo", True)),
        )


@dataclass
class Nulos:
    pct: float = 0.0
    mecanismo: str = MecanismoNulo.MCAR.value
    depende_de: str | None = None
    representacao: str = "nan"

    def to_dict(self) -> dict:
        return {
            "pct": self.pct,
            "mecanismo": self.mecanismo,
            "depende_de": self.depende_de,
            "representacao": self.representacao,
        }

    @classmethod
    def from_dict(cls, d: dict | None) -> "Nulos":
        d = d or {}
        return cls(
            pct=float(d.get("pct", 0.0)),
            mecanismo=d.get("mecanismo", MecanismoNulo.MCAR.value),
            depende_de=d.get("depende_de"),
            representacao=d.get("representacao", "nan"),
        )


@dataclass
class Outliers:
    pct: float = 0.0
    metodo: str = MetodoOutlier.NENHUM.value
    intensidade: float = 3.0

    def to_dict(self) -> dict:
        return {"pct": self.pct, "metodo": self.metodo, "intensidade": self.intensidade}

    @classmethod
    def from_dict(cls, d: dict | None) -> "Outliers":
        d = d or {}
        return cls(
            pct=float(d.get("pct", 0.0)),
            metodo=d.get("metodo", MetodoOutlier.NENHUM.value),
            intensidade=float(d.get("intensidade", 3.0)),
        )


@dataclass
class Qualidade:
    nulos: Nulos = field(default_factory=Nulos)
    outliers: Outliers = field(default_factory=Outliers)

    def to_dict(self) -> dict:
        return {"nulos": self.nulos.to_dict(), "outliers": self.outliers.to_dict()}

    @classmethod
    def from_dict(cls, d: dict | None) -> "Qualidade":
        d = d or {}
        return cls(
            nulos=Nulos.from_dict(d.get("nulos")),
            outliers=Outliers.from_dict(d.get("outliers")),
        )


@dataclass
class Evolucao:
    """Comportamento da variavel ao longo dos meses de um painel.

    Só tem efeito quando `DatasetConfig.painel.habilitado`. Fora do painel a
    variavel e gerada uma unica vez (cross-section).
    """

    modo: str = ModoEvolucao.ESTATICA.value
    #: Persistencia mensal (autocorrelacao lag-1) para o modo `ar1`.
    phi: float = 0.9
    #: Incremento por mes para o modo `tendencia`.
    passo: float = 1.0

    def to_dict(self) -> dict:
        return {"modo": self.modo, "phi": self.phi, "passo": self.passo}

    @classmethod
    def from_dict(cls, d: dict | None) -> "Evolucao":
        d = d or {}
        return cls(
            modo=d.get("modo", ModoEvolucao.ESTATICA.value),
            phi=float(d.get("phi", 0.9)),
            passo=float(d.get("passo", 1.0)),
        )


@dataclass
class VariableConfig:
    nome: str
    dtype: str = DType.CONTINUA.value
    distribuicao: str = Distribuicao.NORMAL.value
    params: dict[str, Any] = field(default_factory=lambda: copy.deepcopy(PARAMS_DEFAULT[Distribuicao.NORMAL]))
    descricao: str = ""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    escala: Escala = field(default_factory=Escala)
    qualidade: Qualidade = field(default_factory=Qualidade)
    #: Expressao segura usada quando `dtype == derivada`.
    expressao: str = ""
    evolucao: Evolucao = field(default_factory=Evolucao)

    # -- conveniencias -------------------------------------------------
    @property
    def tipo(self) -> DType:
        return DType(self.dtype)

    @property
    def dist(self) -> Distribuicao:
        return Distribuicao(self.distribuicao)

    def resetar_geracao(self) -> None:
        """Ajusta distribuicao e params ao tipo atual (cascata da UI)."""
        permitidas = DISTRIBUICOES_POR_TIPO[self.tipo]
        if not permitidas:
            # Tipo derivada: nao usa distribuicao nem params.
            self.distribuicao = Distribuicao.NORMAL.value
            self.params = {}
            return
        if self.dist not in permitidas:
            self.distribuicao = permitidas[0].value
        self.params = copy.deepcopy(PARAMS_DEFAULT[self.dist])
        self.escala.permitir_negativo = self.tipo not in (DType.DISCRETA,)

    # -- serializacao --------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "nome": self.nome,
            "descricao": self.descricao,
            "dtype": self.dtype,
            "geracao": {"distribuicao": self.distribuicao, "params": copy.deepcopy(self.params)},
            "escala": self.escala.to_dict(),
            "qualidade": self.qualidade.to_dict(),
            "expressao": self.expressao,
            "evolucao": self.evolucao.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "VariableConfig":
        ger = d.get("geracao") or {}
        return cls(
            id=d.get("id") or uuid.uuid4().hex[:8],
            nome=d.get("nome", "var"),
            descricao=d.get("descricao", ""),
            dtype=d.get("dtype", DType.CONTINUA.value),
            distribuicao=ger.get("distribuicao", Distribuicao.NORMAL.value),
            params=copy.deepcopy(ger.get("params") or {}),
            escala=Escala.from_dict(d.get("escala")),
            qualidade=Qualidade.from_dict(d.get("qualidade")),
            expressao=d.get("expressao", ""),
            evolucao=Evolucao.from_dict(d.get("evolucao")),
        )


def nova_variavel(indice: int) -> VariableConfig:
    return VariableConfig(nome=f"var_{indice}")


# ---------------------------------------------------------------------
# validacao
# ---------------------------------------------------------------------


def _num(params: dict, chave: str, erros: list[str], *, positivo: bool = False) -> None:
    valor = params.get(chave)
    if not isinstance(valor, (int, float)) or isinstance(valor, bool):
        erros.append(f"parametro '{chave}' precisa ser numerico")
        return
    if positivo and valor <= 0:
        erros.append(f"parametro '{chave}' precisa ser > 0")


def _itens(params: dict, campo_valor: str, erros: list[str], minimo: int = 2) -> None:
    itens = params.get("itens")
    if not isinstance(itens, list) or len(itens) < minimo:
        erros.append(f"'{campo_valor}' precisa de ao menos {minimo} itens")
        return
    total = 0.0
    for item in itens:
        if not isinstance(item, dict) or item.get(campo_valor) is None:
            erros.append(f"item sem '{campo_valor}'")
            return
        peso = item.get("peso", 0)
        if not isinstance(peso, (int, float)) or peso <= 0:
            erros.append("todo peso precisa ser numerico e > 0")
            return
        total += peso
    if total <= 0:
        erros.append("soma dos pesos precisa ser > 0")


def validar_params(var: VariableConfig) -> list[str]:
    """Valida os parametros da distribuicao escolhida."""
    erros: list[str] = []
    d = var.dist
    p = var.params or {}

    if d is Distribuicao.NORMAL:
        _num(p, "mu", erros)
        _num(p, "sigma", erros, positivo=True)
    elif d is Distribuicao.UNIFORME:
        _num(p, "a", erros)
        _num(p, "b", erros)
        if isinstance(p.get("a"), (int, float)) and isinstance(p.get("b"), (int, float)) and p["b"] <= p["a"]:
            erros.append("uniforme: 'b' precisa ser maior que 'a'")
    elif d is Distribuicao.LOG_NORMAL:
        _num(p, "mu_log", erros)
        _num(p, "sigma_log", erros, positivo=True)
    elif d is Distribuicao.EXPONENCIAL:
        _num(p, "lam", erros, positivo=True)
    elif d is Distribuicao.GAMA:
        _num(p, "shape", erros, positivo=True)
        _num(p, "scale", erros, positivo=True)
    elif d is Distribuicao.BETA:
        _num(p, "alpha", erros, positivo=True)
        _num(p, "beta", erros, positivo=True)
    elif d is Distribuicao.TRIANGULAR:
        for chave in ("min", "moda", "max"):
            _num(p, chave, erros)
        if not erros:
            if not (p["min"] <= p["moda"] <= p["max"]):
                erros.append("triangular: precisa de min <= moda <= max")
            if p["max"] <= p["min"]:
                erros.append("triangular: 'max' precisa ser maior que 'min'")
    elif d is Distribuicao.POISSON:
        _num(p, "lam", erros, positivo=True)
    elif d is Distribuicao.BINOMIAL:
        _num(p, "n", erros, positivo=True)
        _num(p, "p", erros)
        if not erros and not (0 < p["p"] < 1):
            erros.append("binomial: 'p' precisa estar entre 0 e 1")
    elif d is Distribuicao.BINOMIAL_NEGATIVA:
        _num(p, "r", erros, positivo=True)
        _num(p, "p", erros)
        if not erros and not (0 < p["p"] <= 1):
            erros.append("binomial negativa: 'p' precisa estar em (0, 1]")
    elif d is Distribuicao.GEOMETRICA:
        _num(p, "p", erros)
        if not erros and not (0 < p["p"] <= 1):
            erros.append("geometrica: 'p' precisa estar em (0, 1]")
    elif d is Distribuicao.EMPIRICA:
        _itens(p, "valor", erros)
    elif d is Distribuicao.CATEGORIAS:
        _itens(p, "rotulo", erros)
    elif d is Distribuicao.BERNOULLI:
        _num(p, "p", erros)
        if not erros and not (0 <= p["p"] <= 1):
            erros.append("bernoulli: 'p' precisa estar entre 0 e 1")
    elif d is Distribuicao.INTERVALO_DATAS:
        if not p.get("inicio") or not p.get("fim"):
            erros.append("intervalo de datas: informe inicio e fim")
    elif d is Distribuicao.ID:
        modo = p.get("modo", "formatado")
        if modo not in {"sequencial", "formatado", "aleatorio"}:
            erros.append("ID: modo invalido")
        if modo == "formatado":
            prefixo = p.get("prefixo", "")
            if any(c not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-" for c in prefixo):
                erros.append("ID: prefixo aceita apenas letras, digitos, '-' e '_'")
            if int(p.get("n_digitos", 5)) < 1:
                erros.append("ID: 'n_digitos' precisa ser >= 1")
        if modo == "aleatorio":
            if int(p.get("comprimento", 8)) < 1:
                erros.append("ID: 'comprimento' precisa ser >= 1")
            if p.get("alfabeto", "alnum") not in {"numeric", "alpha", "alnum"}:
                erros.append("ID: alfabeto invalido")
    return erros


def validar_variavel(
    var: VariableConfig,
    n_linhas: int = 1000,
    colunas: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Retorna (erros, avisos) de uma variavel.

    `colunas` (opcional) e a lista de variaveis ja declaradas; serve para
    validar a expressao de variaveis derivadas e garantir que elas so
    referenciem colunas anteriores.
    """
    erros: list[str] = []
    avisos: list[str] = []
    nome = (var.nome or "").strip()

    if not nome:
        erros.append("nome da variavel nao pode ser vazio")
    elif not nome.replace("_", "").isalnum():
        erros.append(f"'{nome}': use apenas letras, digitos e '_'")
    elif nome[0].isdigit():
        erros.append(f"'{nome}': nome nao pode comecar com digito")

    if var.tipo is DType.DERIVADA:
        expressao = (var.expressao or "").strip()
        if not expressao:
            erros.append(f"'{nome}': variavel derivada exige uma expressao")
        elif colunas is not None:
            from gerador.expressions import validar_expressao

            erro = validar_expressao(expressao, colunas)
            if erro:
                erros.append(f"'{nome}': expressao invalida: {erro}")
    else:
        if var.dist not in DISTRIBUICOES_POR_TIPO[var.tipo]:
            erros.append(f"distribuicao '{var.distribuicao}' nao e valida para o tipo '{var.dtype}'")
        erros.extend(validar_params(var))

    e = var.escala
    if e.minimo is not None and e.maximo is not None and e.minimo > e.maximo:
        erros.append(f"'{nome}': min maior que max")
    if e.casas_decimais < 0 or e.casas_decimais > 6:
        erros.append(f"'{nome}': casas decimais precisa estar entre 0 e 6")

    ev = var.evolucao
    if ev.modo not in {m.value for m in ModoEvolucao}:
        erros.append(f"'{nome}': modo de evolucao invalido")
    if not 0.0 <= ev.phi <= 0.999:
        erros.append(f"'{nome}': phi da evolucao precisa estar entre 0 e 0.999")
    if var.tipo is DType.TEXTO and ev.modo != ModoEvolucao.ESTATICA.value:
        avisos.append(f"'{nome}': texto nao evolui; a coluna sera repetida em todos os meses")
    if var.tipo is DType.DERIVADA and ev.modo != ModoEvolucao.ESTATICA.value:
        avisos.append(f"'{nome}': evolucao e ignorada em variavel derivada")

    q = var.qualidade
    if not 0 <= q.nulos.pct <= 100:
        erros.append(f"'{nome}': percentual de nulos precisa estar entre 0 e 100")
    if q.nulos.mecanismo == MecanismoNulo.MAR.value and not q.nulos.depende_de:
        erros.append(f"'{nome}': mecanismo MAR exige uma variavel de dependencia")
    if not 0 <= q.outliers.pct <= 100:
        erros.append(f"'{nome}': percentual de outliers precisa estar entre 0 e 100")

    if var.tipo in (DType.DISCRETA, DType.BOOLEANA) and q.outliers.metodo != MetodoOutlier.NENHUM.value:
        avisos.append(f"'{nome}': outliers em variavel discreta/booleana raramente faz sentido")
    if var.tipo in (DType.TEXTO, DType.DERIVADA) and q.outliers.metodo != MetodoOutlier.NENHUM.value:
        avisos.append(f"'{nome}': outliers sao ignorados neste tipo")

    if var.tipo is not DType.DERIVADA and var.dist in (Distribuicao.NORMAL, Distribuicao.LOG_NORMAL) and var.tipo is DType.DISCRETA:
        avisos.append(
            f"'{nome}': {var.dist.value} + contagem gera decimais; sera arredondado e limitado a >= 0"
        )

    if var.dist is Distribuicao.ID and var.params.get("modo") == "aleatorio" and var.params.get("unico", True):
        from gerador.engine.text import probabilidade_colisao, tamanho_alfabeto

        tam = tamanho_alfabeto(var.params.get("alfabeto", "alnum")) ** int(var.params.get("comprimento", 8))
        prob = probabilidade_colisao(n_linhas, tam)
        if prob > 0.01:
            erros.append(
                f"'{nome}': comprimento {var.params.get('comprimento')} gera {prob:.1%} de colisao "
                "em IDs aleatorios — aumente o comprimento ou desligue 'unico'"
            )

    if var.dist is Distribuicao.ID and var.params.get("modo") == "formatado":
        digitos = int(var.params.get("n_digitos", 5))
        if n_linhas > 10**digitos:
            erros.append(
                f"'{nome}': {n_linhas} linhas nao cabem em {digitos} digitos do ID formatado"
            )

    return erros, avisos
