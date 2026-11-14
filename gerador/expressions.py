"""Avaliador de expressoes seguro (whitelist via `ast`).

Nunca executa texto cru: a arvore e validada ANTES de compilar, o namespace nao
tem builtins e qualquer `Attribute`, `Subscript`, `Import` ou nome dunder e
rejeitado com mensagem clara.

Duas transformacoes acontecem antes da compilacao, porque semantica de `and`/
`or`/comparacao encadeada nao funciona em arrays numpy (`truth value is
ambiguous`):

    (a > 1) and (b < 2)   ->  logical_and(a > 1, b < 2)
    1 < a < 5             ->  logical_and(1 < a, a < 5)
"""

from __future__ import annotations

import ast

import numpy as np
import pandas as pd

FUNCOES = {
    "log": np.log1p,
    "exp": np.exp,
    "sqrt": np.sqrt,
    "abs": np.abs,
    "clip": np.clip,
    "sin": np.sin,
    "cos": np.cos,
    "round": np.round,
    "minimum": np.minimum,
    "maximum": np.maximum,
    "logical_and": np.logical_and,
    "logical_or": np.logical_or,
    "logical_not": np.logical_not,
}

#: Funcoes de ruido sao fechadas sobre o rng da geracao (nao vivem em FUNCOES).
FUNCOES_RUIDO = {"ruido_normal", "ruido_uniforme"}

_NOS_PERMITIDOS = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.BoolOp,
    ast.Compare,
    ast.Call,
    ast.Name,
    ast.Constant,
    ast.Load,
    ast.operator,
    ast.unaryop,
    ast.boolop,
    ast.cmpop,
)

MAX_NO = 200


def _permitido(no: ast.AST) -> bool:
    return isinstance(no, _NOS_PERMITIDOS)


def validar_expressao(expressao: str, colunas: list[str]) -> str | None:
    """Retorna mensagem de erro, ou None se a expressao for segura."""
    if not expressao or not expressao.strip():
        return "expressao vazia"
    try:
        arvore = ast.parse(expressao, mode="eval")
    except SyntaxError as exc:
        return f"sintaxe invalida: {exc.msg}"

    if sum(1 for _ in ast.walk(arvore)) > MAX_NO:
        return "expressao longa demais"

    nomes_ok = set(colunas) | set(FUNCOES) | FUNCOES_RUIDO

    for no in ast.walk(arvore):
        if not _permitido(no):
            return f"construcao nao permitida: {type(no).__name__}"
        if isinstance(no, ast.Name) and no.id not in nomes_ok:
            return f"nome desconhecido: '{no.id}'"
        if isinstance(no, ast.Call):
            if not isinstance(no.func, ast.Name):
                return "chamada invalida"
            if no.func.id not in FUNCOES and no.func.id not in FUNCOES_RUIDO:
                return f"funcao nao permitida: '{no.func.id}'"
            if no.keywords:
                return "argumentos nomeados nao sao permitidos"
            if len(no.args) > 3:
                return f"'{no.func.id}' recebe no maximo 3 argumentos"
    return None


class _Normalizador(ast.NodeTransformer):
    """Troca `and`/`or`/`not` e comparacoes encadeadas por chamadas numpy."""

    @staticmethod
    def _chamada(nome: str, args: list[ast.expr]) -> ast.expr:
        return ast.Call(
            func=ast.Name(id=nome, ctx=ast.Load()), args=list(args), keywords=[]
        )

    def visit_BoolOp(self, node: ast.BoolOp) -> ast.expr:  # noqa: N802
        self.generic_visit(node)
        nome = "logical_and" if isinstance(node.op, ast.And) else "logical_or"
        acumulado = node.values[0]
        for valor in node.values[1:]:
            acumulado = self._chamada(nome, [acumulado, valor])
        return ast.copy_location(acumulado, node)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> ast.expr:  # noqa: N802
        self.generic_visit(node)
        if isinstance(node.op, ast.Not):
            return ast.copy_location(self._chamada("logical_not", [node.operand]), node)
        return node

    def visit_Compare(self, node: ast.Compare) -> ast.expr:  # noqa: N802
        self.generic_visit(node)
        if len(node.ops) == 1:
            return node
        partes: list[ast.expr] = []
        esquerda: ast.expr = node.left
        for operador, direita in zip(node.ops, node.comparators):
            partes.append(
                ast.Compare(left=esquerda, ops=[operador], comparators=[direita])
            )
            esquerda = direita
        acumulado = partes[0]
        for parte in partes[1:]:
            acumulado = self._chamada("logical_and", [acumulado, parte])
        return ast.copy_location(acumulado, node)


def compilar(expressao: str, colunas: list[str]) -> ast.Expression:
    erro = validar_expressao(expressao, colunas)
    if erro:
        raise ValueError(erro)
    arvore = ast.parse(expressao, mode="eval")
    normalizada = _Normalizador().visit(arvore)
    return ast.fix_missing_locations(normalizada)


def avaliar(expressao: str, df: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    arvore = compilar(expressao, list(df.columns))

    def ruido_normal(mu=0.0, sigma=1.0) -> np.ndarray:
        return rng.normal(float(mu), float(sigma), size=len(df))

    def ruido_uniforme(a=0.0, b=1.0) -> np.ndarray:
        return rng.uniform(float(a), float(b), size=len(df))

    namespace: dict = {col: df[col].to_numpy() for col in df.columns}
    namespace.update(FUNCOES)
    namespace.update({"ruido_normal": ruido_normal, "ruido_uniforme": ruido_uniforme})

    codigo = compile(arvore, "<expressao>", "eval")
    resultado = eval(codigo, {"__builtins__": {}}, namespace)  # noqa: S307 (arvore validada)
    return np.asarray(resultado)
