"""Geracao de identificadores (texto) com numpy puro.

Sem Faker: o caso de uso e chave de juncao. Custo desprezivel, determinismo
total e nenhuma dependencia extra.
"""

from __future__ import annotations

import string

import numpy as np

ALFABETOS: dict[str, str] = {
    "numeric": string.digits,
    "alpha": string.ascii_lowercase,
    "alnum": string.digits + string.ascii_lowercase,
}


def tamanho_alfabeto(nome: str) -> int:
    return len(ALFABETOS.get(nome, ALFABETOS["alnum"]))


def probabilidade_colisao(n: int, espaco: int) -> float:
    """Aproximacao do aniversario: 1 - exp(-n(n-1)/2A)."""
    if espaco <= 0:
        return 1.0
    return float(1.0 - np.exp(-(n * (n - 1)) / (2.0 * espaco)))


def _codigos(n: int, comprimento: int, alfabeto: str, rng: np.random.Generator) -> np.ndarray:
    chars = np.array(list(ALFABETOS.get(alfabeto, ALFABETOS["alnum"])))
    idx = rng.integers(0, len(chars), size=(n, comprimento))
    grade = chars[idx]
    return np.array(["".join(linha) for linha in grade], dtype=object)


def gerar_ids(
    n: int,
    rng: np.random.Generator,
    *,
    modo: str = "formatado",
    prefixo: str = "ID-",
    n_digitos: int = 5,
    comprimento: int = 8,
    alfabeto: str = "alnum",
    unico: bool = True,
) -> np.ndarray:
    """Gera n identificadores.

    sequencial -> 1..n          formatado -> prefixo + zero-padding
    aleatorio  -> alfanumerico de comprimento fixo
    """
    modo = modo or "formatado"

    if modo == "sequencial":
        return np.arange(1, n + 1, dtype="int64")

    if modo == "formatado":
        largura = max(int(n_digitos), len(str(n)))
        return np.array([f"{prefixo}{i:0{largura}d}" for i in range(1, n + 1)], dtype=object)

    if modo != "aleatorio":
        raise ValueError(f"modo de ID desconhecido: {modo}")

    comprimento = max(1, int(comprimento))
    _validar_colisao(n, comprimento, alfabeto, unico)

    codigos = _codigos(n, comprimento, alfabeto, rng)
    if not unico:
        return codigos

    # regera apenas as linhas colididas; nada de sufixo (quebraria o comprimento fixo)
    for _ in range(50):
        _, primeira, contagem = np.unique(codigos, return_index=True, return_counts=True)
        if not np.any(contagem > 1):
            return codigos
        repetidos = np.setdiff1d(np.arange(n), primeira)
        codigos[repetidos] = _codigos(len(repetidos), comprimento, alfabeto, rng)

    return codigos


def _validar_colisao(n: int, comprimento: int, alfabeto: str, unico: bool) -> None:
    if not unico:
        return
    espaco = tamanho_alfabeto(alfabeto) ** comprimento
    prob = probabilidade_colisao(n, espaco)
    if prob > 0.01:
        minimo = 1
        while tamanho_alfabeto(alfabeto) ** minimo < n**2 * 10:
            minimo += 1
        raise ValueError(
            f"{n} IDs de {comprimento} caracteres colidem com probabilidade {prob:.1%}. "
            f"Use ao menos {max(minimo, comprimento + 1)} caracteres, aumente o alfabeto "
            "ou desligue a unicidade."
        )
