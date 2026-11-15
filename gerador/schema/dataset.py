"""Configuracao raiz do dataset: validacao cruzada e serializacao."""

from __future__ import annotations

from dataclasses import dataclass, field

from gerador.schema.painel import AVISO_LINHAS, LIMITE_LINHAS, PainelConfig, normalizar_inicio
from gerador.schema.relations import RelacoesConfig
from gerador.schema.target import TargetConfig
from gerador.schema.types import TIPOS_COPULAVEIS, DType, TargetModo, TargetTipo
from gerador.schema.variable import VariableConfig, validar_variavel

FORMATOS_EXPORT = ["csv", "parquet", "xlsx"]


@dataclass
class DatasetConfig:
    n_linhas: int = 1000
    seed: int = 42
    variaveis: list[VariableConfig] = field(default_factory=list)
    relacoes: RelacoesConfig = field(default_factory=RelacoesConfig)
    target: TargetConfig = field(default_factory=TargetConfig)
    painel: PainelConfig = field(default_factory=PainelConfig)
    formato_export: str = "csv"
    incluir_dicionario: bool = True
    version: int = 1

    # -- conveniencias -------------------------------------------------
    @property
    def nomes(self) -> list[str]:
        return [v.nome for v in self.variaveis]

    @property
    def nomes_copulaveis(self) -> list[str]:
        return [v.nome for v in self.variaveis if v.tipo in TIPOS_COPULAVEIS]

    @property
    def total_linhas(self) -> int:
        return self.painel.total_linhas(self.n_linhas)

    def variavel(self, nome: str) -> VariableConfig | None:
        for v in self.variaveis:
            if v.nome == nome:
                return v
        return None

    # -- serializacao --------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "dataset": {
                "n_linhas": self.n_linhas,
                "seed": self.seed,
                "export": {
                    "formato": self.formato_export,
                    "incluir_dicionario": self.incluir_dicionario,
                },
            },
            "painel": self.painel.to_dict(),
            "variaveis": [v.to_dict() for v in self.variaveis],
            "relacoes": self.relacoes.to_dict(),
            "target": self.target.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DatasetConfig":
        meta = d.get("dataset") or {}
        export = meta.get("export") or {}
        return cls(
            version=int(d.get("version", 1)),
            n_linhas=int(meta.get("n_linhas", 1000)),
            seed=int(meta.get("seed", 42)),
            formato_export=export.get("formato", "csv"),
            incluir_dicionario=bool(export.get("incluir_dicionario", True)),
            variaveis=[VariableConfig.from_dict(v) for v in d.get("variaveis", [])],
            relacoes=RelacoesConfig.from_dict(d.get("relacoes")),
            target=TargetConfig.from_dict(d.get("target")),
            painel=PainelConfig.from_dict(d.get("painel")),
        )

    # -- validacao -----------------------------------------------------
    def validar(self) -> tuple[list[str], list[str]]:
        """Retorna (erros, avisos). Erros impedem a geracao."""
        erros: list[str] = []
        avisos: list[str] = []

        if self.n_linhas < 1:
            erros.append("numero de linhas (entidades) precisa ser >= 1")

        if not self.variaveis:
            erros.append("adicione ao menos uma variavel")

        total = self.total_linhas
        if total > LIMITE_LINHAS:
            erros.append(
                f"{total:,} linhas no total excedem o limite de {LIMITE_LINHAS:,} — "
                "reduza as entidades ou os meses"
            )
        elif total > AVISO_LINHAS:
            avisos.append(f"{total:,} linhas exigem bastante memoria; considere reduzir")

        nomes: list[str] = []
        for var in self.variaveis:
            e, a = validar_variavel(var, self.n_linhas, colunas=list(nomes))
            erros.extend(e)
            avisos.extend(a)
            nomes.append(var.nome)

        duplicados = {n for n in nomes if nomes.count(n) > 1}
        for nome in sorted(duplicados):
            erros.append(f"nome duplicado: '{nome}'")

        if self.variaveis and self.target.habilitado:
            if self.target.nome in nomes:
                erros.append(f"o nome do target ('{self.target.nome}') colide com uma variavel")

        # -- painel
        erros.extend(self._validar_painel(nomes))

        # -- relacoes
        vistos: set[tuple[str, str]] = set()
        for rel in self.relacoes.lista:
            if rel.a == rel.b:
                erros.append(f"relacao com a mesma variavel: '{rel.a}'")
                continue
            for lado in (rel.a, rel.b):
                if lado not in nomes:
                    erros.append(f"relacao aponta para variavel inexistente: '{lado}'")
                else:
                    var_lado = self.variavel(lado)
                    if var_lado is not None and var_lado.tipo is DType.TEXTO:
                        erros.append(
                            f"'{lado}' e do tipo texto (ID) e nao participa da matriz de "
                            "correlacao — IDs sao chaves, nao features"
                        )
                    if var_lado is not None and var_lado.tipo is DType.DERIVADA:
                        erros.append(
                            f"'{lado}' e derivada de uma expressao: a correlacao vem da formula "
                            "e nao entra na matriz"
                        )
            if self.target.habilitado and self.target.nome in (rel.a, rel.b):
                erros.append(
                    f"o target ('{self.target.nome}') nao pode participar da matriz de correlacao "
                    "— ele e derivado das features"
                )
            if not -1.0 <= rel.forca <= 1.0:
                erros.append(f"forca da relacao {rel.a}-{rel.b} precisa estar entre -1 e 1")
            chave = tuple(sorted((rel.a, rel.b)))
            if chave in vistos:
                avisos.append(f"relacao duplicada {rel.a}-{rel.b}: vale a ultima")
            vistos.add(chave)

        # -- matriz PSD (so variaveis copulaveis)
        if len(self.relacoes.lista) >= 2 and not erros:
            from gerador.engine.copula import matriz_relacoes, projetar_psd, validar_psd

            R, indices = matriz_relacoes(self.nomes_copulaveis, self.relacoes.lista)
            ok, min_eig = validar_psd(R)
            if not ok:
                if self.relacoes.completar_psd:
                    avisos.append(
                        "as correlacoes pedidas nao fecham sozinhas: os pares nao "
                        "informados serao completados automaticamente para uma matriz valida"
                    )
                else:
                    culpados = _par_mais_inconsistente(R, indices)
                    detalhe = f" (par mais inconsistente: {culpados})" if culpados else ""
                    erros.append(
                        "a matriz de correlacao pedida nao e valida "
                        f"(menor autovalor {min_eig:.3f}){detalhe} — ative "
                        "'Completar matriz automaticamente' ou ajuste as forcas"
                    )

        # -- target
        t = self.target
        if t.habilitado:
            if not t.nome or not t.nome.replace("_", "").isalnum():
                erros.append("nome do target invalido")
            if t.tipo not in {x.value for x in TargetTipo}:
                erros.append(f"tipo de target invalido: '{t.tipo}'")
            if t.modo == TargetModo.REGRA.value:
                if not t.regra:
                    erros.append("modo 'regra' exige uma expressao de regra")
                else:
                    from gerador.expressions import validar_expressao

                    err = validar_expressao(t.regra, nomes)
                    if err:
                        erros.append(f"regra do target: {err}")
            else:
                if not t.influencias:
                    avisos.append("target sem nenhuma feature influente: sera puro ruido (AUC ~0.5)")
                for inf in t.influencias:
                    if inf.variavel not in nomes:
                        erros.append(f"influencia aponta para variavel inexistente: '{inf.variavel}'")
                    if not 0 <= inf.forca <= 1:
                        erros.append(f"forca da influencia '{inf.variavel}' precisa estar entre 0 e 1")
                    if inf.direcao not in (-1, 0, 1):
                        erros.append(f"direcao da influencia '{inf.variavel}' precisa ser -1, 0 ou 1")

        # -- dependencia de MAR
        for var in self.variaveis:
            dep = var.qualidade.nulos.depende_de
            if dep and dep not in nomes:
                erros.append(f"'{var.nome}': dependencia de MAR aponta para '{dep}', que nao existe")

        # -- aviso de sinal degradado
        if t.habilitado and t.modo == TargetModo.GUIADO.value:
            for inf in t.influencias:
                var = self.variavel(inf.variavel)
                if var and inf.forca >= 0.5 and var.qualidade.nulos.pct >= 30:
                    avisos.append(
                        f"'{inf.variavel}' tem {var.qualidade.nulos.pct:.0f}% de nulos e forca "
                        f"{inf.forca:.1f} no target — o sinal aprendivel vai degradar"
                    )

        return erros, avisos

    def _validar_painel(self, nomes: list[str]) -> list[str]:
        p = self.painel
        if not p.habilitado:
            return []
        erros: list[str] = []

        if p.n_meses < 2:
            erros.append("painel: use ao menos 2 meses")
        if p.n_meses > 120:
            erros.append("painel: no maximo 120 meses")
        try:
            normalizar_inicio(p.data_inicio)
        except (ValueError, TypeError):
            erros.append(f"painel: data de inicio invalida ('{p.data_inicio}')")

        if not p.coluna_id or p.coluna_id not in nomes:
            erros.append(f"painel: a coluna de id '{p.coluna_id}' nao existe")
        else:
            var_id = self.variavel(p.coluna_id)
            if var_id is not None and var_id.tipo is not DType.TEXTO:
                erros.append(
                    f"painel: a coluna de id '{p.coluna_id}' precisa ser do tipo texto (ID)"
                )

        tempo = (p.coluna_tempo or "").strip()
        if not tempo:
            erros.append("painel: informe o nome da coluna de tempo")
        elif not tempo.replace("_", "").isalnum() or tempo[0].isdigit():
            erros.append(f"painel: nome de coluna de tempo invalido ('{tempo}')")
        elif tempo in nomes:
            erros.append(f"painel: a coluna de tempo '{tempo}' colide com uma variavel")
        elif self.target.habilitado and tempo == self.target.nome:
            erros.append(f"painel: a coluna de tempo '{tempo}' colide com o target")
        elif tempo == p.coluna_id:
            erros.append("painel: coluna de tempo e coluna de id precisam ser diferentes")

        return erros


def _par_mais_inconsistente(R, indices: dict[str, int]) -> str | None:
    """Aponta o par cujo valor mais destoa da matriz projetada (heuristica)."""
    import numpy as np

    try:
        from gerador.engine.copula import projetar_psd

        Rp = projetar_psd(R)
        diff = np.abs(R - Rp)
        np.fill_diagonal(diff, 0.0)
        i, j = np.unravel_index(int(np.argmax(diff)), diff.shape)
        nomes = {v: k for k, v in indices.items()}
        if i == j:
            return None
        return f"{nomes[int(i)]} x {nomes[int(j)]} (delta {diff[i, j]:.2f})"
    except Exception:  # pragma: no cover - diagnostico best-effort
        return None
