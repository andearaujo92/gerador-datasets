"""Cartao de configuracao de uma variavel."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import streamlit as st

from gerador.schema.types import (
    DISTRIBUICOES_POR_TIPO,
    PARAMS_DEFAULT,
    ROTULOS_DISTRIBUICAO,
    ROTULOS_DTYPE,
    ROTULOS_EVOLUCAO,
    DType,
    Distribuicao,
    MecanismoNulo,
    MetodoOutlier,
    ModoEvolucao,
    Transformacao,
)
from gerador.schema.variable import VariableConfig, validar_variavel
from ui.state import nome_livre, remover_variavel


def chave(var: VariableConfig, campo: str) -> str:
    return f"{var.id}.{campo}"


def chave_param(var: VariableConfig, campo: str) -> str:
    # a distribuicao entra na chave: trocar a distribuicao nao herda valores antigos
    return f"{var.id}.{var.distribuicao}.{campo}"


# ---------------------------------------------------------------------
# render principal
# ---------------------------------------------------------------------


def render_todas(config) -> None:
    st.subheader("Variaveis")
    st.caption(
        "Cada caixa define **tipo**, **distribuicao**, **escala**, **qualidade** e, "
        "opcionalmente, relacoes. Nada e gerado ate voce clicar em *Gerar dataset*."
    )

    if not config.variaveis:
        st.info("Nenhuma variavel ainda. Adicione a primeira abaixo.")

    for indice, var in enumerate(list(config.variaveis)):
        _render_variavel(var, config, indice)

    if st.button("Adicionar variavel", key="gd.add_var"):
        config.variaveis.append(VariableConfig(nome=nome_livre(config)))
        st.rerun()


def _render_variavel(var: VariableConfig, config, indice: int) -> None:
    erros, avisos = validar_variavel(
        var, config.n_linhas, colunas=list(config.nomes[:indice])
    )
    tipo_rotulo = ROTULOS_DTYPE.get(var.tipo, var.dtype)
    partes = [f"{indice + 1}. {var.nome or '(sem nome)'}", tipo_rotulo]
    if var.qualidade.nulos.pct > 0:
        partes.append(f"{var.qualidade.nulos.pct:g}% nulos")
    if var.qualidade.outliers.metodo != MetodoOutlier.NENHUM.value:
        partes.append(f"{var.qualidade.outliers.pct:g}% outliers")
    titulo = "  ·  ".join(partes)

    with st.expander(titulo, expanded=bool(erros)):
        esquerda, direita = st.columns(2, gap="large")

        with esquerda:
            st.markdown("**Identidade**")
            var.nome = st.text_input(
                "Nome da variavel", value=var.nome, key=chave(var, "nome")
            )
            var.descricao = st.text_input(
                "Descricao (vai para o dicionario)",
                value=var.descricao,
                key=chave(var, "descricao"),
            )

            st.markdown("**Tipo & geracao**")
            _render_tipo(var)
            _render_evolucao(var, config)

        with direita:
            _render_escala(var)
            _render_qualidade(var, config)

        if erros:
            for erro in erros:
                st.error(erro)
        if avisos:
            for aviso in avisos:
                st.warning(aviso)

        if st.button("Remover variavel", key=chave(var, "remover")):
            remover_variavel(config, var.id)
            st.rerun()


def _render_tipo(var: VariableConfig) -> None:
    tipos = list(DType)
    atual = tipos.index(var.tipo)
    novo = st.selectbox(
        "Tipo de dado",
        tipos,
        index=atual,
        format_func=lambda d: ROTULOS_DTYPE[d],
        key=chave(var, "dtype"),
    )
    if novo is not var.tipo:
        var.dtype = novo.value
        var.resetar_geracao()
        # descarta a escolha anterior de distribuicao (pode nao valer para o novo tipo)
        st.session_state.pop(chave(var, "dist"), None)

    if var.tipo is DType.DERIVADA:
        st.caption(
            "A expressao roda **depois** das variaveis base, na ordem da lista: "
            "so pode usar colunas declaradas acima desta."
        )
        var.expressao = st.text_area(
            "Expressao",
            value=var.expressao,
            key=chave(var, "expressao"),
            help=(
                "Ex.: (atraso_em_dias > 0) * valor_a_vencer * ruido_uniforme(0.1, 0.9). "
                "Funcoes: log, exp, sqrt, abs, clip, minimum, maximum, ruido_uniforme(a,b), "
                "ruido_uniforme() e operadores matematicos/comparacoes."
            ),
        )
        return

    permitidas = DISTRIBUICOES_POR_TIPO[var.tipo]
    idx = permitidas.index(var.dist) if var.dist in permitidas else 0
    nova = st.selectbox(
        "Distribuicao",
        permitidas,
        index=idx,
        format_func=lambda d: ROTULOS_DISTRIBUICAO[d],
        key=chave(var, "dist"),
    )
    if nova is not var.dist:
        var.distribuicao = nova.value
        var.params = copy.deepcopy(PARAMS_DEFAULT[nova])

    _render_params(var)


def _render_evolucao(var: VariableConfig, config) -> None:
    """Controles temporais; so existem quando o painel esta ligado."""
    if not config.painel.habilitado or var.tipo in (DType.TEXTO, DType.DERIVADA):
        return

    st.markdown("**Evolucao no painel**")
    modos = list(ModoEvolucao)
    try:
        atual = ModoEvolucao(var.evolucao.modo)
    except ValueError:
        atual = ModoEvolucao.ESTATICA
    escolhido = st.selectbox(
        "Modo",
        modos,
        index=modos.index(atual),
        format_func=lambda m: ROTULOS_EVOLUCAO[m],
        key=chave(var, "evolucao_modo"),
    )
    var.evolucao.modo = escolhido.value

    if escolhido is ModoEvolucao.AR1:
        var.evolucao.phi = float(
            st.slider(
                "Persistencia (phi)",
                min_value=0.0,
                max_value=0.99,
                value=float(min(max(var.evolucao.phi, 0.0), 0.99)),
                step=0.01,
                key=chave(var, "evolucao_phi"),
                help=(
                    "Autocorrelacao mes a mes: 0 = cada mes e um sorteio novo; "
                    "0,9 = o cliente tende a permanecer parecido com o mes anterior."
                ),
            )
        )
    elif escolhido is ModoEvolucao.TENDENCIA:
        var.evolucao.passo = float(
            st.number_input(
                "Passo por mes",
                value=float(var.evolucao.passo),
                step=1.0,
                key=chave(var, "evolucao_passo"),
                help="Quanto a variavel muda a cada mes (ex.: +1 para tempo de conta).",
            )
        )


# ---------------------------------------------------------------------
# parametros por distribuicao
# ---------------------------------------------------------------------


def _num(var: VariableConfig, campo: str, label: str, *, minimo=None, maximo=None,
         passo: float = 0.1, inteiro: bool = False) -> None:
    atual = var.params.get(campo, 0)
    chave_w = chave_param(var, campo)
    if inteiro:
        valor = int(round(float(atual)))
        if minimo is not None:
            valor = max(valor, int(minimo))
        if maximo is not None:
            valor = min(valor, int(maximo))
        var.params[campo] = st.number_input(
            label,
            value=valor,
            step=int(passo) or 1,
            min_value=int(minimo) if minimo is not None else None,
            max_value=int(maximo) if maximo is not None else None,
            key=chave_w,
        )
    else:
        valor = float(atual)
        if minimo is not None:
            valor = max(valor, float(minimo))
        if maximo is not None:
            valor = min(valor, float(maximo))
        var.params[campo] = st.number_input(
            label,
            value=valor,
            step=float(passo),
            min_value=float(minimo) if minimo is not None else None,
            max_value=float(maximo) if maximo is not None else None,
            key=chave_w,
        )


def _itens(var: VariableConfig, campo_valor: str, rotulo: str) -> None:
    itens = var.params.get("itens") or []
    numerico = campo_valor == "valor"
    if itens:
        base = pd.DataFrame(itens)
    elif numerico:
        base = pd.DataFrame({campo_valor: [0.0, 1.0], "peso": [1.0, 1.0]})
    else:
        base = pd.DataFrame({campo_valor: ["A", "B"], "peso": [1.0, 1.0]})
    if "peso" not in base.columns:
        base["peso"] = 1.0
    base = base[[campo_valor, "peso"]]
    if numerico:
        base[campo_valor] = pd.to_numeric(base[campo_valor], errors="coerce").astype(float)
        coluna_valor = st.column_config.NumberColumn(rotulo, required=True)
    else:
        base[campo_valor] = base[campo_valor].astype(str)
        coluna_valor = st.column_config.TextColumn(rotulo, required=True)
    editado = st.data_editor(
        base,
        num_rows="dynamic",
        key=chave_param(var, "itens"),
        column_config={
            campo_valor: coluna_valor,
            "peso": st.column_config.NumberColumn("peso", min_value=0.01, step=0.5, default=1.0),
        },
    )
    limpos: list[dict] = []
    for _, linha in editado.iterrows():
        valor = linha[campo_valor]
        if valor is None or (isinstance(valor, float) and np.isnan(valor)):
            continue
        if numerico:
            limpo = float(valor)
        else:
            limpo = str(valor).strip()
            if not limpo:
                continue
        peso = linha["peso"]
        if peso is None or (isinstance(peso, float) and np.isnan(peso)) or float(peso) <= 0:
            peso = 1.0
        limpos.append({campo_valor: limpo, "peso": float(peso)})
    if limpos:
        var.params["itens"] = limpos


def _render_params(var: VariableConfig) -> None:
    d = var.dist

    if d is Distribuicao.NORMAL:
        _num(var, "mu", "Media (mu)", passo=1.0)
        _num(var, "sigma", "Desvio padrao (sigma)", minimo=0.01, passo=0.1)
    elif d is Distribuicao.UNIFORME:
        _num(var, "a", "Minimo (a)", passo=1.0)
        _num(var, "b", "Maximo (b)", passo=1.0)
    elif d is Distribuicao.LOG_NORMAL:
        _num(var, "mu_log", "Media do log (mu_log)", passo=0.1)
        _num(var, "sigma_log", "Desvio do log (sigma_log)", minimo=0.01, passo=0.1)
    elif d is Distribuicao.EXPONENCIAL:
        _num(var, "lam", "Taxa (lambda)", minimo=0.001, passo=0.1)
    elif d is Distribuicao.GAMA:
        _num(var, "shape", "Forma", minimo=0.01, passo=0.1)
        _num(var, "scale", "Escala", minimo=0.01, passo=0.1)
    elif d is Distribuicao.BETA:
        _num(var, "alpha", "Alpha", minimo=0.01, passo=0.1)
        _num(var, "beta", "Beta", minimo=0.01, passo=0.1)
    elif d is Distribuicao.TRIANGULAR:
        _num(var, "min", "Minimo", passo=1.0)
        _num(var, "moda", "Moda", passo=1.0)
        _num(var, "max", "Maximo", passo=1.0)
    elif d is Distribuicao.POISSON:
        _num(var, "lam", "Media (lambda)", minimo=0.001, passo=0.5)
    elif d is Distribuicao.BINOMIAL:
        _num(var, "n", "Numero de tentativas (n)", minimo=1, passo=1, inteiro=True)
        _num(var, "p", "Probabilidade (p)", minimo=0.001, maximo=0.999, passo=0.05)
    elif d is Distribuicao.BINOMIAL_NEGATIVA:
        _num(var, "r", "Sucessos (r)", minimo=0.1, passo=1.0)
        _num(var, "p", "Probabilidade (p)", minimo=0.001, maximo=1.0, passo=0.05)
    elif d is Distribuicao.GEOMETRICA:
        _num(var, "p", "Probabilidade (p)", minimo=0.001, maximo=1.0, passo=0.05)
    elif d is Distribuicao.BERNOULLI:
        _num(var, "p", "Probabilidade de 1 (p)", minimo=0.0, maximo=1.0, passo=0.05)
    elif d is Distribuicao.EMPIRICA:
        st.caption("Amostra valores discretos respeitando os pesos (nao precisa ordenar).")
        _itens(var, "valor", "valor")
    elif d is Distribuicao.CATEGORIAS:
        st.caption("A ordem das linhas e a ordem da variavel (importante para ordinal).")
        _itens(var, "rotulo", "rotulo")
    elif d is Distribuicao.INTERVALO_DATAS:
        _render_datas(var)
    elif d is Distribuicao.ID:
        _render_id(var)


def _render_datas(var: VariableConfig) -> None:
    from datetime import date

    def _data(texto: str, padrao: date) -> date:
        try:
            return date.fromisoformat(str(texto)[:10])
        except (ValueError, TypeError):
            return padrao

    inicio = st.date_input(
        "Inicio", value=_data(var.params.get("inicio"), date(2024, 1, 1)), key=chave_param(var, "inicio")
    )
    fim = st.date_input(
        "Fim", value=_data(var.params.get("fim"), date(2025, 12, 31)), key=chave_param(var, "fim")
    )
    var.params["inicio"] = inicio.isoformat()
    var.params["fim"] = fim.isoformat()

    tendencias = {
        "nenhuma": "Uniforme no intervalo",
        "crescente": "Mais datas recentes",
        "decrescente": "Mais datas antigas",
        "sazonal": "Concentracao sazonal",
    }
    atual = var.params.get("tendencia", "nenhuma")
    opcoes = list(tendencias)
    var.params["tendencia"] = st.selectbox(
        "Tendencia",
        opcoes,
        index=opcoes.index(atual) if atual in opcoes else 0,
        format_func=lambda k: tendencias[k],
        key=chave_param(var, "tendencia"),
    )
    var.params["com_hora"] = st.checkbox(
        "Incluir hora", value=bool(var.params.get("com_hora", False)), key=chave_param(var, "com_hora")
    )


def _render_id(var: VariableConfig) -> None:
    modos = {
        "formatado": "Prefixo + sequencia com zeros a esquerda",
        "sequencial": "Inteiro 1..n",
        "aleatorio": "Alfanumerico aleatorio de comprimento fixo",
    }
    atual = var.params.get("modo", "formatado")
    opcoes = list(modos)
    modo = st.selectbox(
        "Formato do ID",
        opcoes,
        index=opcoes.index(atual) if atual in opcoes else 0,
        format_func=lambda k: modos[k],
        key=chave_param(var, "modo"),
    )
    var.params["modo"] = modo

    if modo == "formatado":
        var.params["prefixo"] = st.text_input(
            "Prefixo", value=str(var.params.get("prefixo", "ID-")), key=chave_param(var, "prefixo")
        )
        _num(var, "n_digitos", "Digitos", minimo=1, passo=1, inteiro=True)
    elif modo == "aleatorio":
        _num(var, "comprimento", "Comprimento", minimo=1, passo=1, inteiro=True)
        alfabetos = {
            "alnum": "a-z + 0-9",
            "numeric": "apenas digitos",
            "alpha": "apenas letras",
        }
        atual_alfa = var.params.get("alfabeto", "alnum")
        opcoes_alfa = list(alfabetos)
        var.params["alfabeto"] = st.selectbox(
            "Alfabeto",
            opcoes_alfa,
            index=opcoes_alfa.index(atual_alfa) if atual_alfa in opcoes_alfa else 0,
            format_func=lambda k: alfabetos[k],
            key=chave_param(var, "alfabeto"),
        )
        var.params["unico"] = st.checkbox(
            "Exigir unicidade",
            value=bool(var.params.get("unico", True)),
            key=chave_param(var, "unico"),
            help="Se a probabilidade de colisao passar de 1%, a geracao e recusada.",
        )


# ---------------------------------------------------------------------
# escala
# ---------------------------------------------------------------------


def _render_escala(var: VariableConfig) -> None:
    st.markdown("**Escala & limites**")
    if var.dtype not in (DType.CONTINUA.value, DType.DISCRETA.value):
        st.caption("Nao se aplica a este tipo.")
        return

    transformacoes = {
        Transformacao.NENHUMA: "Nenhuma",
        Transformacao.LOG: "Log(1+x)",
        Transformacao.RAIZ: "Raiz quadrada",
        Transformacao.ZSCORE: "Padronizar (z-score)",
        Transformacao.MINMAX: "Normalizar 0-1",
    }
    opcoes = list(transformacoes)
    try:
        atual = Transformacao(var.escala.transformacao)
    except ValueError:
        atual = Transformacao.NENHUMA
    var.escala.transformacao = st.selectbox(
        "Transformacao",
        opcoes,
        index=opcoes.index(atual) if atual in opcoes else 0,
        format_func=lambda t: transformacoes[t],
        key=chave(var, "transformacao"),
    ).value

    c1, c2, c3 = st.columns(3)
    with c1:
        var.escala.multiplicador = st.number_input(
            "Multiplicador",
            value=float(var.escala.multiplicador),
            step=0.1,
            key=chave(var, "multiplicador"),
        )
    with c2:
        var.escala.offset = st.number_input(
            "Soma (offset)",
            value=float(var.escala.offset),
            step=1.0,
            key=chave(var, "offset"),
        )
    with c3:
        inteiro = var.dtype == DType.DISCRETA.value
        var.escala.casas_decimais = (
            0
            if inteiro
            else int(
                st.number_input(
                    "Casas decimais",
                    value=int(var.escala.casas_decimais),
                    min_value=0,
                    max_value=6,
                    step=1,
                    key=chave(var, "casas"),
                )
            )
        )

    var.escala.permitir_negativo = st.checkbox(
        "Permitir valores negativos",
        value=bool(var.escala.permitir_negativo),
        key=chave(var, "negativos"),
    )

    usar_min = st.checkbox(
        "Definir minimo", value=var.escala.minimo is not None, key=chave(var, "usar_min")
    )
    if usar_min:
        var.escala.minimo = st.number_input(
            "Minimo",
            value=float(var.escala.minimo if var.escala.minimo is not None else 0.0),
            key=chave(var, "min"),
        )
    else:
        var.escala.minimo = None

    usar_max = st.checkbox(
        "Definir maximo", value=var.escala.maximo is not None, key=chave(var, "usar_max")
    )
    if usar_max:
        var.escala.maximo = st.number_input(
            "Maximo",
            value=float(var.escala.maximo if var.escala.maximo is not None else 1.0),
            key=chave(var, "max"),
        )
    else:
        var.escala.maximo = None


# ---------------------------------------------------------------------
# qualidade
# ---------------------------------------------------------------------


def _render_qualidade(var: VariableConfig, config) -> None:
    st.markdown("**Qualidade**")
    q = var.qualidade

    q.nulos.pct = float(
        st.slider(
            "Nulos (%)",
            min_value=0.0,
            max_value=100.0,
            value=float(q.nulos.pct),
            step=1.0,
            key=chave(var, "nulos_pct"),
        )
    )

    mecanismos = {
        MecanismoNulo.MCAR: "MCAR — aleatorio puro",
        MecanismoNulo.MAR: "MAR — depende de outra variavel",
    }
    opcoes = list(mecanismos)
    try:
        atual = MecanismoNulo(q.nulos.mecanismo)
    except ValueError:
        atual = MecanismoNulo.MCAR
    escolhido = st.selectbox(
        "Mecanismo do nulo",
        opcoes,
        index=opcoes.index(atual) if atual in opcoes else 0,
        format_func=lambda m: mecanismos[m],
        key=chave(var, "mecanismo"),
    )
    q.nulos.mecanismo = escolhido.value

    if escolhido is MecanismoNulo.MAR:
        candidatos = [n for n in config.nomes if n != var.nome]
        if candidatos:
            atual_dep = q.nulos.depende_de
            q.nulos.depende_de = st.selectbox(
                "Depende de",
                candidatos,
                index=candidatos.index(atual_dep) if atual_dep in candidatos else 0,
                key=chave(var, "depende_de"),
            )
        else:
            st.caption("Adicione outra variavel para usar MAR.")
            q.nulos.depende_de = None
    else:
        q.nulos.depende_de = None

    representacoes = {
        "nan": "NaN (vazio real)",
        "": "String vazia",
        "-1": "-1",
        "desconhecido": "'desconhecido'",
    }
    opcoes_rep = list(representacoes)
    atual_rep = q.nulos.representacao if q.nulos.representacao in representacoes else "nan"
    q.nulos.representacao = st.selectbox(
        "Representar nulo como",
        opcoes_rep,
        index=opcoes_rep.index(atual_rep),
        format_func=lambda k: representacoes[k],
        key=chave(var, "representacao"),
    )

    st.markdown("**Outliers**")
    if var.dtype not in (DType.CONTINUA.value, DType.DISCRETA.value):
        st.caption("Outliers so se aplicam a variaveis numericas.")
        q.outliers.metodo = MetodoOutlier.NENHUM.value
        q.outliers.pct = 0.0
        return

    metodos = {
        MetodoOutlier.NENHUM: "Nenhum",
        MetodoOutlier.CAUDA: "Cauda da distribuicao",
        MetodoOutlier.IQR: "Regra do IQR",
        MetodoOutlier.FIXO: "Valor fixo",
    }
    opcoes_m = list(metodos)
    try:
        atual_m = MetodoOutlier(q.outliers.metodo)
    except ValueError:
        atual_m = MetodoOutlier.NENHUM
    escolhido_m = st.selectbox(
        "Metodo",
        opcoes_m,
        index=opcoes_m.index(atual_m) if atual_m in opcoes_m else 0,
        format_func=lambda m: metodos[m],
        key=chave(var, "outlier_metodo"),
    )
    q.outliers.metodo = escolhido_m.value

    if escolhido_m is not MetodoOutlier.NENHUM:
        q.outliers.pct = float(
            st.slider(
                "Percentual de outliers",
                min_value=0.1,
                max_value=30.0,
                value=float(max(q.outliers.pct, 0.1)),
                step=0.1,
                key=chave(var, "outlier_pct"),
            )
        )
        q.outliers.intensidade = float(
            st.number_input(
                "Intensidade",
                value=float(q.outliers.intensidade),
                min_value=0.5,
                step=0.5,
                key=chave(var, "outlier_int"),
            )
        )
    else:
        q.outliers.pct = 0.0
