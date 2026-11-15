"""Presets prontos: estrutura, validacao e dicionario."""

from __future__ import annotations

from gerador.engine.copula import matriz_relacoes, validar_psd
from gerador.io.export import dicionario_md
from gerador.io.profile import carregar_schema, salvar_schema
from gerador.presets import listar_presets, obter_preset
from gerador.schema.types import DType


def test_lista_de_presets():
    presets = listar_presets()
    assert "credito_mensal" in presets
    assert presets["credito_mensal"]


def test_preset_credito_valida_sem_erros():
    config = obter_preset("credito_mensal")
    erros, avisos = config.validar()
    assert erros == []
    assert any("completados automaticamente" in a for a in avisos)


def test_preset_credito_e_painel_de_12_meses():
    config = obter_preset("credito_mensal")
    assert config.painel.habilitado
    assert config.painel.coluna_id == "id_cliente"
    assert config.painel.coluna_tempo == "data_mes"
    assert config.painel.data_inicio == "2025-12-01"
    assert config.painel.n_meses == 12
    assert config.n_linhas == 6000
    assert config.total_linhas == 72_000


def test_preset_credito_nao_tem_target():
    config = obter_preset("credito_mensal")
    assert config.target.habilitado is False
    assert "churn" not in config.nomes


def test_preset_credito_tem_as_22_variaveis_na_ordem():
    config = obter_preset("credito_mensal")
    assert config.nomes == [
        "id_cliente",
        "idade",
        "sistema_celular",
        "limite",
        "renda",
        "score_bureau_A",
        "qtd_instituicoes",
        "contratos_ativos",
        "valor_a_vencer",
        "atraso_em_dias",
        "valor_entradas",
        "valor_saidas",
        "patrimonio",
        "UF",
        "ocupacao",
        "escolaridade",
        "score_bureau_B",
        "tempo_de_conta",
        "valor_vencido",
        "valor_prejuizo",
        "valor_refinanciado",
        "limite_total_instituicoes",
    ]


def test_derivadas_ficam_no_fim_e_tem_expressao():
    config = obter_preset("credito_mensal")
    derivadas = [v for v in config.variaveis if v.tipo is DType.DERIVADA]
    assert [v.nome for v in derivadas] == [
        "valor_vencido",
        "valor_prejuizo",
        "valor_refinanciado",
        "limite_total_instituicoes",
    ]
    assert config.variaveis[-4:] == derivadas
    assert all(v.expressao.strip() for v in derivadas)


def test_campos_que_nao_evoluem_ficam_estaticos():
    config = obter_preset("credito_mensal")
    estaticas = {
        "idade",
        "sistema_celular",
        "limite",
        "UF",
        "escolaridade",
    }
    for nome in estaticas:
        assert config.variavel(nome).evolucao.modo == "estatica"
    assert config.variavel("tempo_de_conta").evolucao.modo == "tendencia"
    assert config.variavel("renda").evolucao.modo == "ar1"
    assert config.variavel("qtd_instituicoes").evolucao.modo == "ar1"


def test_relacoes_nao_incluem_derivadas_nem_id():
    config = obter_preset("credito_mensal")
    proibidas = {"id_cliente", "valor_vencido", "valor_prejuizo",
                 "valor_refinanciado", "limite_total_instituicoes"}
    for rel in config.relacoes.lista:
        assert rel.a not in proibidas
        assert rel.b not in proibidas
    assert len(config.relacoes.lista) == 42


def test_matriz_completada_e_psd():
    config = obter_preset("credito_mensal")
    R, _ = matriz_relacoes(
        config.nomes_copulaveis,
        config.relacoes.lista,
        completar=config.relacoes.completar_psd,
    )
    ok, _ = validar_psd(R)
    assert ok


def test_roundtrip_preserva_painel_e_derivadas():
    config = obter_preset("credito_mensal")
    voltou = carregar_schema(salvar_schema(config))
    assert voltou.painel.to_dict() == config.painel.to_dict()
    assert voltou.relacoes.completar_psd is True
    assert [v.nome for v in voltou.variaveis] == config.nomes
    original = config.variavel("valor_refinanciado")
    copia = voltou.variavel("valor_refinanciado")
    assert copia.tipo is DType.DERIVADA
    assert copia.expressao == original.expressao
    assert voltou.variavel("renda").evolucao.to_dict() == {"modo": "ar1", "phi": 0.9, "passo": 1.0}


def test_dicionario_mostra_painel_e_expressao():
    texto = dicionario_md(obter_preset("credito_mensal"))
    assert "Clientes (entidades): 6000" in texto
    assert "Meses: 12" in texto
    assert "Linhas totais: 72000" in texto
    assert "2025-12-01 a 2026-11-01" in texto
    assert "sem lags" in texto or "nem agregados" in texto
    assert "expressao" in texto
    assert "valor_vencido" in texto
    assert "target: nenhum" in texto.lower() or "Target: nenhum" in texto


def test_preset_desconhecido_levanta_erro():
    import pytest

    with pytest.raises(KeyError):
        obter_preset("nao_existe")
