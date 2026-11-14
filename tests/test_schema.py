"""Schema: serializacao e regras de validacao."""

from __future__ import annotations

from gerador.io.profile import carregar_schema, salvar_schema
from gerador.schema.dataset import DatasetConfig
from gerador.schema.relations import Relacao, RelacoesConfig
from gerador.schema.target import Influencia, TargetConfig
from gerador.schema.variable import VariableConfig

from tests.conftest import var


def test_roundtrip_json(config_basica):
    config_basica.relacoes = RelacoesConfig(lista=[Relacao("renda", "saldo", 0.6)])
    config_basica.target = TargetConfig(
        habilitado=True, influencias=[Influencia("saldo", -1, 0.7)]
    )
    texto = salvar_schema(config_basica)
    voltou = carregar_schema(texto)

    assert voltou.n_linhas == config_basica.n_linhas
    assert voltou.seed == config_basica.seed
    assert [v.nome for v in voltou.variaveis] == ["renda", "saldo", "num_tvs"]
    assert voltou.variaveis[0].params["mu"] == 3000.0
    assert voltou.relacoes.lista[0].forca == 0.6
    assert voltou.target.influencias[0].direcao == -1


def test_config_valida_sem_erros(config_basica):
    erros, avisos = config_basica.validar()
    assert erros == []


def test_nome_duplicado(config_basica):
    config_basica.variaveis.append(var("renda"))
    erros, _ = config_basica.validar()
    assert any("duplicado" in e for e in erros)


def test_nome_invalido(config_basica):
    config_basica.variaveis.append(var("1renda com espaco"))
    erros, _ = config_basica.validar()
    assert any("apenas letras" in e or "comecar com digito" in e for e in erros)


def test_relacao_com_target_e_rejeitada(config_basica):
    config_basica.target = TargetConfig(habilitado=True, nome="y")
    config_basica.relacoes = RelacoesConfig(lista=[Relacao("renda", "y", 0.5)])
    erros, _ = config_basica.validar()
    assert any("nao pode participar" in e for e in erros)


def test_matriz_nao_psd_e_detectada(config_basica):
    config_basica.variaveis.append(var("idade", params={"mu": 30, "sigma": 5}))
    config_basica.relacoes = RelacoesConfig(
        lista=[
            Relacao("renda", "saldo", 0.95),
            Relacao("saldo", "idade", 0.95),
            Relacao("renda", "idade", -0.95),
        ]
    )
    erros, _ = config_basica.validar()
    assert any("nao e valida" in e for e in erros)


def test_min_maior_que_max(config_basica):
    config_basica.variaveis[0].escala.minimo = 100
    config_basica.variaveis[0].escala.maximo = 10
    erros, _ = config_basica.validar()
    assert any("min maior que max" in e for e in erros)


def test_relacao_com_texto_e_rejeitada(config_basica):
    config_basica.variaveis.append(
        VariableConfig(
            nome="id_cliente",
            dtype="texto",
            distribuicao="id",
            params={"modo": "formatado", "prefixo": "C-", "n_digitos": 5},
        )
    )
    config_basica.relacoes = RelacoesConfig(lista=[Relacao("renda", "id_cliente", 0.5)])
    erros, _ = config_basica.validar()
    assert any("nao participa" in e for e in erros)


def test_mar_sem_dependencia(config_basica):
    config_basica.variaveis[0].qualidade.nulos.pct = 10
    config_basica.variaveis[0].qualidade.nulos.mecanismo = "mar"
    erros, _ = config_basica.validar()
    assert any("MAR exige" in e for e in erros)


def test_ordinal_exige_dois_itens(config_basica):
    v = VariableConfig(
        nome="escolaridade",
        dtype="ordinal",
        distribuicao="categorias",
        params={"itens": [{"rotulo": "medio", "peso": 1.0}]},
    )
    config_basica.variaveis.append(v)
    erros, _ = config_basica.validar()
    assert any("ao menos 2 itens" in e for e in erros)


def test_distribuicao_invalida_para_tipo(config_basica):
    v = VariableConfig(
        nome="quebrada", dtype="categorica", distribuicao="normal", params={"mu": 0, "sigma": 1}
    )
    config_basica.variaveis.append(v)
    erros, _ = config_basica.validar()
    assert any("nao e valida para o tipo" in e for e in erros)


def test_normal_em_contagem_gera_aviso(config_basica):
    config_basica.variaveis.append(
        var("filhos", dtype="numerica_discreta", distribuicao="normal", params={"mu": 2, "sigma": 1})
    )
    _, avisos = config_basica.validar()
    assert any("arredondado" in a for a in avisos)


def test_id_aleatorio_curto_e_rejeitado(config_basica):
    v = VariableConfig(
        nome="id",
        dtype="texto",
        distribuicao="id",
        params={"modo": "aleatorio", "comprimento": 3, "alfabeto": "numeric", "unico": True},
    )
    config_basica.variaveis.append(v)
    erros, _ = config_basica.validar()
    assert any("colisao" in e for e in erros)


def test_novo_projeto_vazio_tem_erro():
    erros, _ = DatasetConfig().validar()
    assert any("ao menos uma variavel" in e for e in erros)
