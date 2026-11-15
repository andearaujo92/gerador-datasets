# gerador-datasets

App Streamlit para construir **datasets sintéticos para machine learning**: você define
cada variável (tipo, distribuição, parâmetros, escala, nulos, outliers), as correlações
entre elas e um alvo derivado — e recebe o dataset final exportável, com dicionário de
dados e reprodutível por seed.

## Como rodar

```bash
# ambiente já criado em .venv
.venv/bin/streamlit run app.py
```

Abre em `http://localhost:8501`.

```bash
.venv/bin/python -m pytest -q      # 148 testes
```

## Modelos prontos

A barra lateral traz uma base pronta para começar

Carregar um modelo substitui o schema atual; o JSON salvo continua sendo a forma de
guardar o seu.

## O que dá para configurar

Cada variável tem quatro blocos:

| Bloco | Conteúdo |
|---|---|
| Identidade | nome, descrição (vai para o dicionário) |
| Tipo & geração | tipo de dado, distribuição e parâmetros |
| Escala & limites | transformação (log, raiz, z-score, min-max), multiplicador, offset, casas decimais, mínimo/máximo, permitir negativos |
| Qualidade | % de nulos (MCAR/MAR + representação), % de outliers (cauda/IQR/valor fixo) |

Além disso:

- **Relações**: lista de pares com força em Spearman (−1 a 1), com validação da matriz,
  calibração opcional e preenchimento automático dos pares não declarados.
- **Alvo**: binária, multiclasse, regressão, contagem ou ordinal; modo guiado (features
  influentes) ou modo regra (expressão).
- **Painel temporal** (opcional): uma linha por entidade por mês, com coluna de tempo e
  evolução por variável (estática, AR1 com persistência φ ou tendência mensal).
- **Global**: número de linhas (ou clientes, no painel), seed, formato de export, schema
  JSON salvável/carregável.

## Tipos e distribuições

| Tipo | Distribuições |
|---|---|
| Numérica contínua | normal, uniforme, log-normal, exponencial, gama, beta, triangular, empírica |
| Numérica discreta (contagem) | normal, uniforme, log-normal, exponencial, triangular, Poisson, binomial, binomial negativa, geométrica, empírica |
| Ordinal | categorias com pesos, empírica |
| Categórica nominal | categorias com pesos, empírica |
| Booleana | Bernoulli |
| Data / hora | intervalo de datas com tendência (uniforme, crescente, decrescente, sazonal) |
| Texto (ID) | sequencial, formatado (prefixo + zeros) ou alfanumérico aleatório — **numpy puro, sem Faker** |
| Derivada | expressão segura sobre as variáveis declaradas acima (ex.: `(atraso > 0) * valor_a_vencer`) |

## Como a correlação funciona

A força é declarada em **Spearman**, que mede associação monotônica e por isso vale para
qualquer tipo (contínua, contagem, categórica, data). O motor:

1. converte cada força para o coeficiente latente: `R = 2·sin(π·ρ/6)`;
2. monta a matriz e checa os autovalores;
3. se a matriz for inválida, você escolhe entre **completar** os pares não informados
   (matriz válida mais próxima, preservando os pares pedidos) ou projetar para a matriz
   válida mais próxima (Higham) — a UI avisa qual par é o mais inconsistente;
4. amostra uma normal multivariada, aplica a CDF normal e passa `u` pela **ppf** de cada
   variável (cópula gaussiana);
5. opcionalmente **calibra**: variáveis discretas/categóricas atenuam a correlação por
   causa dos empates, então a matriz é ajustada iterativamente até a tolerância.

O diagnóstico sempre mostra **pedido vs. realizado**, com status `ok` (±0,05),
`atencao` (±0,12) e `divergente`.

O alvo **não** participa da matriz: ele é derivado das features.

## Painel temporal (dados em formato longo)

Ligando **Painel temporal** na barra lateral, a base passa a ter uma linha por entidade
por mês (`id_cliente` × `data_mes`), em vez de uma linha por entidade. A coluna de tempo
é sempre o primeiro dia do mês e a base é balanceada.

Cada variável escolhe como evolui:

| Modo | Significado |
|---|---|
| Estática | o mesmo valor em todos os meses (idade, UF, limite) |
| AR1 | persistência mês a mês com autocorrelação `φ` (renda, scores, atraso) |
| Tendência | valor inicial + passo fixo por mês (tempo de conta) |

O motor usa `Z = √φ·B + √(1−φ)·W`, em que `B` é o traço permanente da entidade e `W` o
choque do mês. Isso produz persistência e transições realistas sem congelar a variável.
A matriz de correlação é compensada por `φ` para valer em cada mês; o diagnóstico confere
o pedido vs. realizado no primeiro mês e mostra a autocorrelação realizada por variável.

A base entrega **somente o estado em `t`**: sem lags, sem janelas móveis e sem agregados.
Isso é proposital — evita vazamento de futuro e deixa a engenharia de features com quem
conhece o problema.

## Ordem do pipeline

```
marginais + cópula (ou painel entidade × mês) → transformar/escalar/arredondar/clipar
→ derivadas (ordem de declaração) → target
→ outliers → nulos → ordenar colunas (data_mes após o id, target por último) → dtypes
```

O alvo é gerado antes dos nulos de propósito: ele enxerga as features íntegras, como um
rótulo real seria construído. No painel, as derivadas são recalculadas mês a mês.

## Alvo

| Tipo | Como é gerado |
|---|---|
| Binária | score logístico das features; intercepto resolvido por bisseção na amostra para bater a prevalência pedida; ruído de rótulo configurável |
| Multiclasse | softmax com alinhamento por classe e prior configurável |
| Regressão | soma ponderada + ruído gaussiano proporcional |
| Contagem | Poisson com média exposta ao score e λ base preservado |
| Ordinal | latente contínuo cortado por quantis — proporções exatas |

Modo **regra** aceita expressões como `(renda > 3000) and (compras_12m <= 2)`, avaliadas
por um interpretador `ast` com whitelist (sem `eval` de texto cru, sem acesso a atributos,
imports ou builtins). Funções: `log`, `exp`, `sqrt`, `abs`, `clip`, `sin`, `cos`, `round`,
`minimum`, `maximum`, `ruido_normal`, `ruido_uniforme`.

## Export

- CSV, Parquet ou Excel;
- dicionário de dados em Markdown (tipo, distribuição, parâmetros, escala, evolução,
  nulos, outliers, descrição, painel e expressões derivadas);
- schema em JSON — reabre a configuração inteira, inclusive as chaves de widget.

## Estrutura

```
app.py                  # entrypoint (só composição)
gerador/
  schema/               # tipos, variável, relações, alvo, painel, dataset + validação
  engine/               # marginals, copula, transforms, text, target, missing,
                        # outliers, painel, pipeline, diagnostics
  expressions.py        # parser seguro de regras e derivadas
  io/                   # export (csv/parquet/xlsx + dicionário) e schema JSON
  presets/              # modelos prontos (crédito mensal)
ui/                     # state, sidebar, variable_card, relations, target_card, diagnostics
tests/                  # 148 testes (schema, marginais, cópula, alvo, painel, presets, app)
docs/BASE-CREDITO.md    # documentação do preset de crédito mensal
docs/PLANO-TECNICO.md   # plano de implementação
```

## Desvios em relação ao plano técnico

Decididos durante a execução, por risco de dependência:

1. **dataclasses em vez de pydantic** — o schema é validado por regras explícitas
   (`DatasetConfig.validar()`), com mensagens melhores e uma dependência menos.
2. **gráficos nativos do Streamlit + heatmap HTML** em vez de plotly — o heatmap de
   correlação é gerado com CSS inline, sem matplotlib.
3. **códigos internos para categóricas** — a cópula trabalha com `0..k-1` e a ordem
   declarada pelo usuário, não com a ordem alfabética dos rótulos. Isso corrige um viés
   real: rankear strings alfabeticamente inverteria a correlação.

## Limites conhecidos (v2)

- Interações entre features no alvo (`renda × idade`);
- correlação condicional por grupo;
- alvo de sobrevivência (time-to-event);
- geração incremental para > 1M linhas.

## Licença

Distribuído sob a licença **MIT** — veja [LICENSE](LICENSE). Você pode usar, copiar,
modificar, publicar, distribuir, sublicenciar e vender cópias, inclusive em projetos
comerciais e fechados, desde que o aviso de copyright e a licença sejam mantidos. Fork é
bem-vindo.
