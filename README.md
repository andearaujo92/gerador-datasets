# gerador-dados

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
.venv/bin/python -m pytest -q      # 107 testes
```

## O que dá para configurar

Cada variável tem quatro blocos:

| Bloco | Conteúdo |
|---|---|
| Identidade | nome, descrição (vai para o dicionário) |
| Tipo & geração | tipo de dado, distribuição e parâmetros |
| Escala & limites | transformação (log, raiz, z-score, min-max), multiplicador, offset, casas decimais, mínimo/máximo, permitir negativos |
| Qualidade | % de nulos (MCAR/MAR + representação), % de outliers (cauda/IQR/valor fixo) |

Além disso:

- **Relações**: lista de pares com força em Spearman (−1 a 1), com validação da matriz e
  calibração opcional.
- **Alvo**: binária, multiclasse, regressão, contagem ou ordinal; modo guiado (features
  influentes) ou modo regra (expressão).
- **Global**: número de linhas, seed, formato de export, schema JSON salvável/carregável.

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

## Como a correlação funciona

A força é declarada em **Spearman**, que mede associação monotônica e por isso vale para
qualquer tipo (contínua, contagem, categórica, data). O motor:

1. converte cada força para o coeficiente latente: `R = 2·sin(π·ρ/6)`;
2. monta a matriz e checa os autovalores;
3. se a matriz for inválida, projeta para a matriz válida mais próxima (Higham) — a UI
   avisa qual par é o mais inconsistente;
4. amostra uma normal multivariada, aplica a CDF normal e passa `u` pela **ppf** de cada
   variável (cópula gaussiana);
5. opcionalmente **calibra**: variáveis discretas/categóricas atenuam a correlação por
   causa dos empates, então a matriz é ajustada iterativamente até a tolerância.

O diagnóstico sempre mostra **pedido vs. realizado**, com status `ok` (±0,05),
`atencao` (±0,12) e `divergente`.

O alvo **não** participa da matriz: ele é derivado das features.

## Ordem do pipeline

```
marginais + cópula → transformar/escalar/arredondar/clipar → target
→ outliers → nulos → ordenar colunas (target por último) → ajustar dtypes
```

O alvo é gerado antes dos nulos de propósito: ele enxerga as features íntegras, como um
rótulo real seria construído.

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
- dicionário de dados em Markdown (tipo, distribuição, parâmetros, escala, nulos, outliers,
  descrição e influências do alvo);
- schema em JSON — reabre a configuração inteira, inclusive as chaves de widget.

## Estrutura

```
app.py                  # entrypoint (só composição)
gerador/
  schema/               # tipos, variável, relações, alvo, dataset + validação
  engine/               # marginals, copula, transforms, text, target, missing,
                        # outliers, pipeline, diagnostics
  expressions.py        # parser seguro de regras
  io/                   # export (csv/parquet/xlsx + dicionário) e schema JSON
ui/                     # state, sidebar, variable_card, relations, target_card, diagnostics
tests/                  # 107 testes (schema, marginais, cópula, alvo, pipeline, app)
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
