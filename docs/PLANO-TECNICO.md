# Plano de Implementação — gerador-datasets

App Streamlit para construir datasets sintéticos para machine learning: o usuário define
um schema de variáveis (tipo, distribuição, parâmetros, escala, nulos), relações de
correlação entre elas e um target derivado, e recebe o dataset final exportável e
reprodutível.

**Status:** plano aprovado, aguardando execução.
**Origem:** sessão de definição (questioning + desenho do catálogo de características).

---

## 0. Decisões travadas

Fechadas. Agentes downstream não devem reabri-las.

| Decisão | Escolha | Por quê |
|---|---|---|
| Motor de amostragem | interface única `Marginal.ppf(u)` | cópula funciona para todos os tipos sem caso especial; distribuição nova = 3 linhas |
| Correlação | cópula gaussiana, força em **Spearman** | invariante a transformação monótona; Pearson só vale entre normais |
| Conversão | `R = 2·sen(π·ρ/6)` | fechada e exata para contínuas |
| Matriz inválida | projeção de Higham + confirmação do usuário | nunca alterar os valores pedidos em silêncio |
| Target | derivado, **fora** da matriz de correlação | evita especificar uma coisa e gerar outra |
| Intercepto do target | root-finding na amostra | acerta a prevalência realizada, não só a esperada |
| Ordem do pipeline | amostrar → transformar → **target** → outliers → nulos | target enxerga features íntegras |
| Texto/ID | **numpy puro** (sequencial, formatado ou alfanumérico) | sem Faker: dependência a menos e ~10x mais rápido; o caso de uso é chave, não dado pessoal |
| Estado da UI | `st.session_state["config"]`, keys por `id` | rerun do Streamlit não pode perder estado |
| Lib de ML | sem `sklearn` | AUC/R² em numpy; menos dependência |
| Módulos | `schema` / `engine` / `ui` separados | motor testável sem Streamlit |

---

## 1. Escopo do v1

### Entra

| # | Capacidade |
|---|---|
| 1 | Catálogo de variáveis com 4 blocos: Identidade, Tipo & Geração, Escala & Limites, Qualidade |
| 2 | 7 tipos: contínua, discreta, ordinal, categórica, booleana, data/hora, texto (ID) |
| 3 | ~17 distribuições com parâmetros condicionais ao tipo |
| 4 | Correlação entre variáveis via lista de relações + cópula gaussiana + validação PSD + calibração |
| 5 | Target derivado: binária, multiclasse, regressão, contagem, ordinal — 2 modos (guiado, regra) |
| 6 | Nulos (MCAR/MAR) e outliers com percentual configurável |
| 7 | Preview por variável + aba de Diagnóstico (pedido vs. realizado) |
| 8 | Export CSV / Parquet / Excel + dicionário de dados |
| 9 | Salvar/carregar schema em JSON + seed reprodutível |

### Não entra (v2)

- Interações entre features (`renda × idade`) na geração do target
- Correlação condicional por grupo (relação diferente dentro de cada classe)
- Target de sobrevivência (time-to-event)
- Geração incremental / streaming para n > 1M
- Modo expressão do target (parser fica pronto, mas atrás de flag desligada)

---

## 2. Stack

| Camada | Escolha | Motivo |
|---|---|---|
| UI | `streamlit` | pedido explícito, simples |
| Dados | `numpy`, `pandas` | vetorizado, rápido até 1M linhas |
| Estatística | `scipy.stats` | ppf exato por distribuição, sem reimplementar |
| Schema/validação | `pydantic` v2 | union discriminada por dtype, validação + serialização JSON |
| Gráficos | `plotly` | interativo, heatmap nativo, funciona no Streamlit |
| Parquet / Excel | `pyarrow`, `openpyxl` | export |
| Testes | `pytest` | — |
| Deps/ambiente | `uv` + `pyproject.toml` | reprodutível |

**Sem `sklearn`**: AUC, R² e correlações são ~10 linhas de numpy/pandas. Evita dependência pesada.

---

## 3. Estrutura de arquivos

```
gerador-datasets/
├── app.py                        # entrypoint Streamlit (só composição)
├── pyproject.toml
├── README.md
├── .streamlit/config.toml        # tema, limites de upload
├── gerador/
│   ├── __init__.py
│   ├── schema/
│   │   ├── types.py              # DType, Distribuicao, MecanismoNulo, MetodoOutlier, ...
│   │   ├── variable.py           # VariableConfig + Geracao (union discriminada)
│   │   ├── target.py             # TargetConfig, Influencia
│   │   ├── relations.py          # Relacao, RelacoesConfig
│   │   └── dataset.py            # DatasetConfig (raiz)
│   ├── engine/
│   │   ├── marginals.py          # registro de marginais: ppf(u) por distribuição
│   │   ├── copula.py             # matriz, PSD, cópula gaussiana, calibração
│   │   ├── transforms.py         # log/raiz/zscore/minmax, escala, arredondar, clipar
│   │   ├── target.py             # geração de y (guiado, regra, expressão)
│   │   ├── missing.py            # MCAR / MAR + representação
│   │   ├── outliers.py           # cauda / IQR / valor fixo
│   │   ├── pipeline.py           # orquestra a ordem fixa 1..6
│   │   ├── diagnostics.py        # correlação realizada, AUC, R², prevalência
│   │   └── text.py               # IDs: sequencial, formatado, alfanumérico (numpy)
│   ├── expressions.py            # parser seguro via `ast` + whitelist
│   └── io/
│       ├── export.py             # csv/parquet/xlsx + dicionário (.md)
│       └── profile.py            # salvar/carregar schema.json
├── ui/
│   ├── state.py                  # helpers de st.session_state + hash da config
│   ├── sidebar.py                # dataset-level: n_linhas, seed, export, salvar/carregar
│   ├── variable_card.py          # blocos 1–4 + preview
│   ├── relations.py              # bloco 5 + heatmap
│   ├── target_card.py            # bloco 6
│   └── diagnostics.py            # aba de diagnóstico
└── tests/
    ├── test_schema.py
    ├── test_marginals.py
    ├── test_copula.py
    ├── test_target.py
    ├── test_pipeline.py
    ├── test_expressions.py
    └── test_export.py
```

---

## 4. Contrato central: `DatasetConfig` (JSON)

O JSON é o contrato entre UI, motor e disco. A UI nunca chama o motor com argumentos
soltos — sempre com um `DatasetConfig` validado.

```json
{
  "version": 1,
  "dataset": {
    "n_linhas": 1000,
    "seed": 42,
    "ordem_colunas": ["renda", "saldo", "num_tvs", "escolaridade", "estado_civil"],
    "export": { "formato": "csv", "incluir_dicionario": true }
  },
  "variaveis": [
    {
      "id": "v1",
      "nome": "saldo",
      "descricao": "Saldo em conta (R$)",
      "dtype": "numerica_continua",
      "geracao": {
        "distribuicao": "log_normal",
        "params": { "mu_log": 8.0, "sigma_log": 1.2 }
      },
      "escala": {
        "transformacao": "nenhuma",
        "multiplicador": 1000.0,
        "offset": 0.0,
        "casas_decimais": 2,
        "min": 0.0,
        "max": null,
        "permitir_negativo": false
      },
      "qualidade": {
        "nulos": {
          "pct": 5.0,
          "mecanismo": "mcar",
          "depende_de": null,
          "representacao": "nan"
        },
        "outliers": { "pct": 0.0, "metodo": "nenhum", "intensidade": 3.0 }
      }
    }
  ],
  "relacoes": {
    "lista": [
      { "a": "saldo", "b": "renda", "forca": 0.72, "medida": "spearman" }
    ],
    "calibrar": true,
    "tolerancia": 0.02,
    "max_iter": 20
  },
  "target": {
    "habilitado": true,
    "nome": "inadimplente",
    "tipo": "binaria",
    "modo": "guiado",
    "params": { "prevalencia_base": 0.25, "ruido": 0.35, "forcar_prevalencia": true },
    "influencias": [
      { "variavel": "saldo",  "direcao": -1, "forca": 0.8 },
      { "variavel": "renda",  "direcao": -1, "forca": 0.5 },
      { "variavel": "num_tvs","direcao":  1, "forca": 0.2 }
    ],
    "regra": null,
    "expressao": null,
    "permitir_nulos": false
  }
}
```

### Regras de modelagem

- `geracao` é uma **union discriminada por `dtype`** no pydantic. `params` é tipado por
  distribuição — `NormalParams(mu, sigma)`, `PoissonParams(lam)`, `CategoricaParams(itens)`,
  `TextoParams(modo, prefixo, n_digitos, comprimento, alfabeto)`, etc.
- `id` é estável (uuid curto) e sobrevive a renomear; `relacoes` e `influencias` referenciam
  **`nome`**, e o validador rejeita nome órfão ou duplicado.
- `target` **não** aparece em `relacoes`. Validador erro se aparecer.

---

## 5. Motor de geração

### 5.1 Abstração unificadora

Toda marginal implementa uma única interface:

```python
class Marginal(Protocol):
    nome: str
    def ppf(self, u: np.ndarray) -> np.ndarray: ...   # u ∈ [0,1) -> valor
```

Isso é o que permite a cópula funcionar para **todos** os tipos sem caso especial:

- contínua → `scipy.stats.<dist>.ppf`
- discreta/contagem → `scipy.stats.poisson.ppf` (produz os empates corretos)
- categórica/ordinal → `np.searchsorted(np.cumsum(pesos), u)`
- booleana → `u > 1 - p`
- data/hora → `inicio + ppf_uniforme(u) * (fim - inicio)` com tendência aplicada por
  reordenação dos uniformes (não destrói a correlação)
- empírica → `np.interp(u, np.cumsum(pesos_norm), valores)`
- texto/ID → gerado fora da cópula (não tem ordinalidade); ver §5.7

> Essa interface é a decisão arquitetural mais importante do projeto. Qualquer
> distribuição nova custa uma função de 3 linhas.

### 5.2 Ordem fixa do pipeline

```
1. amostrar marginais (independentes)          -> DataFrame
2. aplicar cópula (se houver relações)         -> reordena para a correlação alvo
3. transformações + escala + arredondar + clip -> respeita escala/limites
4. gerar target a partir das features limpas   -> coluna y
5. aplicar outliers
6. aplicar nulos
7. reordenar colunas (target por último)
```

Passos 5 e 6 vêm depois do target de propósito: o target enxerga features íntegras.
`diagnostics` mede o resultado final e reporta a degradação.

### 5.3 Cópula gaussiana

```
para cada par (i,j) com forca rho_s:
    R[i][j] = 2 * sin(pi * rho_s / 6)      # Spearman -> latente (fechado)
R = completar_diagonal(R)
R = validar_psd(R)  # se min(eigvals) < -1e-8 -> projetar_higham(R)
Z = rng.multivariate_normal(zeros, R, size=n)
U = norm.cdf(Z)
X[:, j] = marginal_j.ppf(U[:, j])
```

- **Por que Spearman e não Pearson:** invariante a transformação monótona. Funciona com
  log-normal, ordinal e categórica. Pearson só é interpretável entre duas normais.
- **Projeção PSD (Higham):** `w,V = eigh(R)`; `w = clip(w, 0, None)`; `R' = V@diag(w)@V.T`;
  renormalizar diagonal para 1. Se `R'` mudou, a UI mostra o Δ e exige confirmação antes
  de aplicar.
- **Calibração:** quando há discretas/categóricas, os empates atenuam o realizado.
  Loop: gera → mede Spearman realizado → `R_ij += (desejado - realizado) * 0.5` →
  reprojeta PSD → repete até `|Δ| ≤ tolerancia` ou `max_iter`. Falha de convergência é
  reportada como aviso, nunca como exceção.

### 5.4 Target

Todos os modos geram a partir das features **padronizadas** (`z`), com o RNG da seed.

**Binária (guiado)**

```
s     = b0 + sum(forca_i * direcao_i * z_i)
b0    = root_scalar(b0 -> mean(sigmoid(b0 + s)) == prevalencia_base)   # resolve na amostra
y     ~ Bernoulli(sigmoid(b0 + s))
ruido : flipa y em `ruido`% das linhas (pós-geração)
forcar_prevalencia: rejection sampling até bater a proporção (teto de tentativas)
```

Resolver `b0` na amostra garante prevalência realizada ≈ pedida em vez de só esperada.

**Regressão:** `y = b0 + Σ forca_i·direcao_i·x_i + N(0, sigma)`, `sigma` derivado do
slider de ruído. R² realizado é medido, não estimado.

**Contagem:** `lambda = exp(b0 + Σ ...)`, `y ~ Poisson(lambda)`.

**Multiclasse:** matriz `features × K` de pesos (grid dentro de um expander) → softmax.
Default: coluna da classe 0 zerada.

**Ordinal:** score latente + cortes obtidos das proporções pedidas via `norm.ppf`.

**Regra/limiar:** expressão booleana segura sobre colunas (`saldo > 10000`) + `ruido` de
rótulo. Avaliada com o parser do `expressions.py`, nunca `eval`.

**Expressão (v2):** `ast.parse` + walk com whitelist de nós (`BinOp`, `Num`, `Name`,
`Call`), nomes permitidos = colunas + funções (`log`, `exp`, `sqrt`, `abs`, `clip`,
`ruido_normal`, `ruido_uniforme`). Qualquer `Attribute`, `Subscript`, `Import`,
`__dunder__` é rejeitado com mensagem clara.

### 5.5 Nulos

- **MCAR:** máscara aleatória independente, `pct` uniforme por coluna.
- **MAR:** exige `depende_de` (outra variável). Propensão `p_i = sigmoid(k·(s_i − q))`,
  com `q` no quantil `1 − pct` de `s`, `k` ajustado para acertar `pct`. Assim a chance de
  nulo cresce com o valor da dependente — MAR de verdade, não MCAR disfarçado.
- **Representação:** `nan`, `""`, `-1`, `"desconhecido"` — a coluna vira `object` quando
  não for `nan`.

### 5.6 Outliers

| Método | Implementação |
|---|---|
| `cauda` | sorteia linhas e amostra de `F` truncada em `u > 1 − pct` |
| `iqr` | `x *= intensidade` (ou `/= )` conforme o lado, sobre linhas sorteadas |
| `fixo` | atribui `intensidade` exato |

Aplicado depois do target; `pct` é fração das linhas da coluna.

### 5.7 Texto e ID (numpy)

O tipo `texto` existe **só para identificadores**. Sem nomes, e-mail ou CPF — geração
100% numpy, custo desprezível (~ms para 1M linhas).

| Modo | Gera | Exemplo |
|---|---|---|
| `sequencial` | `1..n` na ordem de geração | `1, 2, 3, …` |
| `formatado` | sequencial com prefixo e zero-padding | `ID-00001` |
| `aleatorio` | alfanumérico de comprimento fixo | `k7m2x9` |

**Params:** `modo`, `prefixo`, `n_digitos`, `comprimento`, `alfabeto`
(`numeric` / `alpha` / `alnum`), `unico`.

```python
# aleatorio: vetorizado, sem laço em Python
idx = rng.integers(0, len(alfabeto), size=(n, comprimento))
ids = np.asarray(alfabeto)[idx]          # shape (n, comprimento)
```

**Unicidade.** `sequencial` e `formatado` são únicos por construção. `aleatorio` é
verificado: o app calcula a probabilidade de colisão (`1 − exp(−n²/2·A^c)`) e recusa
gerar se passar de ~1% — mensagem pede comprimento maior. Havendo colisão, as linhas
afetadas são regeradas; nada de sufixo, que quebraria o comprimento fixo.

**Determinismo.** Vem direto do `rng` do dataset. Sem estado global, sem RNG paralelo.

**Por que não Faker.** Custo (~10x) e dependência nova sem retorno: o caso de uso real é
chave de junção, não dado pessoal verossímil. Texto categórico de verdade já é coberto
pelo tipo `categorica`.

**Nulos** se aplicam (`pct_nulos`). **Outliers** ficam desabilitados na UI para texto.

---

## 6. UI (Streamlit)

### Layout

```
┌─ SIDEBAR ─────────────────────┐  ┌─ MAIN ──────────────────────────────┐
│ Dataset                       │  │ [ Variáveis | Relações | Target |   │
│  n_linhas  [1000]             │  │   Diagnóstico | Preview ]           │
│  seed      [42]  (re-gerar)   │  │                                     │
│  export    [CSV ▾]            │  │  ▼ var_1  (saldo)        🗑          │
│  ☑ dicionário                 │  │    bloco 1 Identidade               │
│ ── Schema ──                  │  │    bloco 2 Tipo & Geração           │
│  [Salvar .json] [Carregar]    │  │    bloco 3 Escala & Limites         │
│ ── Ações ──                   │  │    bloco 4 Qualidade                │
│  [＋ Variável]                │  │    preview: histograma + resumo     │
│  [▶ GERAR DATASET]            │  │                                     │
│  [⬇ baixar]                   │  │  [＋ adicionar variável]            │
└───────────────────────────────┘  └─────────────────────────────────────┘
```

### Regras de UI que evitam armadilha

1. **Cascata real:** o dropdown de distribuição é filtrado por `dtype`; trocar o tipo
   reseta `geracao` para o default daquele tipo (com confirmação se perder parâmetros).
2. **`st.session_state` é a fonte da verdade** — a config vive em `st.session_state["config"]`
   como dict serializável. Widgets usam `key=` estável derivado do `id` da variável
   (`f"v{v1}_{campo}"`), nunca do índice da lista (índice muda ao remover).
3. **`st.form` apenas nos cards**, não na página: edita o card e aplica no "Salvar variável",
   evitando rerun a cada tecla — mas o preview de um slider ainda precisa atualizar.
   Solução: sliders fora do form, textos dentro.
4. **Geração é explícita.** Nada gera em cada rerun. `GERAR` grava
   `st.session_state["dataset"]` e `["diagnostico"]`.
5. **Cache:** `@st.cache_data` com `hash(config_json)` — regenerar com o mesmo schema é
   instantâneo.
6. **Avisos de coerência** calculados no schema, antes de gerar: normal+contagem, min>max,
   triangular com moda fora da faixa, ordinal com 1 rótulo, MAR sem dependente,
   target em `relacoes`, matriz não-PSD.
7. **Nunca esconder atenuação:** toda tabela de correlação mostra alvo, realizado e Δ,
   com status `✓ / ⚠`.

### Blocos, resumo

| Bloco | Conteúdo |
|---|---|
| 1 Identidade | `nome`, `descricao` |
| 2 Tipo & Geração | `dtype` → `distribuicao` → `params` (condicional) + extras (rótulos do ordinal, forçar inteiro) |
| 3 Escala & Limites | `transformacao`, `multiplicador`, `offset`, `casas_decimais`, `min`/`max`, `permitir_negativo` |
| 4 Qualidade | `nulos` (pct, mecanismo, representação), `outliers` (pct, método, intensidade) |
| 5 Relações | lista A↔B + força + medida, toggle calibrar, heatmap pedido/realizado |
| 6 Target | tipo, modo, params, influências, diagnóstico de AUC/R² |

---

## 7. Validação (pydantic + regras cruzadas)

| Regra | Erro/Warn |
|---|---|
| `nome` único, identificador válido | erro |
| `nome` ≠ nome do target | erro |
| `relacao.a != relacao.b` | erro |
| `relacao` envolvendo o target | erro |
| `relacao` duplicada (A,B) | warn (mantém a última) |
| matriz não-PSD | erro + botão projetar |
| `min > max` | erro |
| triangular `min ≤ moda ≤ max` | erro |
| ordinal com `< 2` rótulos | erro |
| categórica com peso total 0 | erro |
| `mecanismo=mar` sem `depende_de` | erro |
| `dtype=discreta` + normal/log-normal | warn (arredonda + clipa) |
| `pct_nulos` em feature usada com `forca ≥ 0.5` | warn (degrada o sinal) |
| `texto` + modo `aleatorio` com comprimento insuficiente para `n` (colisão > 1%) | erro |
| `texto` + `formatado` com `n_digitos` menor que os dígitos de `n` | erro |
| `texto` + `prefixo` com caracteres fora de `[A-Za-z0-9_-]` | erro |
| `n_linhas > 1_000_000` | warn (memória) |

---

## 8. Ordem de construção

Cada fase termina com o app rodando e testável — nada de fase que só faz sentido no fim.

### Fase 0 — Esqueleto
`pyproject.toml` (uv), `.gitignore`, `git init`, `app.py` com layout vazio, `README.md`.
**Aceite:** `uv run streamlit run app.py` abre sem erro.

### Fase 1 — Schema + marginais + IDs
`types.py`, `variable.py`, `dataset.py`; `marginals.py` com todas as distribuições via
`ppf`; `text.py` com os 3 modos de ID (sequencial, formatado, aleatório); testes
estatísticos (média/desvio/quantis contra `scipy`) e de determinismo/unicidade.
**Aceite:** `pytest tests/test_marginals.py tests/test_text.py` verde; amostrar 100k de
cada distribuição bate os momentos esperados dentro da tolerância; IDs idênticos com a
mesma seed; colisão de `aleatorio` recusada quando o comprimento é insuficiente.

### Fase 2 — UI dos blocos 1–4
`variable_card.py` com cascata tipo→distribuição→params; add/remove/reordenar variável;
preview por card.
**Aceite:** montar o exemplo `saldo` pela UI gera a coluna correta no preview.

### Fase 3 — Escala, qualidade, pipeline, export
`transforms.py`, `missing.py`, `outliers.py`, `pipeline.py` (sem cópula ainda),
`io/export.py`, `io/profile.py`, `sidebar.py`.
**Aceite:** gerar CSV com 5 variáveis do exemplo; salvar schema → reabrir → gerar idêntico
(mesma seed).

### Fase 4 — Correlação
`copula.py` (matriz, PSD, amostragem, calibração), `relations.py` + heatmap.
**Aceite:** pedir `saldo↔renda = 0.72` entre duas contínuas entrega Spearman dentro de
±0.02; com `num_tvs` (Poisson) a calibração converge ou avisa; matriz inválida é detectada
e projeção deixa o menor autovalor ≥ -1e-8.

### Fase 5 — Target
`target.py` (binária, regressão, contagem, multiclasse, ordinal, regra), `target_card.py`,
report de AUC/R²/prevalência.
**Aceite:** prevalência pedida 0.25 entrega 0.25 ±0.02; AUC cai monotonicamente conforme
o slider de ruído sobe; regra `saldo > 10000` produz taxa coerente e o ruído de rótulo
move essa taxa na direção esperada.

### Fase 6 — Diagnóstico e polimento
Aba Diagnóstico completa (heatmap diff, tabela Δ, integridade, dicionário), avisos de
coerência, dicionário `.md`, estados vazios, mensagens de erro, `README` com screenshots.
**Aceite:** checklist da seção 11 fechado.

---

## 9. Testes

| Arquivo | Cobre |
|---|---|
| `test_schema.py` | roundtrip JSON, union discriminada, todas as regras de validação |
| `test_marginals.py` | momentos e quantis de cada distribuição vs. `scipy` |
| `test_copula.py` | Spearman realizado p/ contínuas, monotonicidade, PSD/projeção, convergência da calibração |
| `test_target.py` | prevalência, AUC vs. ruído, R², multiclasse com proporções, regra + ruído |
| `test_text.py` | determinismo por seed, unicidade, recusa por colisão, padding e prefixo |
| `test_pipeline.py` | ordem 1..6, target não enxerga nulos, clipping, reprodutibilidade por seed |
| `test_expressions.py` | rejeita `__import__`, atributos, subscript; aceita funções da whitelist |
| `test_export.py` | csv/parquet/xlsx, dicionário, dtypes preservados |

Meta: núcleo do motor (`engine/`) com cobertura ≥ 90%. UI não é testada por unidade —
é verificada manualmente pelos critérios de aceite de cada fase.

---

## 10. Riscos e mitigação

| Risco | Mitigação |
|---|---|
| Calibração de correlação não converge com muitas categóricas | teto de iterações + aviso explícito + mostrar realizado |
| Projeção PSD muda os valores pedidos silenciosamente | diff visível + confirmação antes de aplicar |
| Rerun do Streamlit perde estado / `key` instável | config em `session_state` + keys derivadas de `id`, nunca de índice |
| `st.form` bloqueia atualização do preview | sliders fora do form |
| Normal em variável de contagem gera valores impossíveis | validador avisa + round/clip automático documentado |
| Correlação entre categóricas é "latente" (policórica) | rótulo explícito na UI: força em escala latente; Δ sempre visível |
| Memória em n grande com muitas categóricas `object` | aviso acima de 1M; `category` dtype opcional no export |
| Expressão do target como vetor de execução de código | `ast` com whitelist, sem `eval`; testes adversariais |

---

## 11. Definition of Done (v1)

- [ ] Montar o dataset de exemplo completo pela UI, do zero, em < 3 minutos
- [ ] Cada uma das ~17 distribuições gera valores estatisticamente corretos
- [ ] Correlação pedida vs. realizada visível e dentro de tolerância (ou avisada)
- [ ] Target binário e de regressão com diagnóstico de qualidade na tela
- [ ] Nulos e outliers refletem exatamente o percentual configurado
- [ ] IDs únicos e reprodutíveis nos 3 modos (sequencial, formatado, aleatório)
- [ ] Export CSV + Parquet + dicionário `.md`
- [ ] Salvar/carregar schema reproduz o mesmo dataset com a mesma seed
- [ ] Nenhuma mensagem de erro crua do numpy/pandas chega ao usuário
- [ ] `pytest` verde; `engine/` ≥ 90% de cobertura
- [ ] `README` com instruções de instalação e um exemplo ponta a ponta

---

## 12. Comandos

```bash
uv sync
uv run streamlit run app.py
uv run pytest -q
uv run pytest --cov=gerador.engine --cov-report=term-missing
```

---

*Plano registrado em `/home/ander/.opencode/plan/gerador-dados-plano.md`.*
*Nenhum arquivo do projeto foi criado — execução requer sair do Plan mode.*

---

## 13. Registro de execução (o que mudou ao construir)

O plano foi executado integralmente. Três desvios conscientes:

| Item do plano | Como ficou | Motivo |
|---|---|---|
| pydantic para o schema | `dataclasses` + `validar()` explícito | menos dependência; mensagens de erro em português e por campo |
| plotly nos gráficos | `st.bar_chart` + heatmap HTML/CSS | evita matplotlib/plotly; heatmap sem dependência |
| `Relacao`/`Influencia` sem identificador | campo `uid` (serializado no JSON) | chaves de widget estáveis ao adicionar/remover linhas |

Correção de design encontrada pelos testes (não prevista no plano):

- A cópula precisa trabalhar com **códigos** de categoria (0..k−1) na ordem declarada, e
  não com os rótulos: `rankdata` sobre strings ordena alfabeticamente e pode **inverter**
  o sinal da correlação pedida. O pipeline converte códigos → `pd.Categorical` ordenado no
  fim, então o arquivo exportado continua legível e o diagnóstico continua correto.

Estado final: 107 testes passando, app sobe via `streamlit run app.py`.
