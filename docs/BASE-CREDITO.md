# Base de crédito mensal (painel sintético)

Preset `credito_mensal` — carregável em **Modelos prontos → Carregar modelo** ou via
`obter_preset("credito_mensal")`.

## O que essa base é

Um **painel balanceado** de crédito: cada cliente aparece **uma vez por mês**, com o
estado daquele mês. A base entrega apenas o **estado em `t`** — não há defasagens, somas
móveis, agregados de 12 meses nem tempo de relacionamento derivado. A engenharia de
features e a construção do rótulo ficam com quem consome o arquivo, o que evita vazamento
de informação do futuro para o passado.

| Item | Valor padrão |
|---|---|
| Clientes (entidades) | 6.000 (ajustável na barra lateral) |
| Meses | 12 |
| Linhas | 72.000 |
| Janela | 2025-12-01 → 2026-11-01 |
| Chave | `id_cliente` |
| Coluna de tempo | `data_mes` (sempre dia 1) |
| Target | **nenhum** |
| Seed | 42 |

O painel é balanceado: todo cliente tem os 12 meses.

```python
from gerador.presets import obter_preset
from gerador.engine.pipeline import gerar_dataset

config = obter_preset("credito_mensal")
config.n_linhas = 1000          # clientes
config.painel.n_meses = 6
res = gerar_dataset(config)
res.df.to_csv("credito.csv", index=False)
```

## Variáveis

Ordem das colunas: `id_cliente`, `data_mes`, depois as features.

| Coluna | Tipo | Geração | Evolução | Observação |
|---|---|---|---|---|
| `id_cliente` | texto | `CLI-000001`… | repetido | chave do painel |
| `idade` | contínua | normal(39; 12), 18–90 | estática | constante no painel |
| `sistema_celular` | categórica | Android 70 / outro 2 / iOS 28 | estática | — |
| `limite` | contínua | log-normal(9,0; 0,8), 500–200.000 | estática | limite individual |
| `renda` | contínua | log-normal(7,6; 0,6), mín. 780 | AR1 φ=0,90 | renda mensal |
| `score_bureau_A` | contínua | normal(600; 110), 0–1000 | AR1 φ=0,95 | — |
| `qtd_instituicoes` | discreta | Poisson(2,4), 0–15 | AR1 φ=0,95 | muda mês a mês |
| `contratos_ativos` | discreta | Poisson(1,8), 0–12 | AR1 φ=0,92 | — |
| `valor_a_vencer` | contínua | log-normal(6,8; 0,9) | AR1 φ=0,85 | saldo futuro |
| `atraso_em_dias` | discreta | empírica (70% em 0; depois 5…360) | AR1 φ=0,97 | escada de atraso |
| `valor_entradas` | contínua | log-normal(7,75; 0,55) | AR1 φ=0,80 | fluxo do mês |
| `valor_saidas` | contínua | log-normal(7,70; 0,55) | AR1 φ=0,80 | fluxo do mês |
| `patrimonio` | contínua | log-normal(11,0; 1,2), máx. 20 mi | AR1 φ=0,98 | — |
| `UF` | categórica | 27 UFs, peso populacional | estática | ordem do menor→maior renda média |
| `ocupacao` | categórica | 6 níveis | AR1 φ=0,97 | desempregado…servidor público |
| `escolaridade` | ordinal | fundamental→pós-graduação | estática | 4 níveis |
| `score_bureau_B` | contínua | normal(550; 130), 0–1000 | AR1 φ=0,95 | — |
| `tempo_de_conta` | discreta | uniforme 1–240 | tendência +1/mês | cresce com o tempo |
| `valor_vencido` | derivada | ver fórmula | — | parcela vencida |
| `valor_prejuizo` | derivada | ver fórmula | — | prejuízo (atraso ≥ 180) |
| `valor_refinanciado` | derivada | ver fórmula | — | refinanciamento |
| `limite_total_instituicoes` | derivada | ver fórmula | — | soma dos limites |

## Como o tempo funciona

O motor não sorteia cada mês de forma independente. Ele usa um modelo **B + W** por
variável:

```
Z[j,t] = a[j] · B[j] + b[j] · W[j,t]      com a = √φ e b = √(1−φ)
```

- `B[j]` é o traço permanente da entidade (quem o cliente é);
- `W[j,t]` é o choque do mês;
- `φ` é a persistência: a autocorrelação de Spearman entre meses consecutivos.

Assim, `renda` de um cliente tende a ficar perto da renda do mês anterior, mas não é
congelada; `atraso_em_dias` com φ=0,97 produz transições realistas de *estável*, *piora* e
*cura* sem que essas probabilidades apareçam como configuração — elas emergem da
persistência alta combinada com a marginal empírica.

A correlação entre variáveis é definida na cross-section e compensada por `φ`, de modo
que a matriz pedida vale **em cada mês**, não só no agregado. O diagnóstico mede
pedido vs. realizado no **primeiro mês**.

Modos disponíveis em `Evolucao`:

| Modo | Efeito |
|---|---|
| `estatica` | mesmo valor em todos os meses |
| `ar1` | persistência `φ` mês a mês |
| `tendencia` | valor inicial + `passo` por mês |

Texto e variáveis derivadas são sempre estáticos (o texto é repetido; a derivada é
recalculada mês a mês a partir das bases).

## Variáveis derivadas

Rodam **depois** das variáveis base, na ordem de declaração, e enxergam as colunas já
declaradas acima. A correlação delas vem da fórmula, não da cópula — por isso não entram
na matriz de relações.

```
valor_vencido            = (atraso_em_dias > 0) * valor_a_vencer * ruido_uniforme(0.1, 0.9)
valor_prejuizo           = (atraso_em_dias >= 180) * valor_vencido * ruido_uniforme(0.5, 0.8)
valor_refinanciado       = (atraso_em_dias >= 90) * (ruido_uniforme() < 0.20)
                           * valor_vencido * ruido_uniforme(0.9, 1.3)
limite_total_instituicoes = maximum(limite, limite * (1 + ruido_uniforme(0.2, 2.5)))
```

A derivada é recalculada **mês a mês** e aceita a mesma escala das demais variáveis
(transformação, multiplicador, offset, casas decimais, mínimo/máximo). No preset, a escala
padrão apenas arredonda para centavos.

## Correlações

42 pares declarados em Spearman. Âncoras principais:

| Par | ρ | Par | ρ |
|---|---|---|---|
| renda ↔ valor_entradas | 0,85 | score_A ↔ score_B | 0,75 |
| valor_entradas ↔ valor_saidas | 0,80 | score_A ↔ atraso | −0,55 |
| renda ↔ limite | 0,65 | renda ↔ patrimônio | 0,55 |
| qtd_instituicoes ↔ contratos | 0,60 | limite ↔ valor_a_vencer | 0,55 |
| renda ↔ ocupação | 0,50 | score_B ↔ atraso | −0,50 |
| renda ↔ escolaridade | 0,45 | renda ↔ score_A | 0,45 |

As correlações pedidas **não fecham sozinhas** (dois clientes com renda alta e limite alto
implicam alguma correlação entre limite e fluxo de entradas, por exemplo). O preset liga
`completar_psd=True`: os pares não declarados são preenchidos pela matriz válida mais
próxima, em vez de recusar a geração. A UI mostra um aviso e o diagnóstico confirma os
pares declarados dentro da tolerância (0,04).

## Exemplos de uso a jusante

Com a base em mãos, a feature engineering fica explícita e controlada:

```python
import pandas as pd

df = pd.read_csv("credito.csv", parse_dates=["data_mes"])
df = df.sort_values(["id_cliente", "data_mes"])

g = df.groupby("id_cliente")
df["renda_lag1"] = g["renda"].shift(1)
df["atraso_12m_max"] = g["atraso_em_dias"].transform(
    lambda s: s.rolling(12, min_periods=1).max()
)
df["entradas_3m"] = g["valor_entradas"].transform(
    lambda s: s.rolling(3, min_periods=1).sum()
)

# rótulo de exemplo: atraso grave no mês seguinte
df["y"] = (g["atraso_em_dias"].shift(-1) >= 90).astype(int)
```

## Limites

- É uma base **sintética**: as distribuições e correlações são plausíveis, não calibradas
  com dados reais de nenhum bureau.
- Categorias ordinais (UF, escolaridade, ocupação) entram na cópula pela **ordem
  declarada**; reordenar os níveis muda a correlação realizada.
- A calibração controla a correlação de Spearman na cross-section do primeiro mês. Em
  amostras pequenas há ruído amostral.
- Variáveis discretas com muitos empates (`atraso_em_dias`, `qtd_instituicoes`) realizam
  autocorrelação um pouco abaixo do `φ` pedido — o teto de Spearman com empates é menor
  que 1. Na geração padrão: 0,88 para `atraso_em_dias` (φ=0,97, com 70% dos meses em zero)
  e 0,91 para `qtd_instituicoes` (φ=0,95).
- O volume padrão (72 mil linhas) gera em ~1,5 s; acima de 500 mil linhas a UI avisa e
  acima de 1 milhão recusa.
