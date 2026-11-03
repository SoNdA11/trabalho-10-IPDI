# Guia geral — o trabalho inteiro, passo a passo

Este guia explica **toda a implementação**, do download das imagens ao mapa final, de forma
que qualquer integrante consiga responder sobre qualquer parte. Os guias individuais
aprofundam o trecho de cada um; o [guia da demonstração](guia-demonstracao.md) explica a interface.

## Sumário

1. [O problema e a ideia da solução](#1-o-problema-e-a-ideia-da-solução)
2. [Conceitos básicos de sensoriamento remoto](#2-conceitos-básicos-de-sensoriamento-remoto)
3. [Os dados](#3-os-dados)
4. [Visão geral do código](#4-visão-geral-do-código)
5. [Passo 1 — Aquisição](#passo-1--aquisição-srcdata_loaderpy)
6. [Passo 2 — Exploração](#passo-2--exploração-srcfigurespy)
7. [Passo 3 — Pré-processamento](#passo-3--pré-processamento-srcpreprocessingpy)
8. [Passo 4 — Extração de características](#passo-4--extração-de-características-srcfeature_extractionpy)
9. [Passo 5 — Preparação das amostras](#passo-5--preparação-das-amostras-srcsamplingpy)
10. [Passo 6 — Treinamento](#passo-6--treinamento-srctrainpy)
11. [Passo 7 — Avaliação](#passo-7--avaliação-srcevaluationpy)
12. [Passo 8 — Mapa de classificação](#passo-8--mapa-de-classificação-srcpredictpy)
13. [Passo 9 — Análise dos erros](#passo-9--análise-dos-erros)
14. [A interface e a apresentação](#a-interface-e-a-apresentação)
15. [Resultados em uma tabela](#resultados-em-uma-tabela)
16. [Como rodar tudo](#como-rodar-tudo)
17. [Organização do seminário](#organização-do-seminário)
18. [Glossário](#glossário)
19. [Perguntas que qualquer um pode receber](#perguntas-que-qualquer-um-pode-receber)

---

## 1. O problema e a ideia da solução

**Problema.** Uma imagem de satélite é uma grade de pixels. Em cada pixel, o satélite mede
quanto de luz o terreno reflete em várias faixas do espectro (as **bandas**). Queremos
transformar essa grade num **mapa temático**: cada pixel pintado com a classe de cobertura do
solo que existe ali, entre **vegetação, área agrícola, área urbana, solo exposto e água**.

**Ideia da solução (classificação supervisionada).**

1. Juntar exemplos em que já se sabe a resposta (pixels **rotulados**).
2. Descrever cada exemplo por números (**características**: valores das bandas, índices, textura).
3. Treinar um algoritmo (**Random Forest**) que aprende a relação "números → classe".
4. Medir o acerto em exemplos que o algoritmo **nunca viu** (**teste**).
5. Aplicar o modelo a todos os pixels da cena e obter o mapa.

**Três níveis de classificação** (o professor pediu para diferenciá-los):

| nível | a unidade classificada | a saída | onde está no projeto |
|---|---|---|---|
| **imagem inteira** | um recorte de 64×64 px (640 m × 640 m) | **um** rótulo para o recorte | EuroSAT |
| **por pixel** | cada pixel de 10 m | um **mapa** com uma classe por pixel | cena do INPE (Mossoró) |
| **por região** | uma janela de 64×64 px que desliza pela cena | um mapa **em blocos** | modelo do EuroSAT sobre Mossoró |

## 2. Conceitos básicos de sensoriamento remoto

- **Espectro eletromagnético.** A luz do Sol tem vários comprimentos de onda. O olho vê de ~400
  a ~700 nm (azul, verde, vermelho). Os satélites medem também o **infravermelho próximo (NIR,
  ~700–1000 nm)** e o **infravermelho de ondas curtas (SWIR, ~1400–2300 nm)**.
- **Reflectância.** A fração da luz que o alvo reflete, de 0 (absorve tudo) a 1 (reflete tudo).
- **Assinatura espectral.** Cada material tem uma curva própria de reflectância. A **vegetação**
  absorve o vermelho (clorofila) e reflete muito o NIR (estrutura da folha). A **água** absorve o
  NIR e o SWIR. **Solo e asfalto** são claros no SWIR. É essa diferença que permite classificar.
- **Resoluções.**
  - espacial: tamanho do pixel no chão (10, 20 ou 60 m no Sentinel-2);
  - espectral: quantas bandas e quão estreitas (13 no Sentinel-2, 4 no CBERS-4 MUX);
  - temporal: de quanto em quanto tempo o satélite passa (5 dias com os dois Sentinel-2);
  - radiométrica: quantos níveis cada pixel registra (12 bits = 4096 níveis no Sentinel-2).
- **Níveis de processamento.**
  - **L1C:** reflectância no **topo da atmosfera** (sem retirar o efeito do ar).
  - **L2A/SR:** reflectância de **superfície**, já com correção atmosférica.
- **Composição colorida.** Escolher três bandas e exibir como vermelho, verde e azul na tela.
  Cor verdadeira = B04, B03, B02. Falsa cor = B08 (NIR), B04, B03, em que a vegetação viva fica vermelha.

## 3. Os dados

| dado | origem | o que é | papel |
|---|---|---|---|
| **Sentinel-2B MSI L1C** | **INPE** · Brazil Data Cube, coleção `S2_L1C_BUNDLE-1` | cena de 01/09/2024, tile 24MXV, 13 bandas, 0% de nuvem | imagem principal da classificação por pixel e por região |
| **CBERS-4 MUX L4 SR** | **INPE** · Brazil Data Cube, coleção `CB4-MUX-L4-SR-1` | cena de 04/10/2024, 4 bandas (azul, verde, vermelho, NIR), 20 m, com máscara de nuvem | mesmo experimento num satélite sino-brasileiro sem SWIR |
| **EuroSAT MS** | Zenodo (Helber et al., 2019) | 27.000 recortes 64×64 do Sentinel-2, 13 bandas, 10 classes rotuladas à mão | classificação da imagem inteira |
| **ESA WorldCover 2021** | Agência Espacial Europeia | mapa global de cobertura, 10 m, 11 classes | **rótulo** (gabarito) dos pixels de Mossoró |

**Área de estudo:** Mossoró/RN, 25,6 × 24,4 km (lon −37,45 a −37,22; lat −5,30 a −5,08), com
cidade, Rio Apodi-Mossoró, açudes, salinas, agricultura, caatinga e solo exposto.

**Por que essas escolhas?**

- O professor pediu dados do INPE: as duas cenas classificadas vêm do catálogo oficial do INPE.
- O INPE não fornece um gabarito pixel a pixel para Mossoró. O WorldCover cumpre esse papel.
- O EuroSAT é o único conjunto público de recortes Sentinel-2 com 13 bandas e rótulos manuais.
- L1C foi escolhido porque o EuroSAT também é L1C: assim o modelo do EuroSAT pode ser aplicado à cena do INPE.

**Classes do trabalho e como foram obtidas:**

| classe | códigos do WorldCover | classes do EuroSAT |
|---|---|---|
| Vegetação | 10 árvores, 20 arbustos, 30 herbácea, 90 área úmida, 95 mangue | Floresta, Vegetação herbácea |
| Área agrícola | 40 cultivos | Cultura anual, Cultura permanente, Pastagem |
| Área urbana | 50 construído | Residencial, Industrial, Rodovia |
| Solo exposto | 60 solo/vegetação esparsa | — (não existe) |
| Água | 80 água | Rio, Mar/lago |

Na área de estudo: 82,4% vegetação, 6,8% urbana, 4,3% agrícola, 3,7% água e 2,9% solo exposto.
O forte **desbalanceamento** influencia a amostragem e a escolha das métricas.

## 4. Visão geral do código

```text
src/config.py              ← tudo o que identifica o experimento: caminhos, IDs das cenas, classes, cores, bandas
src/data_loader.py         passo 1   baixa e prepara os dados
src/figures.py             passos 2 e 9   figuras de exploração, mapas e erros
src/preprocessing.py       passo 3   reflectância, máscaras, realce, filtros
src/feature_extraction.py  passo 4   índices, estatísticas, textura
src/sampling.py            passo 5   divisão espacial e amostragem
src/train.py               passo 6   treino e estudos comparativos
src/evaluation.py          passo 7   métricas
src/predict.py             passo 8   mapa por pixel, por região e classificação de recorte
src/app.py + web/          interface de demonstração
presentation/              slides (Reveal.js) + scripts que exportam os números reais
tests/                     24 testes automatizados (pytest)
```

**Fluxo dos dados:**

```text
INPE (STAC) ──► recorte + alinhamento ──► s2_mossoro_l1c_10m.tif ─┐
WorldCover ───► reprojeção ─────────────► s2_mossoro_worldcover ──┼─► reflectância → características por pixel
                                                                   │      → blocos treino/val/teste → Random Forest
EuroSAT MS ──► 27.000 GeoTIFF ──► características por recorte ─────┘      → métricas (JSON) → mapa (GeoTIFF/PNG)
                                                                          → slides e interface
```

---

## Passo 1 — Aquisição (`src/data_loader.py`)

**O que faz.** Baixa os quatro conjuntos de dados e deixa as cenas prontas, recortadas e numa
grade comum.

**Como funciona.**

1. `stac_item(coleção, id)` consulta o catálogo STAC do INPE
   (`https://data.inpe.br/bdc/stac/v1/collections/<coleção>/items/<id>`) e recebe um JSON com
   os links de cada arquivo. O catálogo é aberto e não exige cadastro.
2. `ensure_s2_zip()` baixa o pacote SAFE do Sentinel-2 (~700 MB). Cada banda é um JPEG2000 dentro do zip.
3. `aoi_grid(res)` cria a grade da área de estudo em UTM 24S, com origem em múltiplos de 20 m.
   Assim um pixel CBERS (20 m) cobre exatamente 2×2 pixels Sentinel-2 (10 m).
4. `prepare_s2_scene()` lê cada banda direto do zip (`/vsizip/`), recorta a janela da área e
   **reprojeta para a grade de 10 m** (`rasterio.warp.reproject`). As bandas de 20 e 60 m são
   interpoladas (bilinear). O resultado é um GeoTIFF com 13 bandas.
5. `prepare_cbers_scene()` lê **só a janela** das 4 bandas e da máscara de nuvem direto da
   internet (COG + `/vsicurl/`), sem baixar a cena inteira.
6. `prepare_reference()` reprojeta o WorldCover para a grade de cada cena: vizinho mais próximo
   a 10 m e **moda** a 20 m (classe mais frequente dos 2×2 pixels; rótulo não tem média).
7. `ensure_eurosat_ms()` baixa e extrai os 27.000 GeoTIFF do EuroSAT (~2 GB).

```python
# trecho de prepare_s2_scene: cada banda vai para a mesma grade de 10 m
reproject(data, out[i], src_transform=..., src_crs=src.crs,
          dst_transform=transform, dst_crs=C.AOI_CRS,
          resampling=Resampling.nearest if src.res[0] == 10 else Resampling.bilinear)
```

**Saídas:** `data/processed/s2_mossoro_l1c_10m.tif` (2442×2558 px, 13 bandas),
`cbers4_mux_mossoro_20m.tif` (1221×1279 px, 4 bandas + máscara) e as referências reprojetadas.

## Passo 2 — Exploração (`src/figures.py`)

Antes de treinar, olhamos os dados:

- **composições** cor verdadeira, falsa cor e SWIR (`scene_figures`);
- **assinaturas espectrais** médias de cada classe (`presentation/scripts/exportar_dados.py`):
  confirmam que as classes têm curvas diferentes;
- **proporção das classes**: revela o desbalanceamento (82% vegetação);
- **boxplots dos índices** por classe: mostram que nenhum índice separa tudo sozinho;
- **bandas e resoluções**: revelam que é preciso alinhar as bandas.

## Passo 3 — Pré-processamento (`src/preprocessing.py`)

| operação | função | por quê |
|---|---|---|
| números digitais → reflectância | `to_reflectance` | o arquivo guarda inteiros (DN); reflectância = DN / 10000 |
| harmonização da baseline | `harmonize_l1c` | desde 2022 a ESA soma 1000 aos DN do L1C (`RADIO_ADD_OFFSET = −1000`); o EuroSAT é de 2017–2018 e não tem esse deslocamento |
| máscara de pixels válidos | `valid_mask` | DN 0 = sem dado; 65535 = saturado; no CBERS, −9999 = sem dado e a **CMASK** do INPE marca nuvem (127 = limpo) |
| tratamento de inválidos | `fix_invalid` | NaN/inf viram a mediana da banda; valores ficam em [0; 1,5] |
| realce para exibição | `stretch`, `composite` | estica os valores entre os percentis 1–99% para a imagem ficar visível; **não** entra no classificador |
| normalização z-score | `StandardScaler` no `train.py` | só para SVM e k-NN, ajustada **no treino**; o Random Forest não precisa |
| filtro de moda 3×3 | `majority_filter` | depois de classificar, troca cada rótulo pelo mais frequente da vizinhança (remove "sal e pimenta") |
| filtro de mediana | `denoise_median` | disponível; não usado, porque a imagem não tem ruído impulsivo |

```python
def to_reflectance(dn, sensor, offset=None):
    x = dn.astype(np.float32)
    if sensor in ("sentinel2", "rgb"):
        return (x + (offset or 0.0)) / 10000.0     # L1C atual: offset = −1000
    if sensor == "cbers4_mux":
        return x * 1e-4
```

**Exemplo:** um pixel de água tem DN 1330 na banda B08 do L1C de 2024. Reflectância =
(1330 − 1000)/10000 = **0,033**. Sem a harmonização, daria 0,133, quatro vezes mais clara.

## Passo 4 — Extração de características (`src/feature_extraction.py`)

O classificador não recebe a imagem: recebe **uma lista de números por amostra**.

**Índices espectrais**, todos da forma (a − b)/(a + b), entre −1 e 1:

| índice | fórmula | o que realça |
|---|---|---|
| NDVI | (NIR − Red)/(NIR + Red) | vegetação |
| NDWI | (Green − NIR)/(Green + NIR) | água |
| MNDWI | (Green − SWIR1)/(Green + SWIR1) | água, separando-a melhor da área urbana |
| NDBI | (SWIR1 − NIR)/(SWIR1 + NIR) | área construída |
| BSI | ((SWIR1+Red) − (NIR+Blue))/((SWIR1+Red) + (NIR+Blue)) | solo exposto |

O código procura as bandas pelo **papel físico** (`config.SENSOR_BANDS`). Se o sensor não tem a
banda, o índice **não é calculado** (`spectral_index` levanta erro). CBERS-4 → NDVI e NDWI;
modelo só RGB → nenhum índice.

**Por recorte (EuroSAT), 97 números** (`patch_features`):

- 13 bandas × 5 estatísticas (média, desvio, percentis 10, 50, 90) = 65;
- 5 índices × 4 estatísticas = 20;
- histograma do NDVI em 6 faixas = 6;
- textura na banda NIR = 6: GLCM (contraste, homogeneidade, energia, correlação), gradiente de
  Sobel e variância local.

**Por pixel (cena), 22 números no Sentinel-2 / 10 no CBERS** (`pixel_features`):

- reflectância das bandas (13 ou 4);
- índices disponíveis (5 ou 2);
- contexto 5×5: média e desvio do NDVI e do NIR na vizinhança (4).

**Exemplo real (pixels da cena, valores calculados pelo código):**

| pixel | B04 | B08 | B11 | NDVI | MNDWI | NDBI | classe |
|---|---|---|---|---|---|---|---|
| (2095, 80) | 0,065 | 0,033 | 0,019 | −0,33 | **+0,70** | −0,28 | Água (100% dos votos) |
| (1336, 937) | 0,201 | 0,331 | 0,371 | 0,25 | −0,34 | 0,06 | Área urbana (94%) |
| (617, 2059) | 0,128 | 0,226 | 0,350 | 0,28 | −0,53 | 0,22 | Vegetação (85%) |
| (2260, 147) | 0,190 | 0,219 | 0,311 | **0,07** | −0,27 | 0,17 | Solo exposto (93%) |

Conta do NDVI da água: (0,033 − 0,065)/(0,033 + 0,065) = −0,032/0,098 = −0,33.

## Passo 5 — Preparação das amostras (`src/sampling.py`)

**O risco: vazamento espacial.** Pixels vizinhos são quase iguais. Se o treino tem um pixel e o
teste tem o vizinho, o modelo "acerta" por memória, e a métrica fica otimista.

**Solução na cena** (`block_split_map`):

1. cortar a cena em blocos de **1,28 km** (128 × 128 px);
2. sortear cada bloco inteiro para **treino (60%)**, **validação (20%)** ou **teste (20%)**;
3. descartar uma faixa de **2 px** na borda de cada bloco, porque a característica de contexto
   olha uma janela 5×5 e não pode ver pixels de outro conjunto.

**Solução no EuroSAT** (`geo_groups`, `spatial_split`): cada recorte tem coordenada. Os recortes
são agrupados em células de 0,25° (~28 km), e `StratifiedGroupKFold` divide os **grupos**:
19.285 treino, 3.857 validação, 3.858 teste, em 2.047 células.

**Quantas amostras de cada classe no treino?** (`sample_pixels_sqrt`) Testamos três opções no mesmo teste:

| amostragem | acurácia | F1-macro | comportamento |
|---|---|---|---|
| balanceada (8.000/classe) | 77,5% | 58,4% | exagera as classes raras |
| proporcional | 91,2% | 60,0% | quase ignora as raras |
| **raiz da frequência** (usada) | 88,6% | **64,2%** | meio-termo |

Com a raiz da frequência, uma classe 16 vezes mais comum recebe só 4 vezes mais amostras.
São 40.000 pixels de treino, 20.000 de validação (aleatórios) e o teste usa **todos** os
1.096.671 pixels dos blocos de teste.

## Passo 6 — Treinamento (`src/train.py`)

**Algoritmo: Random Forest** (Breiman, 2001).

- **Árvore de decisão:** uma sequência de perguntas "característica ≤ limiar?". Em cada nó, a
  pergunta escolhida é a que deixa os dois lados mais "puros" (menor impureza de Gini, 1 − Σ pₖ²).
- **Floresta:** centenas de árvores, cada uma treinada numa amostra sorteada com reposição
  (**bootstrap**) e, em cada nó, só com √d características sorteadas. As árvores ficam
  diferentes, e a **votação** cancela os erros individuais.
- **Probabilidade:** a fração de árvores que votou em cada classe.

**Exemplo real:** a primeira árvore do modelo da cena classifica o pixel urbano (1336, 937)
com 17 perguntas. As seis primeiras são:

| nó | pergunta | valor do pixel | caminho |
|---|---|---|---|
| 1 | NDVI ≤ 0,200? | 0,245 | não → direita |
| 2 | B8A ≤ 0,296? | 0,347 | não → direita |
| 3 | B03 ≤ 0,114? | 0,182 | não → direita |
| 4 | NDVI_media5x5 ≤ 0,201? | 0,189 | sim → esquerda |
| 5 | BSI ≤ 0,175? | 0,064 | sim → esquerda |
| 6 | NIR_desvio5x5 ≤ 0,009? | 0,022 | não → direita |
| … | (mais 11 perguntas) | | folha: **Área urbana** |

Das 200 árvores, 94% votaram "Área urbana". As árvores têm, em média, profundidade 32 e cerca de 4.100 folhas.

**Ajuste de hiperparâmetros** (`tune_rf`): 4 combinações de `max_features` ∈ {sqrt, 0,3} e
`min_samples_leaf` ∈ {1, 3}, escolhidas pelo F1-macro **na validação**. Venceu `sqrt` com folha
mínima 1 (o CBERS escolheu 3). Só depois o modelo é medido no teste, uma única vez.

**Estudos extras no mesmo teste** (todos em `train.py`):

- SVM (RBF) e k-NN (k = 7), com normalização ajustada no treino;
- Random Forest só com as 3 bandas visíveis (o valor do NIR e do SWIR);
- divisão aleatória (o tamanho do otimismo do vazamento);
- as três estratégias de amostragem.

**Saídas:** `models/rf_eurosat_ms.joblib`, `rf_eurosat_rgb.joblib`, `rf_s2_pixel.joblib`,
`rf_cbers_pixel.joblib` e `results/metrics/*.json`.

## Passo 7 — Avaliação (`src/evaluation.py`)

Para uma classe X: VP = era X e o modelo disse X; FP = não era X e disse X; FN = era X e disse outra.

| métrica | fórmula | responde |
|---|---|---|
| acurácia | acertos / total | quanto acertou no geral |
| precisão | VP/(VP+FP) | quando diz X, quanto acerta |
| recall | VP/(VP+FN) | dos X reais, quantos achou |
| F1 | 2PR/(P+R) | equilíbrio entre os dois |
| F1-macro | média do F1 das classes | todas as classes com o mesmo peso |
| kappa | (pₒ − pₑ)/(1 − pₑ) | acerto além do acaso |
| matriz de confusão | contagem real × previsto | onde estão os erros |

**Conta real do kappa (Sentinel-2):** pₒ = 0,886 (acurácia); pₑ = 0,711 (o acerto esperado se as
previsões fossem sorteadas com as mesmas proporções); kappa = (0,886 − 0,711)/(1 − 0,711) = **0,606**.
O pₑ é alto porque quase tudo é vegetação: "chutar vegetação" já acertaria muito.

**Conta real do F1 da área urbana:** precisão 0,695, recall 0,893 → F1 = 2·0,695·0,893 /
(0,695 + 0,893) = **0,782**.

## Passo 8 — Mapa de classificação (`src/predict.py`)

- **Por pixel** (`classify_scene`, também em `train.train_scene`): calcula as características em
  faixas de 256 linhas, prediz todos os 6,2 milhões de pixels válidos (~27 s), aplica o filtro de
  moda e grava `results/predictions/s2_map.tif` (GeoTIFF com legenda de cores, abre no QGIS).
- **Por região** (`region_map`): uma janela de 64×64 px anda de 32 em 32 px pela cena; cada
  janela é classificada pelo **modelo do EuroSAT** (5.850 janelas em 12 s) e pinta seu bloco central.
- **Imagem inteira** (`classify_chip`): um recorte → um rótulo e as probabilidades.
- **Janela ao vivo** (`classify_window`): usada pela interface, classifica um pedaço da cena na hora.

## Passo 9 — Análise dos erros

- **Mapa de erros** (`figures.map_figures`): nos blocos de teste, verde = acerto e vermelho =
  erro. Os erros ficam nas **bordas** entre classes.
- **Matriz de confusão:** agrícola → vegetação (pousio × caatinga seca); água → solo exposto
  (açudes secos, salinas); no EuroSAT, cultura anual → cultura permanente.
- **Exemplos de erro do EuroSAT** (`figures.error_examples`): os pares mais confundidos do teste.
- **Causas:** pixels mistos, datas diferentes (referência de 2021 × imagem de 2024), estação
  seca e a própria referência, que também tem erros (a ESA declara cerca de 77% de acurácia).
- **Por região:** o modelo europeu chama a caatinga de "vegetação herbácea" e "cultura
  permanente". É um exemplo de **transferência entre domínios** que não funciona.

---

## A interface e a apresentação

- **Interface** (`src/app.py` + `web/`): um servidor Flask que carrega os modelos e responde a
  pedidos da página. Cada clique roda o modelo de verdade e devolve o tempo gasto. Detalhes no
  [guia da demonstração](guia-demonstracao.md).
- **Slides** (`presentation/index.html`): Reveal.js, offline. Os números vêm de
  `presentation/data/dados.js`, gerado por `presentation/scripts/exportar_dados.py` a partir de
  `results/metrics/`. Se o modelo for retreinado, basta rodar o script de novo.

## Resultados em uma tabela

| experimento | acurácia | kappa | F1-macro | observação |
|---|---|---|---|---|
| imagem inteira · EuroSAT · RF | 93,2% | 0,924 | 92,9% | SVM: 95,7%; k-NN: 91,1%; RF só RGB: 85,6% |
| pixel · Sentinel-2 | 88,6% | 0,606 | 64,2% | com filtro de moda: 89,2% / 0,620 / 65,1% |
| pixel · CBERS-4 | 86,1% | 0,511 | 56,8% | urbano cai de 0,78 para 0,53 sem SWIR |
| região · EuroSAT → Mossoró | 62,8% de concordância | — | — | modelo europeu na caatinga |
| vazamento (S2, divisão aleatória) | 89,3% | — | 73,0% | otimismo de quase 9 p.p. no F1-macro |

## Como rodar tudo

```bash
cd trab-10
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
./run.sh              # download → treino → região → figuras → dados dos slides → testes → interface
# ou passo a passo:
.venv/bin/python src/data_loader.py
.venv/bin/python src/train.py
.venv/bin/python src/predict.py
.venv/bin/python src/evaluation.py
.venv/bin/python presentation/scripts/exportar_dados.py
.venv/bin/python -m pytest tests -q
.venv/bin/python src/app.py          # http://localhost:5000
```

## Organização do seminário

| # | integrante | slides | tema | tempo |
|---|---|---|---|---|
| 1 | Ana Beatriz | 1–6 | contexto, sensoriamento, dados, área, classes | ~5 min |
| 2 | Cindy Vitoria | 7–11 | metodologia, bandas, composições, pré-processamento, amostras | ~5 min |
| 3 | Eduardo Marinho | 12–15 | características, índices, histogramas, importância | ~4 min |
| 4 | Luis Jeronimo | 16–19 | Random Forest, métricas, matriz de confusão, comparação | ~4,5 min |
| 5 | Paulo Sérgio | 20–27 | resultados, mapa, erros, granularidades, demonstração, conclusão, referências, agradecimento | ~7 min |

**Teclas:** `→` avança · `S` visão do apresentador (notas + cronômetro) · `F` tela cheia ·
`T` tema claro/escuro · `Esc` visão geral.

**Capa:** a faixa colorida embaixo do título representa as **faixas do espectro** que o satélite
mede, da esquerda para a direita: azul (B02, 490 nm), verde (B03, 560 nm), vermelho (B04, 665 nm),
infravermelho próximo (B08, 842 nm) e infravermelho de ondas curtas (B11, 1610 nm). As duas
últimas o olho não vê, e são as que mais ajudam a classificar. A mesma faixa volta no slide de agradecimento.

## Glossário

| termo | significado |
|---|---|
| banda | faixa de comprimento de onda medida pelo sensor |
| DN | número digital, o valor inteiro gravado no arquivo |
| reflectância | fração da luz refletida (0–1) |
| L1C / SR | topo da atmosfera / superfície |
| NIR / SWIR | infravermelho próximo / de ondas curtas |
| índice espectral | combinação de bandas que realça um alvo |
| GLCM | matriz de coocorrência: conta pares de tons vizinhos para medir textura |
| Sobel | filtro que mede bordas (variação de intensidade) |
| característica | número que descreve uma amostra |
| rótulo / referência | a classe "verdadeira" de uma amostra |
| treino / validação / teste | aprender / escolher ajustes / medir o resultado final |
| vazamento | o teste "vê" algo do treino e a métrica fica otimista |
| bootstrap | sorteio com reposição |
| hiperparâmetro | ajuste do algoritmo escolhido antes do treino (ex.: nº de árvores) |
| filtro de moda | troca o rótulo pelo mais frequente da vizinhança |
| STAC | padrão de catálogo de imagens de satélite usado pelo INPE |
| COG | GeoTIFF otimizado para ler pedaços pela internet |

## Perguntas que qualquer um pode receber

**"Por que não usaram só imagens do INPE?"** As duas cenas classificadas são do INPE. O que o
INPE não tem são rótulos por pixel para Mossoró, por isso usamos o WorldCover como gabarito; o
EuroSAT é o conjunto rotulado para a classificação da imagem inteira.

**"O WorldCover não é um mapa pronto? Vocês copiaram?"** Ele é o gabarito. O modelo aprende
com a **imagem** dos blocos de treino e é avaliado em blocos que nunca viu. A referência tem
erros e é de outra data, então parte do "erro" medido é dela.

**"Por que a acurácia é alta e o F1-macro é baixo?"** 82% dos pixels são vegetação. A acurácia é
dominada por ela; o F1-macro dá o mesmo peso às cinco classes.

**"Os números são reais?"** Sim, vêm de `results/metrics/*.json` e são lidos automaticamente
pelos slides. A interface classifica ao vivo.

**"Por que Random Forest e não deep learning?"** É explicável, treina em segundos na CPU, não exige
normalização, mostra a importância das características e classifica 6,2 milhões de pixels em 27 s.
Uma CNN é uma melhoria natural.

**"O que mais melhoraria o resultado?"** Uma segunda data na estação chuvosa (separa caatinga de
agricultura), rótulos de campo e uma referência regional como o MapBiomas.
