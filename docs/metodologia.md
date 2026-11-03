# Metodologia

Este documento descreve o experimento etapa por etapa: o que foi feito, por que foi feito
assim e onde está o código. Todos os números saem de `results/metrics/*.json` (gerados por
`src/train.py` e `src/predict.py`).

## 0. Pergunta e escopo

**Pergunta:** é possível identificar automaticamente a cobertura do solo de Mossoró/RN
em imagens de satélite do INPE, com um classificador que dê para explicar em sala?

**Classes consideradas no início:** vegetação, água, área urbana, solo exposto e área agrícola.

**Viabilidade, verificada nos dados antes de treinar:**

| classe | referência disponível | experimento |
| --- | --- | --- |
| Vegetação | WorldCover (árvores, arbustos, herbácea, áreas úmidas, mangue) · EuroSAT (floresta, vegetação herbácea) | pixel e imagem |
| Área agrícola | WorldCover (cropland) · EuroSAT (cultura anual, permanente, pastagem) | pixel e imagem |
| Área urbana | WorldCover (built-up) · EuroSAT (residencial, industrial, rodovia) | pixel e imagem |
| Água | WorldCover (água permanente) · EuroSAT (rio, mar/lago) | pixel e imagem |
| Solo exposto | **só** WorldCover (bare/sparse vegetation) | só pixel |

Solo exposto não existe no EuroSAT. Em vez de inventar rótulos, a classe só entra no
experimento por pixel.

## 1. Aquisição (`src/data_loader.py`)

| dado | identificação | por que foi escolhido |
| --- | --- | --- |
| Sentinel-2B MSI L1C | INPE/BDC, `S2_L1C_BUNDLE-1`, item `S2B_MSIL1C_20240901T125309_N0511_R052_T24MXV_20240901T132229`, 01/09/2024, 0% de nuvens no tile | 13 bandas (inclui SWIR); mesmo nível de processamento (L1C) do EuroSAT |
| CBERS-4 MUX L4 SR | INPE/BDC, `CB4-MUX-L4-SR-1`, item `CBERS_4_MUX_20241004_149_106_L4`, 04/10/2024, 2,3% de nuvens | satélite sino-brasileiro; comparação com um sensor sem SWIR |
| EuroSAT MS | Zenodo 7711810 (Helber et al., 2019) | único conjunto público de recortes Sentinel-2 com 13 bandas e rótulos manuais |
| ESA WorldCover 2021 v200 | tile S06W039 | rótulos por pixel, 10 m, mesma resolução do Sentinel-2 |

A área de estudo é uma caixa de 25,6 × 24,4 km (lon −37,45 a −37,22; lat −5,30 a −5,08)
que reúne cidade, Rio Apodi-Mossoró, açudes, salinas, agricultura, caatinga e solo exposto.
`aoi_grid` define uma grade UTM 24S alinhada a múltiplos de 20 m, comum às duas cenas.

**Por que não só INPE?** O INPE fornece as imagens, mas não um conjunto de pixels
rotulados para Mossoró. Rótulos de campo não existem para a área; por isso usamos o
WorldCover como referência e o EuroSAT como conjunto rotulado de recortes. A limitação
está documentada na seção 9.

## 2. Exploração (`src/figures.py`, slides 3, 5, 6 e 8)

- Composições cor verdadeira, falsa cor (NIR) e SWIR da cena.
- Assinaturas espectrais médias por classe (20.000 pixels por classe).
- Proporção das classes: 82,4% vegetação, 6,8% urbana, 4,3% agrícola, 3,7% água e
  2,9% solo exposto. O desbalanceamento orienta a amostragem (seção 5) e a escolha das métricas (seção 7).
- Boxplots dos índices por classe: nenhum índice isolado separa as cinco classes.

## 3. Pré-processamento (`src/preprocessing.py`)

| operação | onde | finalidade |
| --- | --- | --- |
| recorte da AOI e **alinhamento** das bandas 20/60 m → 10 m (bilinear) | `prepare_s2_scene` | empilhar as 13 bandas numa grade única |
| reprojeção da referência: vizinho mais próximo (10 m) e **moda** (20 m) | `prepare_reference` | rótulo é categórico: média de rótulos não existe |
| DN → reflectância, com **harmonização** da baseline L1C (offset −1000) | `to_reflectance`, `harmonize_l1c` | desde a baseline 04.00 (2022) o L1C soma 1000 aos DN; o EuroSAT (2017–2018) não |
| máscara de validade: DN 0/65535 (S2), −9999 e **CMASK** do INPE (CBERS, 8,7% de nuvem) | `valid_mask` | excluir pixels sem dado ou com nuvem do treino, do teste e do mapa |
| NaN/inf → mediana; faixa física [0; 1,5] | `fix_invalid` | evitar divisões inválidas nos índices |
| realce por percentis (com limites comuns às três bandas na cor verdadeira) | `stretch`, `composite` | **só visualização** |
| z-score ajustado no treino | `Pipeline(StandardScaler, …)` | SVM e k-NN dependem de escala; o RF não |
| filtro de moda 3×3 | `majority_filter` | pós-classificação: remove pixels isolados |
| filtro de mediana | `denoise_median` | disponível; não foi necessário (dados L1C sem ruído impulsivo) |

## 4. Extração de características (`src/feature_extraction.py`)

Os índices são calculados por **papel físico** da banda (blue, green, red, nir, swir1, swir2).
Se o sensor não tem a banda exigida, `spectral_index` levanta erro; nenhum índice é aproximado.

| índice | fórmula | sensores |
| --- | --- | --- |
| NDVI (Rouse et al., 1974) | (NIR − Red)/(NIR + Red) | Sentinel-2, CBERS-4 |
| NDWI (McFeeters, 1996) | (Green − NIR)/(Green + NIR) | Sentinel-2, CBERS-4 |
| MNDWI (Xu, 2006) | (Green − SWIR1)/(Green + SWIR1) | Sentinel-2 |
| NDBI (Zha et al., 2003) | (SWIR1 − NIR)/(SWIR1 + NIR) | Sentinel-2 |
| BSI | ((SWIR1 + Red) − (NIR + Blue))/((SWIR1 + Red) + (NIR + Blue)) | Sentinel-2 |

### Por recorte (imagem inteira): 97 características (27 no modelo só RGB)

- 13 bandas × {média, desvio, p10, p50, p90} = 65
- 5 índices × {média, desvio, p10, p90} = 20
- histograma do NDVI em 6 faixas = 6
- textura na banda NIR: GLCM (Haralick et al., 1973; 16 níveis, distância 1, 0° e 90°:
  contraste, homogeneidade, energia, correlação), média do gradiente de Sobel e variância local 3×3 = 6

### Por pixel: 22 características (Sentinel-2) ou 10 (CBERS-4)

- reflectância de cada banda (13 ou 4)
- índices disponíveis (5 ou 2)
- contexto 5×5: média e desvio do NDVI e do NIR (4)

## 5. Preparação das amostras (`src/sampling.py`)

**Vazamento espacial:** amostras vizinhas são quase idênticas. Por isso a divisão é feita
por grupos geográficos, nunca por amostra.

- **EuroSAT:** centro de cada recorte → células de 0,25° (2.047 células).
  `StratifiedGroupKFold` (7 dobras): 1 dobra para teste, 1 para validação e o resto para treino,
  o que dá 19.285 / 3.857 / 3.858 recortes. Um teste automatizado garante que nenhum grupo se repete.
- **Cenas do INPE:** blocos de 1,28 km (128 px no S2, 64 px no CBERS) sorteados em
  60/20/20. Uma faixa de 2 px na borda de cada bloco é descartada: a janela 5×5 das
  características não "enxerga" pixels de outro conjunto.
- **Amostragem do treino (pixels):** 40.000 pixels, proporcionais à **raiz da frequência**
  de cada classe (mínimo de 1.500 por classe). A validação é uma amostra aleatória
  (mesma distribuição do teste). O teste usa **todos** os pixels dos blocos de teste:
  1.096.671 no S2 e 238.314 no CBERS.

O estudo de amostragem (mesmo modelo, mesmo teste, `s2_pixel.json → sampling_study`)
justifica a escolha:

| amostragem do treino | acurácia | F1-macro |
| --- | --- | --- |
| balanceada (8.000 por classe) | 77,5% | 58,4% |
| proporcional (aleatória) | 91,2% | 60,0% |
| **raiz da frequência (usada)** | 88,6% | **64,2%** |

## 6. Treinamento (`src/train.py`)

**Algoritmo principal:** Random Forest (Breiman, 2001), `scikit-learn`.
Hiperparâmetros escolhidos pelo F1-macro **na validação**, numa grade de 4 combinações:

| `max_features` | `min_samples_leaf` | EuroSAT (F1 val.) | Sentinel-2 (F1 val.) |
| --- | --- | --- | --- |
| **sqrt** | **1** | **0,926** | **0,718** |
| sqrt | 3 | 0,920 | 0,716 |
| 0,3 | 1 | 0,920 | 0,717 |
| 0,3 | 3 | 0,920 | 0,716 |

EuroSAT: 300 árvores, `class_weight="balanced"`, retreino em treino + validação.
Cenas: 200 árvores, sem pesos (a amostragem já trata o desbalanceamento). O CBERS escolheu
`min_samples_leaf=3`.

### Justificativa da escolha do Random Forest

| critério | Random Forest |
| --- | --- |
| dados | mistura de escalas (reflectância, índices, textura) sem normalizar |
| amostras | 19 mil recortes, 40 mil pixels: treina em segundos |
| custo | **16,8 ms por mil amostras** na predição, contra 388,8 ms do SVM; o mapa de 6,2 milhões de pixels sai em 27 s |
| interpretabilidade | importância das características; a votação vira "probabilidade" |
| desempenho | 93,2% no EuroSAT, 2,5 p.p. abaixo do SVM (95,7%) e acima do k-NN (91,1%) |

O SVM foi melhor no EuroSAT, e isso está nos slides. Uma ressalva sobre o tempo: o RF
usa os 8 núcleos (`n_jobs=-1`) e o SVM do scikit-learn usa um só. Mesmo assim, o custo de
predição do SVM com kernel RBF cresce com o número de vetores de suporte, o que pesa num
mapa de milhões de pixels.

## 7. Avaliação (`src/evaluation.py`)

Métricas medidas só no teste: acurácia, precisão, recall, F1 (por classe, média macro
e ponderada), kappa de Cohen e matriz de confusão.

| experimento | n teste | acurácia | kappa | F1-macro |
| --- | --- | --- | --- | --- |
| EuroSAT — imagem inteira, 10 classes | 3.858 | 93,2% | 0,924 | 92,9% |
| EuroSAT agrupado em 4 macro-classes | 3.858 | 96,6% | — | — |
| Sentinel-2 — pixel, 5 classes | 1.096.671 | 88,6% | 0,606 | 64,2% |
| Sentinel-2 + filtro de moda 3×3 | 1.096.671 | 89,2% | 0,620 | 65,1% |
| CBERS-4 — pixel, 5 classes | 238.314 | 86,1% | 0,511 | 56,8% |
| CBERS-4 + filtro de moda 3×3 | 238.314 | 87,4% | 0,537 | 58,6% |

**F1 por classe (pixel):**

| classe | Sentinel-2 | CBERS-4 |
| --- | --- | --- |
| Vegetação | 0,942 | 0,933 |
| Área agrícola | 0,433 | 0,382 |
| Área urbana | 0,782 | 0,527 |
| Solo exposto | 0,446 | 0,356 |
| Água | 0,609 | 0,645 |

**Efeito do vazamento (divisão aleatória × espacial, mesmo modelo):**

| experimento | aleatória | espacial |
| --- | --- | --- |
| EuroSAT, acurácia | 94,0% | 93,2% |
| Sentinel-2, F1-macro | 73,0% | 64,2% |
| CBERS-4, F1-macro | 63,6% | 56,8% |

**Ablação de bandas (EuroSAT, mesmo RF):** só RGB → 85,6%; 13 bandas → 93,2% (+7,5 p.p.).

## 8. Mapa de classificação (`src/predict.py`)

- **Por pixel:** `classify_scene` processa a cena em faixas de 256 linhas, prediz todos os
  pixels válidos e aplica o filtro de moda. Saídas: `results/predictions/s2_map.tif`
  (GeoTIFF com paleta), `.npy` e as figuras `s2_mapa.png` e `cbers_mapa.png`.
- **Por região:** `region_map` desliza uma janela de 64×64 px (passo 32) sobre o DN
  harmonizado do Sentinel-2 e classifica cada janela com o modelo do EuroSAT. Foram 5.850
  janelas em 12 s. Comparadas com a classe WorldCover majoritária de cada janela
  (4 macro-classes; solo exposto excluído), a concordância foi de **62,8%**.
- **Imagem inteira:** `classify_chip` devolve uma classe para o recorte inteiro.

| granularidade | unidade | modelo | o que responde |
| --- | --- | --- | --- |
| imagem inteira | recorte 640 m | EuroSAT (10 classes) | "o que há neste recorte?" |
| região | janela 640 m sobre a cena | EuroSAT aplicado à cena | "o que há em cada bloco da cena?" |
| pixel | 10 m (S2) ou 20 m (CBERS) | treinado na cena | "o que há em cada ponto?" |

## 9. Análise dos erros

- **Confusões do EuroSAT** (teste): cultura anual → permanente (42), permanente →
  vegetação herbácea (17), residencial → rodovia (14), rodovia → pastagem (14),
  industrial → rodovia (14). São classes com cor parecida; a textura ajuda, mas não resolve tudo.
- **Confusões da cena:** agrícola ↔ vegetação (caatinga seca × pousio), água → solo
  exposto (açudes que secaram entre 2021 e 2024, salinas), solo exposto → urbano
  (sem SWIR no CBERS, essa confusão aumenta).
- **Causas:** pixels mistos nas bordas; diferença de 3 anos entre a referência e a imagem;
  estação seca; e a própria referência, que é um produto de classificação (a ESA declara
  acurácia global em torno de 77% para o WorldCover 2021).
- **Por região:** o modelo europeu chama a caatinga de "vegetação herbácea" (3.116 janelas)
  e "cultura permanente" (1.202). Acerta 79% das janelas urbanas, mas só 60% das de vegetação.

## 10. Reprodutibilidade

- Semente fixa (`config.SEED = 42`) em todas as divisões, amostragens e modelos.
- Cenas fixadas pelo ID do item no STAC do INPE.
- `./run.sh` refaz tudo, do download aos testes (24 testes, `pytest`).
- Hardware dos números acima: 8 núcleos, 15 GB de RAM, sem GPU.
