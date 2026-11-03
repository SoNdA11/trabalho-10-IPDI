# Trabalho 10 — Classificação de Imagens de Satélite

Estudo de caso de **classificação de cobertura do solo em Mossoró/RN** com imagens
obtidas no catálogo do **INPE** (Sentinel-2 L1C e CBERS-4 MUX, via Brazil Data Cube),
o conjunto rotulado **EuroSAT MS** e um classificador **Random Forest**.

O projeto implementa as três granularidades de classificação e mede cada uma
no conjunto de teste, com divisão **espacial** para evitar vazamento:

| nível | dados | classes | resultado no teste |
| --- | --- | --- | --- |
| **imagem inteira** (1 recorte → 1 classe) | EuroSAT MS, 27.000 recortes 64×64, 13 bandas | 10 | acurácia **93,2%** · kappa 0,924 · F1-macro 92,9% |
| **por pixel** (mapa temático) | Sentinel-2B L1C do INPE, 01/09/2024, 10 m | 5 | acurácia **88,6%** · kappa 0,606 · F1-macro 64,2% (89,2% com filtro de moda) |
| **por pixel** | CBERS-4 MUX L4 SR do INPE, 04/10/2024, 20 m | 5 | acurácia **86,1%** · kappa 0,511 · F1-macro 56,8% |
| **por região** (janelas de 640 m) | modelo do EuroSAT aplicado à cena do INPE | 4 macro | concordância de 62,8% com a referência (transferência Europa → caatinga) |

Classes do estudo de caso: **Vegetação, Área agrícola, Área urbana, Solo exposto e Água**
(rótulos do ESA WorldCover 2021). Os números acima estão em `results/metrics/*.json`
e são regenerados a cada treino.

---

## Estrutura

```text
trab-10/
├── README.md · requirements.txt · run.sh · Dockerfile · docker-compose.yml
├── src/
│   ├── config.py              caminhos, cenas do INPE (IDs STAC), classes, cores, bandas
│   ├── data_loader.py         1. aquisição: EuroSAT MS, STAC do INPE, WorldCover; recorte e alinhamento
│   ├── preprocessing.py       3. DN→reflectância, harmonização L1C, máscaras, realce, filtros
│   ├── feature_extraction.py  4. índices (NDVI, NDWI, MNDWI, NDBI, BSI), estatísticas, GLCM, contexto 5×5
│   ├── sampling.py            5. divisão espacial (grupos/blocos) e amostragem dos pixels
│   ├── train.py               6. treino, ajuste na validação, comparações e estudos de vazamento
│   ├── evaluation.py          7. métricas (acurácia, precisão, recall, F1, kappa, matriz de confusão)
│   ├── predict.py             8. inferência: recorte, janela, cena inteira e mapa por região
│   ├── figures.py             2/9. figuras de exploração, mapas e análise de erros
│   └── app.py                 interface: API Flask
├── web/                       interface de demonstração (HTML/CSS/JS, sem CDN)
├── tests/                     pytest: índices, pré-processamento, divisão espacial, API
├── data/                      raw/ e processed/ (gerados; ver data/README.md)
├── models/                    modelos .joblib (gerados)
├── results/metrics/           métricas de teste em JSON (versionadas)
├── presentation/              seminário em Reveal.js, 27 slides, offline
└── docs/                      metodologia, referências, guias dos integrantes, miniaturas dos slides
```

## Instalação

Python 3.11+ (testado com 3.14). As wheels do `rasterio` já incluem o GDAL.

```bash
cd trab-10
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Obtenção dos dados

```bash
.venv/bin/python src/data_loader.py          # tudo
.venv/bin/python src/data_loader.py eurosat  # só o EuroSAT MS (≈2 GB, Zenodo)
.venv/bin/python src/data_loader.py inpe     # só as cenas do INPE + WorldCover
```

- **Sentinel-2 L1C:** item `S2B_MSIL1C_20240901T125309_N0511_R052_T24MXV_20240901T132229`
  da coleção `S2_L1C_BUNDLE-1` (≈700 MB, zip SAFE). As 13 bandas são recortadas na área
  de estudo e reamostradas para 10 m → `data/processed/s2_mossoro_l1c_10m.tif`.
- **CBERS-4 MUX:** item `CBERS_4_MUX_20241004_149_106_L4` da coleção `CB4-MUX-L4-SR-1`.
  Só a janela da área de estudo é lida direto dos COGs (sem baixar a cena inteira).
- **WorldCover 2021 v200:** tile `S06W039` (60 MB), reprojetado para a grade de cada cena.

Nenhum cadastro é necessário: o catálogo STAC do INPE (`https://data.inpe.br/bdc/stac/v1`)
é aberto. Detalhes e licenças em [`data/README.md`](data/README.md).

## Treinamento e avaliação

```bash
.venv/bin/python src/train.py              # os três experimentos (≈6 min em 8 núcleos)
.venv/bin/python src/train.py eurosat      # só um: eurosat | s2 | cbers
.venv/bin/python src/predict.py            # classificação por região (janela deslizante)
.venv/bin/python src/evaluation.py         # figuras → results/figures e presentation/img
.venv/bin/python -m pytest tests -q        # 24 testes
```

Ou tudo de uma vez: `./run.sh`.

O treino escolhe os hiperparâmetros do Random Forest pela **validação**, retreina e só
então mede no **teste**. Ele também roda, no mesmo teste: SVM (RBF) e k-NN, o modelo
"só RGB", a divisão aleatória (para medir o otimismo do vazamento espacial) e três
estratégias de amostragem dos pixels.

## Interface de demonstração

```bash
.venv/bin/python src/app.py      # http://localhost:5000
```

- **Imagem inteira:** escolha um recorte do **conjunto de teste** do EuroSAT ou envie um
  GeoTIFF de 13 bandas (formato EuroSAT MS) ou um JPG/PNG (modelo só RGB). Mostra
  composições, classe, votos das árvores, índices e características.
- **Mapa por pixel:** clique na cena do INPE (Sentinel-2 ou CBERS-4). A janela é
  classificada na hora e comparada com a referência; o pixel clicado mostra suas
  características. "Classificar a cena inteira" roda o modelo nos 6,2 milhões de pixels (≈30 s).
- **Métricas:** valores de teste, matrizes de confusão e comparações.

Todas as respostas vêm do modelo treinado e informam o tempo de inferência.
Com Docker: `docker compose up --build`, depois de gerar `data/`, `models/` e `results/`.

## Apresentação

Abra `presentation/index.html` no navegador (funciona por `file://`, sem servidor e sem
internet). Teclas: `→`/`espaço` avança, `←` volta, `S` abre a visão do apresentador
(notas e cronômetro), `F` tela cheia, `Esc` visão geral e `T` alterna o tema claro/escuro
(fica salvo no navegador). Os números dos slides vêm de `presentation/data/dados.js`, gerado por
`presentation/scripts/exportar_dados.py` a partir de `results/metrics/`.

```bash
cd presentation && npm install
npm run miniaturas   # docs/img/slide-XX.jpg (usadas nos guias)
npm run pdf          # PDF de contingência
```

## Documentação

- [`docs/metodologia.md`](docs/metodologia.md): o experimento, etapa por etapa, e as decisões tomadas.
- [`docs/referencias.md`](docs/referencias.md): fontes de dados e referências acadêmicas.
- [`docs/guia-geral.md`](docs/guia-geral.md): **a implementação inteira, passo a passo**, com conceitos,
  código e exemplos reais.
- [`docs/guia-demonstracao.md`](docs/guia-demonstracao.md): como usar e interpretar cada tela da interface.
- Guias individuais: [Ana](docs/guia-ana.md) · [Cindy](docs/guia-cindy.md) ·
  [Eduardo](docs/guia-eduardo.md) · [Luis](docs/guia-luis.md) · [Paulo](docs/guia-paulo.md)

## Limitações conhecidas

- Os rótulos dos pixels vêm do WorldCover 2021, que também é um produto de classificação;
  a imagem é de 2024. Parte do "erro" é a discordância com essa referência.
- Uma única data, na estação seca: a caatinga sem folhas se confunde com solo e agricultura.
- As classes raras (agrícola, solo exposto) têm F1 ≈ 0,4.
- O modelo do EuroSAT (Europa) não transfere bem para o bioma caatinga; a classificação
  por região mostra esse limite, não a acurácia do método.
