"""
config.py — Configuração central do projeto (caminhos, dados, classes e cores).

Tudo o que identifica os dados usados no estudo de caso fica aqui, num só lugar,
para que o experimento seja reprodutível: as cenas do INPE são fixadas pelo
identificador do item no catálogo STAC do Brazil Data Cube.
"""

from pathlib import Path

# ─── Caminhos ─────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
METRICS = RESULTS / "metrics"
PREDICTIONS = RESULTS / "predictions"
WEB_DIR = ROOT / "web"
PRESENTATION_DIR = ROOT / "presentation"

SEED = 42

# ─── EuroSAT (Sentinel-2, 13 bandas) ──────────────────────────────────────────
# Helber et al. (2019). Versão multiespectral: GeoTIFF 64×64 px, 13 bandas L1C.

EUROSAT_MS_URL = "https://zenodo.org/records/7711810/files/EuroSAT_MS.zip?download=1"
EUROSAT_MS_ZIP = DATA_RAW / "EuroSAT_MS.zip"
EUROSAT_MS_DIR = DATA_RAW / "EuroSAT_MS"

# Ordem das bandas dentro de cada GeoTIFF do EuroSAT MS.
EUROSAT_BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07",
                 "B08", "B09", "B10", "B11", "B12", "B8A"]

EUROSAT_CLASSES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake",
]

EUROSAT_PT = {
    "AnnualCrop": "Cultura anual",
    "Forest": "Floresta",
    "HerbaceousVegetation": "Vegetação herbácea",
    "Highway": "Rodovia",
    "Industrial": "Industrial",
    "Pasture": "Pastagem",
    "PermanentCrop": "Cultura permanente",
    "Residential": "Residencial",
    "River": "Rio",
    "SeaLake": "Mar / lago",
}

EUROSAT_COLORS = {
    "AnnualCrop": "#E3B505",
    "Forest": "#1B7F3B",
    "HerbaceousVegetation": "#7BC96F",
    "Highway": "#8A8F98",
    "Industrial": "#C2185B",
    "Pasture": "#C9D86B",
    "PermanentCrop": "#E07B39",
    "Residential": "#7E57C2",
    "River": "#3D8BFD",
    "SeaLake": "#0D3B8C",
}

# Agrupamento das 10 classes do EuroSAT nas categorias do trabalho.
# Pastagem é uso agropecuário do solo; vegetação herbácea é cobertura natural.
EUROSAT_TO_MACRO = {
    "AnnualCrop": "Área agrícola",
    "PermanentCrop": "Área agrícola",
    "Pasture": "Área agrícola",
    "Forest": "Vegetação",
    "HerbaceousVegetation": "Vegetação",
    "Highway": "Área urbana",
    "Industrial": "Área urbana",
    "Residential": "Área urbana",
    "River": "Água",
    "SeaLake": "Água",
}

# ─── Classes de cobertura do solo do estudo de caso (Mossoró/RN) ─────────────
# Referência: ESA WorldCover 2021 v200 (10 m). Códigos originais → classe do trabalho.

LANDCOVER_CLASSES = ["Vegetação", "Área agrícola", "Área urbana", "Solo exposto", "Água"]

LANDCOVER_COLORS = {
    "Vegetação": "#2E8B3E",
    "Área agrícola": "#E9B824",
    "Área urbana": "#D63A4A",
    "Solo exposto": "#B07A4F",
    "Água": "#2F6BD8",
}

MACRO_COLORS = LANDCOVER_COLORS  # as macro-classes do EuroSAT usam as mesmas cores

WORLDCOVER_TO_CLASS = {
    10: "Vegetação",      # Tree cover
    20: "Vegetação",      # Shrubland (caatinga arbustiva)
    30: "Vegetação",      # Grassland (vegetação herbácea)
    90: "Vegetação",      # Herbaceous wetland
    95: "Vegetação",      # Mangroves
    40: "Área agrícola",  # Cropland
    50: "Área urbana",    # Built-up
    60: "Solo exposto",   # Bare / sparse vegetation
    80: "Água",           # Permanent water bodies
}

WORLDCOVER_NAMES = {
    10: "Tree cover", 20: "Shrubland", 30: "Grassland", 40: "Cropland",
    50: "Built-up", 60: "Bare / sparse vegetation", 70: "Snow and ice",
    80: "Permanent water bodies", 90: "Herbaceous wetland", 95: "Mangroves",
    100: "Moss and lichen",
}

WORLDCOVER_URL = ("https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
                  "ESA_WorldCover_10m_2021_v200_S06W039_Map.tif")
WORLDCOVER_TIF = DATA_RAW / "worldcover" / "ESA_WorldCover_10m_2021_v200_S06W039_Map.tif"

# ─── Área de estudo ───────────────────────────────────────────────────────────
# Mossoró/RN e entorno: área urbana, Rio Apodi-Mossoró, açudes, agricultura
# irrigada, caatinga e solo exposto. Caixa em graus (lon_min, lat_min, lon_max, lat_max).

AOI_NAME = "Mossoró/RN"
AOI_BBOX = (-37.45, -5.30, -37.22, -5.08)
AOI_CRS = "EPSG:32724"  # UTM 24S, o mesmo das cenas Sentinel-2 e CBERS-4

# ─── Catálogo do INPE (Brazil Data Cube, STAC) ────────────────────────────────

INPE_STAC = "https://data.inpe.br/bdc/stac/v1"

S2_COLLECTION = "S2_L1C_BUNDLE-1"
S2_ITEM = "S2B_MSIL1C_20240901T125309_N0511_R052_T24MXV_20240901T132229"
S2_ZIP = DATA_RAW / "inpe" / f"{S2_ITEM}.zip"
S2_DATE = "2024-09-01"
S2_STACK = DATA_PROCESSED / "s2_mossoro_l1c_10m.tif"
S2_REF = DATA_PROCESSED / "s2_mossoro_worldcover_10m.tif"
# A partir da baseline 04.00 (jan/2022) o L1C soma 1000 aos números digitais.
# O EuroSAT (2017–2018) não tem esse deslocamento: subtraímos para harmonizar.
S2_RADIO_ADD_OFFSET = -1000
S2_QUANTIFICATION = 10000.0

CBERS_COLLECTION = "CB4-MUX-L4-SR-1"
CBERS_ITEM = "CBERS_4_MUX_20241004_149_106_L4"
CBERS_DATE = "2024-10-04"
CBERS_BANDS = ["BAND5", "BAND6", "BAND7", "BAND8"]  # azul, verde, vermelho, NIR
CBERS_STACK = DATA_PROCESSED / "cbers4_mux_mossoro_20m.tif"
CBERS_REF = DATA_PROCESSED / "cbers4_mux_worldcover_20m.tif"
CBERS_SCALE = 1e-4  # reflectância de superfície × 10000 (int16)
CBERS_NODATA = -9999
# Máscara de nuvem do INPE (CMASK): 0 = fora da cena, 127 = limpo, 255 = nuvem/sombra
CBERS_CMASK_CLEAR = 127

# ─── Sensores: nome da banda → posição no empilhamento ───────────────────────
# Os índices espectrais procuram as bandas por nome físico (blue, green, red,
# nir, swir1, swir2). Se a banda não existe no sensor, o índice não é calculado.

SENSOR_BANDS = {
    # Sentinel-2 MSI na ordem do EuroSAT
    "sentinel2": {
        "names": EUROSAT_BANDS,
        "roles": {"blue": "B02", "green": "B03", "red": "B04", "nir": "B08",
                  "swir1": "B11", "swir2": "B12"},
        "wavelength_nm": {"B01": 443, "B02": 490, "B03": 560, "B04": 665, "B05": 705,
                          "B06": 740, "B07": 783, "B08": 842, "B09": 945, "B10": 1375,
                          "B11": 1610, "B12": 2190, "B8A": 865},
        "resolution_m": {"B01": 60, "B02": 10, "B03": 10, "B04": 10, "B05": 20,
                         "B06": 20, "B07": 20, "B08": 10, "B09": 60, "B10": 60,
                         "B11": 20, "B12": 20, "B8A": 20},
    },
    # Só as três bandas visíveis do Sentinel-2 (experimento de ablação "RGB")
    "rgb": {
        "names": ["B04", "B03", "B02"],
        "roles": {"red": "B04", "green": "B03", "blue": "B02"},
    },
    # CBERS-4 MUX (INPE/CAST), 20 m
    "cbers4_mux": {
        "names": CBERS_BANDS,
        "roles": {"blue": "BAND5", "green": "BAND6", "red": "BAND7", "nir": "BAND8"},
        "wavelength_nm": {"BAND5": 485, "BAND6": 555, "BAND7": 660, "BAND8": 830},
        "resolution_m": {b: 20 for b in CBERS_BANDS},
    },
}
