"""
data_loader.py — Carregamento e organização do dataset EuroSAT Multispectral

Dataset: EuroSAT (Sentinel-2 MSI, 13 bandas espectrais, 27.000 imagens 64×64 px)
Fonte:   Zenodo — https://zenodo.org/records/7711810
Artigo:  Helber et al. (2019) "EuroSAT: A Novel Dataset and Deep Learning Benchmark
         for Land Use and Land Cover Classification"

O dataset EuroSAT usa imagens do satélite Sentinel-2 (ESA), o mesmo sensor adotado
pelo INPE no Brazil Data Cube (BDC) para mapeamento de cobertura do solo brasileiro.
"""

import os
import zipfile
import io
import json
import struct
import zlib
import numpy as np
import requests
from pathlib import Path
from tqdm import tqdm
from PIL import Image

# ─── Configurações ────────────────────────────────────────────────────────────

ROOT = Path(__file__).parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"

# EuroSAT RGB (95 MB) — usado para visualização e composição colorida
EUROSAT_RGB_URL = "https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip?download=1"
EUROSAT_RGB_ZIP = DATA_RAW / "EuroSAT_RGB.zip"
EUROSAT_RGB_DIR = DATA_RAW / "EuroSAT_RGB"

# Classes do EuroSAT e mapeamento para categorias do trabalho
CLASS_NAMES = [
    "AnnualCrop",       # Cultura anual (área agrícola)
    "Forest",           # Floresta (vegetação)
    "HerbaceousVegetation",  # Vegetação herbácea (vegetação)
    "Highway",          # Rodovia (área urbana)
    "Industrial",       # Industrial (área urbana)
    "Pasture",          # Pastagem (vegetação / área agrícola)
    "PermanentCrop",    # Cultura permanente (área agrícola)
    "Residential",      # Residencial (área urbana)
    "River",            # Rio (água)
    "SeaLake",          # Mar/lago (água)
]

# Mapeamento para 5 macro-classes do trabalho
MACRO_CLASSES = {
    "AnnualCrop":            "Área Agrícola",
    "Forest":                "Vegetação",
    "HerbaceousVegetation":  "Vegetação",
    "Highway":               "Área Urbana",
    "Industrial":            "Área Urbana",
    "Pasture":               "Área Agrícola",
    "PermanentCrop":         "Área Agrícola",
    "Residential":           "Área Urbana",
    "River":                 "Água",
    "SeaLake":               "Água",
}

MACRO_CLASS_NAMES = ["Água", "Área Agrícola", "Área Urbana", "Vegetação"]

# Cores para visualização de cada classe
CLASS_COLORS = {
    "AnnualCrop":           "#F4D03F",  # amarelo
    "Forest":               "#1E8449",  # verde escuro
    "HerbaceousVegetation": "#58D68D",  # verde claro
    "Highway":              "#808B96",  # cinza
    "Industrial":           "#E74C3C",  # vermelho
    "Pasture":              "#A9DFBF",  # verde claro
    "PermanentCrop":        "#F0B27A",  # laranja claro
    "Residential":          "#AF7AC5",  # roxo
    "River":                "#3498DB",  # azul
    "SeaLake":              "#1A5276",  # azul escuro
}

MACRO_COLORS = {
    "Água":         "#3498DB",
    "Área Agrícola": "#F4D03F",
    "Área Urbana":  "#E74C3C",
    "Vegetação":    "#1E8449",
}


# ─── Funções de download ───────────────────────────────────────────────────────

def _download_with_progress(url: str, dest: Path, desc: str = "Baixando") -> None:
    """Baixa arquivo com barra de progresso."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=desc
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))


def download_eurosat_rgb() -> None:
    """Baixa e extrai o dataset EuroSAT RGB se não existir."""
    if EUROSAT_RGB_DIR.exists() and any(EUROSAT_RGB_DIR.iterdir()):
        print(f"✓ EuroSAT RGB já existe em: {EUROSAT_RGB_DIR}")
        return

    print("📡 Baixando EuroSAT RGB (~95 MB)...")
    _download_with_progress(EUROSAT_RGB_URL, EUROSAT_RGB_ZIP, "EuroSAT RGB")

    print("📦 Extraindo...")
    with zipfile.ZipFile(EUROSAT_RGB_ZIP, "r") as zf:
        zf.extractall(DATA_RAW)

    # Normaliza estrutura de diretórios (pode variar conforme o zip)
    _normalize_eurosat_dir(DATA_RAW, "EuroSAT_RGB")
    print(f"✓ Dataset extraído em: {EUROSAT_RGB_DIR}")


def _normalize_eurosat_dir(base: Path, expected_name: str) -> None:
    """Garante que o diretório do dataset está no local esperado."""
    expected = base / expected_name
    if expected.exists():
        return
    # Procura por diretório com as classes do EuroSAT
    for item in base.iterdir():
        if item.is_dir() and any(
            (item / cls).exists() for cls in CLASS_NAMES[:3]
        ):
            item.rename(expected)
            return


# ─── Carregamento de imagens ──────────────────────────────────────────────────

def load_images_rgb(
    max_per_class: int = 500,
    img_size: int = 64,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Carrega imagens RGB do EuroSAT.

    Parâmetros
    ----------
    max_per_class : int
        Número máximo de imagens por classe (para balancear o dataset).
    img_size : int
        Tamanho das imagens carregadas (redimensionadas se necessário).
    seed : int
        Semente para reprodutibilidade na seleção aleatória.

    Retorna
    -------
    X : ndarray de shape (N, img_size, img_size, 3)
        Imagens RGB normalizadas [0, 1].
    y : ndarray de shape (N,)
        Rótulos numéricos (índice em CLASS_NAMES).
    class_names : list[str]
        Nomes das classes na ordem dos rótulos.
    """
    rng = np.random.default_rng(seed)
    X_list, y_list = [], []
    found_classes = []

    for label_idx, class_name in enumerate(CLASS_NAMES):
        class_dir = EUROSAT_RGB_DIR / class_name
        if not class_dir.exists():
            print(f"  ⚠ Classe não encontrada: {class_name}")
            continue

        found_classes.append(class_name)
        img_files = sorted(class_dir.glob("*.jpg")) + sorted(class_dir.glob("*.png"))

        # Amostragem aleatória
        if len(img_files) > max_per_class:
            indices = rng.choice(len(img_files), max_per_class, replace=False)
            img_files = [img_files[i] for i in sorted(indices)]

        for img_path in img_files:
            try:
                img = Image.open(img_path).convert("RGB")
                if img.size != (img_size, img_size):
                    img = img.resize((img_size, img_size), Image.LANCZOS)
                arr = np.array(img, dtype=np.float32) / 255.0
                X_list.append(arr)
                y_list.append(CLASS_NAMES.index(class_name))
            except Exception as e:
                print(f"  ⚠ Erro ao carregar {img_path}: {e}")

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    print(f"✓ Dataset carregado: {X.shape[0]} imagens, {len(found_classes)} classes")
    return X, y, found_classes


def load_samples_by_class(n_per_class: int = 5, img_size: int = 64) -> dict:
    """
    Retorna amostras de imagens por classe para visualização.

    Retorna
    -------
    dict : {class_name: list of ndarray (H, W, 3)}
    """
    samples = {}
    rng = np.random.default_rng(0)
    for class_name in CLASS_NAMES:
        class_dir = EUROSAT_RGB_DIR / class_name
        if not class_dir.exists():
            continue
        files = sorted(class_dir.glob("*.jpg"))
        if len(files) == 0:
            continue
        selected = rng.choice(len(files), min(n_per_class, len(files)), replace=False)
        imgs = []
        for idx in selected:
            img = Image.open(files[idx]).convert("RGB")
            if img.size != (img_size, img_size):
                img = img.resize((img_size, img_size), Image.LANCZOS)
            imgs.append(np.array(img, dtype=np.uint8))
        samples[class_name] = imgs
    return samples


def save_processed(X: np.ndarray, y: np.ndarray, tag: str = "rgb") -> None:
    """Salva arrays processados em disco para reutilização."""
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    np.save(DATA_PROCESSED / f"X_{tag}.npy", X)
    np.save(DATA_PROCESSED / f"y_{tag}.npy", y)
    print(f"✓ Dados salvos em {DATA_PROCESSED} (tag={tag})")


def load_processed(tag: str = "features") -> tuple[np.ndarray, np.ndarray] | None:
    """Carrega arrays processados do disco, se existirem."""
    X_path = DATA_PROCESSED / f"X_{tag}.npy"
    y_path = DATA_PROCESSED / f"y_{tag}.npy"
    if X_path.exists() and y_path.exists():
        X = np.load(X_path)
        y = np.load(y_path)
        print(f"✓ Dados carregados de {DATA_PROCESSED} (tag={tag}): X={X.shape}, y={y.shape}")
        return X, y
    return None


# ─── Execução direta ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  EuroSAT Data Loader — Trabalho 10 PDI")
    print("=" * 60)
    download_eurosat_rgb()

    print("\nCarregando imagens...")
    X, y, classes = load_images_rgb(max_per_class=500)

    print(f"\nEstatísticas do dataset:")
    print(f"  Shape X: {X.shape}")
    print(f"  Shape y: {y.shape}")
    print(f"  Classes: {classes}")
    for i, cls in enumerate(CLASS_NAMES):
        count = np.sum(y == i)
        print(f"    {cls:30s}: {count:4d} imagens")

    save_processed(X, y, tag="rgb")
    print("\n✓ data_loader.py concluído com sucesso.")
