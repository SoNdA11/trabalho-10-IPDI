"""
feature_extraction.py — Etapa 4: extração de características.

Transforma imagens em vetores numéricos que o classificador consegue separar.

Índices espectrais (só são calculados se o sensor tiver as bandas exigidas):

    NDVI  = (NIR − Red)   / (NIR + Red)      vegetação (clorofila absorve o vermelho,
                                              a estrutura da folha reflete o NIR)
    NDWI  = (Green − NIR) / (Green + NIR)    água (McFeeters, 1996)
    MNDWI = (Green − SWIR1)/(Green + SWIR1)  água, separa melhor de área urbana (Xu, 2006)
    NDBI  = (SWIR1 − NIR) / (SWIR1 + NIR)    área construída (Zha et al., 2003)
    BSI   = ((SWIR1+Red) − (NIR+Blue)) / ((SWIR1+Red) + (NIR+Blue))   solo exposto

Sentinel-2 tem todas as bandas → 5 índices. CBERS-4 MUX não tem SWIR → só NDVI e
NDWI. O recorte "só RGB" não tem NIR → nenhum índice (não inventamos aproximações).

Dois tipos de vetor:
* por **imagem** (recorte EuroSAT 64×64): estatísticas por banda, estatísticas e
  histograma dos índices e textura (GLCM + gradiente);
* por **pixel** (cena do INPE): reflectância das bandas, índices e contexto local
  (média e desvio numa janela 5×5).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent))

import config as C  # noqa: E402

EPS = 1e-6

# índice: (bandas exigidas, função)
INDEX_DEFS = {
    "NDVI": (("nir", "red"), lambda b: _nd(b["nir"], b["red"])),
    "NDWI": (("green", "nir"), lambda b: _nd(b["green"], b["nir"])),
    "MNDWI": (("green", "swir1"), lambda b: _nd(b["green"], b["swir1"])),
    "NDBI": (("swir1", "nir"), lambda b: _nd(b["swir1"], b["nir"])),
    "BSI": (("swir1", "red", "nir", "blue"),
            lambda b: _nd(b["swir1"] + b["red"], b["nir"] + b["blue"])),
}


def _nd(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Diferença normalizada (a − b) / (a + b), protegida contra divisão por zero."""
    return (a - b) / (a + b + EPS)


def roles_of(refl: np.ndarray, sensor: str) -> dict[str, np.ndarray]:
    """Mapeia papel físico → banda (ex.: 'nir' → B08) para o sensor."""
    info = C.SENSOR_BANDS[sensor]
    return {role: refl[..., info["names"].index(band)] for role, band in info["roles"].items()}


def available_indices(sensor: str) -> list[str]:
    """Índices que o sensor permite calcular, conferindo as bandas exigidas."""
    roles = C.SENSOR_BANDS[sensor]["roles"]
    return [k for k, (need, _) in INDEX_DEFS.items() if all(r in roles for r in need)]


def spectral_index(refl: np.ndarray, sensor: str, name: str) -> np.ndarray:
    """Calcula um índice; levanta erro se faltar banda (nunca aproxima)."""
    need, fn = INDEX_DEFS[name]
    roles = C.SENSOR_BANDS[sensor]["roles"]
    missing = [r for r in need if r not in roles]
    if missing:
        raise ValueError(f"{name} exige {need}; o sensor {sensor} não tem {missing}")
    return fn(roles_of(refl, sensor)).astype(np.float32)


# ─── Textura ──────────────────────────────────────────────────────────────────

GLCM_LEVELS = 16


def glcm_features(gray: np.ndarray) -> np.ndarray:
    """GLCM (Haralick, 1973) em lote: (N, H, W) → (N, 4).

    Quantiza cada recorte em 16 níveis (percentis 1–99 do próprio recorte),
    conta pares de vizinhos à direita e abaixo (distância 1, 0° e 90°) e
    extrai contraste, homogeneidade, energia e correlação.
    """
    n = gray.shape[0]
    L = GLCM_LEVELS
    lo = np.percentile(gray.reshape(n, -1), 1, axis=1)[:, None, None]
    hi = np.percentile(gray.reshape(n, -1), 99, axis=1)[:, None, None]
    q = np.clip(((gray - lo) / (hi - lo + EPS) * L).astype(np.int64), 0, L - 1)
    base = (np.arange(n) * L * L)[:, None]
    pairs = [
        (q[:, :, :-1].reshape(n, -1), q[:, :, 1:].reshape(n, -1)),   # 0°
        (q[:, :-1, :].reshape(n, -1), q[:, 1:, :].reshape(n, -1)),   # 90°
    ]
    P = np.zeros((n, L, L), dtype=np.float64)
    for a, b in pairs:
        for x, y in ((a, b), (b, a)):  # matriz simétrica
            P += np.bincount((base + x * L + y).ravel(), minlength=n * L * L).reshape(n, L, L)
    P /= P.sum(axis=(1, 2), keepdims=True)
    i, j = np.meshgrid(np.arange(L), np.arange(L), indexing="ij")
    contrast = (P * (i - j) ** 2).sum(axis=(1, 2))
    homogeneity = (P / (1.0 + np.abs(i - j))).sum(axis=(1, 2))
    energy = np.sqrt((P ** 2).sum(axis=(1, 2)))
    mu_i = (P * i).sum(axis=(1, 2))
    mu_j = (P * j).sum(axis=(1, 2))
    sd_i = np.sqrt((P * (i - mu_i[:, None, None]) ** 2).sum(axis=(1, 2)))
    sd_j = np.sqrt((P * (j - mu_j[:, None, None]) ** 2).sum(axis=(1, 2)))
    corr = ((P * (i - mu_i[:, None, None]) * (j - mu_j[:, None, None])).sum(axis=(1, 2))
            / (sd_i * sd_j + EPS))
    return np.stack([contrast, homogeneity, energy, corr], axis=1).astype(np.float32)


def gradient_features(gray: np.ndarray) -> np.ndarray:
    """Magnitude do gradiente de Sobel (bordas) e variância local 3×3: (N, 2)."""
    gx = ndimage.sobel(gray, axis=2)
    gy = ndimage.sobel(gray, axis=1)
    grad = np.hypot(gx, gy).mean(axis=(1, 2))
    m = ndimage.uniform_filter(gray, size=(1, 3, 3))
    m2 = ndimage.uniform_filter(gray ** 2, size=(1, 3, 3))
    lvar = np.clip(m2 - m ** 2, 0, None).mean(axis=(1, 2))
    return np.stack([grad, lvar], axis=1).astype(np.float32)


def texture_band(refl: np.ndarray, sensor: str) -> np.ndarray:
    """Banda usada na textura: NIR (realça a estrutura da vegetação) ou o brilho visível."""
    roles = C.SENSOR_BANDS[sensor]["roles"]
    if "nir" in roles:
        return roles_of(refl, sensor)["nir"]
    r = roles_of(refl, sensor)
    return (r["red"] + r["green"] + r["blue"]) / 3.0


# ─── Vetor por imagem (recorte inteiro) ───────────────────────────────────────

BAND_STATS = ["mean", "std", "p10", "p50", "p90"]
INDEX_STATS = ["mean", "std", "p10", "p90"]
HIST_BINS = np.array([-1.0, 0.0, 0.15, 0.3, 0.45, 0.6, 1.0])  # NDVI (ou brilho)
TEXTURE = ["glcm_contraste", "glcm_homogeneidade", "glcm_energia", "glcm_correlacao",
           "gradiente_sobel", "variancia_local"]


def patch_feature_names(sensor: str) -> list[str]:
    bands = C.SENSOR_BANDS[sensor]["names"]
    names = [f"{b}_{s}" for b in bands for s in BAND_STATS]
    idx = available_indices(sensor)
    names += [f"{k}_{s}" for k in idx for s in INDEX_STATS]
    hist_src = "NDVI" if "NDVI" in idx else "brilho"
    names += [f"hist_{hist_src}_{i}" for i in range(len(HIST_BINS) - 1)]
    names += TEXTURE
    return names


def patch_features(refl: np.ndarray, sensor: str) -> np.ndarray:
    """Vetor de características de um lote de recortes: (N, H, W, B) → (N, D)."""
    if refl.ndim == 3:
        refl = refl[None]
    n = refl.shape[0]
    flat = refl.reshape(n, -1, refl.shape[-1])
    q = np.percentile(flat, [10, 50, 90], axis=1)  # (3, N, B)
    band = np.stack([flat.mean(axis=1), flat.std(axis=1), q[0], q[1], q[2]], axis=2)  # (N,B,5)
    feats = [band.reshape(n, -1)]

    idx = available_indices(sensor)
    for k in idx:
        v = spectral_index(refl, sensor, k).reshape(n, -1)
        p = np.percentile(v, [10, 90], axis=1)
        feats.append(np.stack([v.mean(axis=1), v.std(axis=1), p[0], p[1]], axis=1))

    if "NDVI" in idx:
        h_src, bins = spectral_index(refl, sensor, "NDVI").reshape(n, -1), HIST_BINS
    else:
        h_src = texture_band(refl, sensor).reshape(n, -1)
        bins = np.array([0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 2.0])
    hist = np.stack([np.histogram(row, bins=bins)[0] for row in h_src]).astype(np.float32)
    feats.append(hist / h_src.shape[1])

    gray = texture_band(refl, sensor).astype(np.float32)
    feats += [glcm_features(gray), gradient_features(gray)]
    return np.concatenate(feats, axis=1).astype(np.float32)


# ─── Vetor por pixel (cena) ───────────────────────────────────────────────────

def pixel_feature_names(sensor: str) -> list[str]:
    names = list(C.SENSOR_BANDS[sensor]["names"])
    idx = available_indices(sensor)
    names += idx
    ctx = "NDVI" if "NDVI" in idx else "brilho"
    tex = "NIR" if "nir" in C.SENSOR_BANDS[sensor]["roles"] else "brilho"
    names += [f"{ctx}_media5x5", f"{ctx}_desvio5x5", f"{tex}_media5x5", f"{tex}_desvio5x5"]
    return names


def pixel_features(refl: np.ndarray, sensor: str, size: int = 5) -> np.ndarray:
    """Cubo de características por pixel: (H, W, B) → (H, W, D).

    O contexto 5×5 (média e desvio local) dá ao classificador por pixel uma
    noção de textura: um pixel de telhado cercado de ruas tem desvio alto; um
    pixel de açude, desvio quase zero.
    """
    layers = [refl[..., b] for b in range(refl.shape[-1])]
    idx = available_indices(sensor)
    ind = {k: spectral_index(refl, sensor, k) for k in idx}
    layers += [ind[k] for k in idx]
    ctx = ind.get("NDVI", texture_band(refl, sensor))
    nir = texture_band(refl, sensor)
    for img in (ctx, nir):
        m = ndimage.uniform_filter(img, size=size)
        m2 = ndimage.uniform_filter(img * img, size=size)
        layers += [m, np.sqrt(np.clip(m2 - m * m, 0, None))]
    return np.stack(layers, axis=-1).astype(np.float32)


# ─── Explicação legível de um vetor ───────────────────────────────────────────

def describe(vector: np.ndarray, names: list[str], keys: list[str] | None = None) -> dict:
    """Dicionário nome → valor (para a API e para os slides)."""
    d = {n: float(v) for n, v in zip(names, vector)}
    return {k: d[k] for k in keys if k in d} if keys else d
