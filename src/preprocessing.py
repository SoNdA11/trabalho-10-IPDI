"""
preprocessing.py — Etapa 3: pré-processamento.

Cada operação existe por um motivo concreto destes dados:

* `to_reflectance`      números digitais → reflectância [0, ~1]. O L1C recente
                         soma 1000 aos DN (baseline ≥ 04.00); o EuroSAT, não.
                         Sem harmonizar, o modelo treinado no EuroSAT veria toda
                         cena nova ~0,1 mais clara em todas as bandas.
* `valid_mask`          pixels sem dado (borda da cena), saturados ou com nuvem
                         (máscara CMASK do INPE no CBERS) ficam de fora.
* `fix_invalid`         NaN/inf e valores fora da faixa física → tratados antes
                         de calcular índices (evita divisões inválidas).
* `stretch`             realce por percentis (2–98 %) só para *visualização*:
                         a reflectância bruta é escura demais para a tela.
* `composite`           composições coloridas: cor verdadeira, falsa cor (NIR)
                         e SWIR (destaca solo exposto e área urbana).
* `denoise_median`      filtro de mediana opcional (ruído impulsivo).
* `majority_filter`     pós-classificação: remove pixels isolados ("sal e pimenta")
                         do mapa, trocando cada rótulo pela moda da vizinhança.

O alinhamento espacial das bandas (20 m e 60 m → 10 m) é feito na leitura,
em `data_loader.prepare_s2_scene`, por reamostragem bilinear para uma grade única.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent))

import config as C  # noqa: E402


# ─── Radiometria ──────────────────────────────────────────────────────────────

def to_reflectance(dn: np.ndarray, sensor: str, offset: float | None = None) -> np.ndarray:
    """Converte números digitais em reflectância (float32).

    sentinel2 (EuroSAT, baseline antiga): ρ = DN / 10000
    sentinel2 (L1C baseline ≥ 04.00):     ρ = (DN + offset) / 10000, offset = −1000
    cbers4_mux (L4 SR do INPE):           ρ = DN × 1e-4
    """
    x = dn.astype(np.float32)
    if sensor in ("sentinel2", "rgb"):
        return (x + (offset or 0.0)) / C.S2_QUANTIFICATION
    if sensor == "cbers4_mux":
        return x * C.CBERS_SCALE
    raise ValueError(f"sensor desconhecido: {sensor}")


def valid_mask(dn: np.ndarray, sensor: str, cmask: np.ndarray | None = None) -> np.ndarray:
    """Máscara booleana (H, W) dos pixels utilizáveis."""
    if sensor == "cbers4_mux":
        ok = np.all(dn != C.CBERS_NODATA, axis=-1)
        if cmask is not None:
            ok &= cmask == C.CBERS_CMASK_CLEAR
        return ok
    # Sentinel-2: DN 0 = sem dado; 65535 = saturado
    return np.all((dn > 0) & (dn < 65535), axis=-1)


def fix_invalid(refl: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    """Substitui NaN/inf pela mediana da banda e limita a faixa física [0, 1.5].

    Reflectância TOA pode passar de 1 em nuvens e alvos especulares; acima de
    1,5 é erro. Valores negativos aparecem por ruído/correção atmosférica.
    """
    out = refl.astype(np.float32, copy=True)
    for b in range(out.shape[-1]):
        ch = out[..., b]
        bad = ~np.isfinite(ch)
        if valid is not None:
            bad |= ~valid
        if bad.any():
            good = ch[~bad]
            ch[bad] = np.median(good) if good.size else 0.0
    np.clip(out, 0.0, 1.5, out=out)
    return out


def harmonize_l1c(dn: np.ndarray) -> np.ndarray:
    """DN do L1C atual → escala do EuroSAT (sem o deslocamento de +1000)."""
    x = dn.astype(np.int32) + C.S2_RADIO_ADD_OFFSET
    return np.clip(x, 0, 65535).astype(np.uint16)


def scene_reflectance(scene: dict) -> tuple[np.ndarray, np.ndarray]:
    """Reflectância tratada + máscara de validade para uma cena de `load_scene`."""
    sensor = scene["sensor"]
    valid = valid_mask(scene["dn"], sensor, scene.get("cmask"))
    offset = C.S2_RADIO_ADD_OFFSET if scene["name"] == "s2" else None
    refl = to_reflectance(scene["dn"], sensor, offset)
    return fix_invalid(refl, valid), valid


# ─── Normalização ─────────────────────────────────────────────────────────────

def standardize_fit(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Média e desvio por coluna — calcular SÓ no treino (evita vazamento)."""
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-12] = 1.0
    return mu, sd


def standardize_apply(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (X - mu) / sd


def normalize_minmax(img: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Min-max por banda → [0, 1] (usada na visualização de bandas isoladas)."""
    img = img.astype(np.float32)
    mn = img.reshape(-1, img.shape[-1]).min(axis=0) if img.ndim == 3 else img.min()
    mx = img.reshape(-1, img.shape[-1]).max(axis=0) if img.ndim == 3 else img.max()
    return (img - mn) / (mx - mn + eps)


# ─── Visualização ─────────────────────────────────────────────────────────────

def stretch(img: np.ndarray, p_low: float = 2, p_high: float = 98,
            valid: np.ndarray | None = None, limits=None, gamma: float = 1.0,
            joint: bool = False):
    """Realce linear por percentis, banda a banda → uint8.

    `limits` permite aplicar o mesmo realce a várias imagens (comparação justa).
    `joint=True` usa os mesmos limites nas três bandas: preserva o balanço de
    cor (sem isso, a névoa no azul deixa a cena magenta).
    Retorna (imagem uint8, limites usados).
    """
    x = img.astype(np.float32)
    if x.ndim == 2:
        x = x[..., None]
    if limits is None:
        sample = x[valid] if valid is not None else x.reshape(-1, x.shape[-1])
        lo = np.percentile(sample, p_low, axis=0)
        hi = np.percentile(sample, p_high, axis=0)
        if joint:
            lo, hi = np.full_like(lo, lo.min()), np.full_like(hi, hi.max())
        limits = (lo, hi)
    lo, hi = limits
    y = np.clip((x - lo) / (hi - lo + 1e-8), 0, 1) ** (1.0 / gamma)
    y = (y * 255).astype(np.uint8)
    return (y[..., 0] if img.ndim == 2 else y), limits


COMPOSITES = {
    # nome: (papéis R, G, B)
    "cor_verdadeira": ("red", "green", "blue"),
    "falsa_cor": ("nir", "red", "green"),       # vegetação em vermelho
    "swir": ("swir2", "nir", "red"),            # solo/urbano em tons de rosa/marrom
}


def band_index(sensor: str, role: str) -> int:
    """Posição da banda que cumpre um papel físico (ex.: 'nir') no sensor."""
    info = C.SENSOR_BANDS[sensor]
    if role not in info["roles"]:
        raise KeyError(f"o sensor {sensor} não tem banda '{role}'")
    return info["names"].index(info["roles"][role])


def composite(refl: np.ndarray, sensor: str, kind: str = "cor_verdadeira",
              valid=None, limits=None, gamma: float | None = None):
    """Composição colorida RGB uint8 a partir dos papéis das bandas."""
    roles = COMPOSITES[kind]
    idx = [band_index(sensor, r) for r in roles]
    natural = kind == "cor_verdadeira"
    return stretch(refl[..., idx], p_low=1 if natural else 2, p_high=99 if natural else 98,
                   valid=valid, limits=limits, joint=natural,
                   gamma=gamma if gamma is not None else (1.3 if natural else 1.0))


# ─── Filtros ──────────────────────────────────────────────────────────────────

def denoise_median(img: np.ndarray, size: int = 3) -> np.ndarray:
    """Filtro de mediana banda a banda (remove ruído impulsivo, preserva bordas)."""
    if img.ndim == 2:
        return ndimage.median_filter(img, size=size)
    return np.stack([ndimage.median_filter(img[..., b], size=size)
                     for b in range(img.shape[-1])], axis=-1)


def majority_filter(labels: np.ndarray, n_classes: int, size: int = 3,
                    valid: np.ndarray | None = None) -> np.ndarray:
    """Filtro de moda (maioria) para mapas temáticos.

    Para cada classe, conta quantos vizinhos numa janela size×size têm aquele
    rótulo (filtro de média sobre a máscara binária) e escolhe a mais votada.
    """
    votes = np.stack([ndimage.uniform_filter((labels == k).astype(np.float32), size=size)
                      for k in range(n_classes)], axis=-1)
    if valid is not None:
        votes[~valid] = 0
    out = votes.argmax(axis=-1).astype(labels.dtype)
    if valid is not None:
        out[~valid] = labels[~valid]
    return out
