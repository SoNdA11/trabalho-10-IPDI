"""
feature_extraction.py — Extração de características espectrais

Características implementadas:
  1. Estatísticas por canal: média, desvio padrão, mínimo, máximo, mediana
  2. NDVI — Normalized Difference Vegetation Index
  3. NDWI — Normalized Difference Water Index
  4. Histograma de intensidades (por canal)
  5. Características de textura (contraste local)

Para o dataset EuroSAT RGB:
  - Canal 0 = Red (≈ B4 do Sentinel-2)
  - Canal 1 = Green (≈ B3 do Sentinel-2)
  - Canal 2 = Blue (≈ B2 do Sentinel-2)

NOTA: NDVI e NDWI exigem o canal NIR (B8 do Sentinel-2), ausente no EuroSAT RGB.
Implementamos uma aproximação baseada nos canais RGB disponíveis e calculamos
índices derivados que capturam informação espectral equivalente.
"""

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ─── Índices Espectrais ────────────────────────────────────────────────────────

def ndvi_approx(img: np.ndarray) -> np.ndarray:
    """
    NDVI aproximado usando canal Red e Green como proxy do NIR.

    Fórmula original: NDVI = (NIR - Red) / (NIR + Red)
    Aproximação RGB:  NDVI_approx = (Green - Red) / (Green + Red + ε)

    Justificativa: sem o canal NIR, o canal Green captura parte da reflexão
    fotossintética e o Red captura absorção pela clorofila. A razão Green/Red
    correlaciona-se positivamente com o NDVI real para vegetação densa.

    Parâmetros
    ----------
    img : ndarray (H, W, 3) com canais RGB normalizados em [0, 1]

    Retorna
    -------
    ndarray (H, W) com valores em [-1, 1]
    """
    red   = img[..., 0].astype(np.float32)
    green = img[..., 1].astype(np.float32)
    num = green - red
    den = green + red + 1e-8
    return num / den


def ndwi_approx(img: np.ndarray) -> np.ndarray:
    """
    NDWI aproximado usando canal Green e Blue como proxy.

    Fórmula original: NDWI = (Green - NIR) / (Green + NIR)
    Aproximação RGB:  NDWI_approx = (Blue - Red) / (Blue + Red + ε)

    Justificativa: a água apresenta alta reflectância no azul e baixa no
    vermelho. Corpos d'água geram NDWI_approx positivo, enquanto vegetação
    e solo exposto tendem a valores negativos.

    Parâmetros
    ----------
    img : ndarray (H, W, 3) com canais RGB normalizados em [0, 1]

    Retorna
    -------
    ndarray (H, W) com valores em [-1, 1]
    """
    red  = img[..., 0].astype(np.float32)
    blue = img[..., 2].astype(np.float32)
    num = blue - red
    den = blue + red + 1e-8
    return num / den


def brightness(img: np.ndarray) -> np.ndarray:
    """
    Brilho médio: média dos canais RGB por pixel.
    Útil para distinguir áreas urbanas (alta reflectância geral)
    de vegetação densa (baixo brilho no vermelho).
    """
    return img.mean(axis=-1)


def redness_index(img: np.ndarray) -> np.ndarray:
    """
    Índice de vermelhidão: (Red - Blue) / (Red + Blue + ε)
    Elevado para solo exposto e áreas secas.
    """
    red  = img[..., 0].astype(np.float32)
    blue = img[..., 2].astype(np.float32)
    return (red - blue) / (red + blue + 1e-8)


# ─── Estatísticas espectrais ──────────────────────────────────────────────────

def spectral_stats(img: np.ndarray) -> np.ndarray:
    """
    Extrai estatísticas de intensidade por canal.

    Características: [mean_R, mean_G, mean_B,
                      std_R, std_G, std_B,
                      min_R, min_G, min_B,
                      max_R, max_G, max_B,
                      median_R, median_G, median_B]
    → 15 valores por imagem

    Parâmetros
    ----------
    img : ndarray (H, W, C)

    Retorna
    -------
    ndarray (5 * C,)
    """
    features = []
    for c in range(img.shape[-1]):
        ch = img[..., c].flatten()
        features.extend([
            ch.mean(),
            ch.std(),
            ch.min(),
            ch.max(),
            np.median(ch),
        ])
    return np.array(features, dtype=np.float32)


def histogram_features(img: np.ndarray, n_bins: int = 8) -> np.ndarray:
    """
    Histograma de intensidade por canal (normalizado).

    Divide o intervalo [0,1] em n_bins compartimentos e conta a frequência
    relativa de pixels em cada faixa. Captura a distribuição espectral
    da imagem de forma compacta.

    Parâmetros
    ----------
    img   : ndarray (H, W, C)
    n_bins: número de compartimentos do histograma

    Retorna
    -------
    ndarray (n_bins * C,) — histograma concatenado e normalizado
    """
    features = []
    for c in range(img.shape[-1]):
        hist, _ = np.histogram(img[..., c].flatten(), bins=n_bins, range=(0, 1))
        hist = hist.astype(np.float32) / (hist.sum() + 1e-8)
        features.extend(hist)
    return np.array(features, dtype=np.float32)


# ─── Textura ──────────────────────────────────────────────────────────────────

def texture_features(img_gray: np.ndarray, window_size: int = 3) -> np.ndarray:
    """
    Características de textura baseadas em variância local.

    Usa a variância de intensidade em janelas deslizantes para capturar
    homogeneidade (florestas têm baixa variância) vs. heterogeneidade
    (áreas urbanas têm alta variância de bordas e telhados).

    Parâmetros
    ----------
    img_gray    : ndarray (H, W) — imagem em escala de cinza
    window_size : tamanho da janela (padrão: 3×3)

    Retorna
    -------
    ndarray (4,) — [mean_variance, std_variance, max_variance, energy]
    """
    from scipy.ndimage import uniform_filter

    pad = window_size // 2
    img_padded = np.pad(img_gray, pad, mode="reflect")
    h, w = img_gray.shape
    variance_map = np.zeros((h, w), dtype=np.float32)

    # Variância em janela deslizante (E[X²] - E[X]²)
    mean_sq = uniform_filter(img_gray**2, size=window_size)
    sq_mean = uniform_filter(img_gray, size=window_size) ** 2
    variance_map = np.clip(mean_sq - sq_mean, 0, None)

    return np.array([
        variance_map.mean(),
        variance_map.std(),
        variance_map.max(),
        # Energia (baixa entropia = textura homogênea)
        1.0 - (variance_map.mean() / (variance_map.max() + 1e-8)),
    ], dtype=np.float32)


# ─── Feature vector completo ──────────────────────────────────────────────────

def extract_features(img: np.ndarray) -> np.ndarray:
    """
    Extrai o vetor de características completo de uma imagem.

    Componentes do vetor:
    ┌─────────────────────────────────────────┬──────────┐
    │ Grupo                                   │ Dim      │
    ├─────────────────────────────────────────┼──────────┤
    │ Estatísticas espectrais (5 × 3 canais)  │ 15       │
    │ Histograma (8 bins × 3 canais)          │ 24       │
    │ NDVI_approx — estatísticas              │ 5        │
    │ NDWI_approx — estatísticas              │ 5        │
    │ Brightness — estatísticas               │ 5        │
    │ Redness Index — estatísticas            │ 5        │
    │ Textura (variância local)               │ 4        │
    ├─────────────────────────────────────────┼──────────┤
    │ Total                                   │ 63       │
    └─────────────────────────────────────────┴──────────┘

    Parâmetros
    ----------
    img : ndarray (H, W, 3) normalizado em [0, 1]

    Retorna
    -------
    ndarray (63,) — vetor de características
    """
    img = img.astype(np.float32)

    # 1. Estatísticas espectrais (15)
    spec = spectral_stats(img)

    # 2. Histograma (24)
    hist = histogram_features(img, n_bins=8)

    # 3. Índices espectrais — cada um como mapa → estatísticas (5 cada)
    ndvi = ndvi_approx(img)
    ndwi = ndwi_approx(img)
    bright = brightness(img)
    red_idx = redness_index(img)

    def _stats5(arr: np.ndarray) -> np.ndarray:
        a = arr.flatten()
        return np.array([a.mean(), a.std(), a.min(), a.max(), np.median(a)], dtype=np.float32)

    ndvi_feat   = _stats5(ndvi)    # 5
    ndwi_feat   = _stats5(ndwi)    # 5
    bright_feat = _stats5(bright)  # 5
    red_feat    = _stats5(red_idx) # 5

    # 4. Textura (4)
    gray = bright  # usa brilho médio como canal de textura
    tex = texture_features(gray)

    feature_vector = np.concatenate([
        spec, hist, ndvi_feat, ndwi_feat, bright_feat, red_feat, tex
    ])
    return feature_vector


def extract_features_batch(X: np.ndarray, verbose: bool = True) -> np.ndarray:
    """
    Extrai características de um dataset completo.

    Parâmetros
    ----------
    X : ndarray (N, H, W, C) normalizado em [0, 1]

    Retorna
    -------
    ndarray (N, D) — matriz de features
    """
    n = len(X)
    features = []
    for i, img in enumerate(X):
        features.append(extract_features(img))
        if verbose and (i + 1) % 500 == 0:
            print(f"  {i+1}/{n} imagens processadas")
    F = np.array(features, dtype=np.float32)
    if verbose:
        print(f"✓ Feature matrix: {F.shape} (N={F.shape[0]}, D={F.shape[1]})")
    return F


# ─── Visualizações ────────────────────────────────────────────────────────────

def plot_spectral_indices(
    img: np.ndarray,
    class_name: str,
    save_path: Path | None = None,
) -> None:
    """
    Visualiza imagem RGB com mapas de NDVI, NDWI e brilho.
    """
    ndvi  = ndvi_approx(img)
    ndwi  = ndwi_approx(img)
    bright = brightness(img)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle(f"Índices Espectrais — {class_name}", fontsize=13, fontweight="bold")

    axes[0].imshow(img)
    axes[0].set_title("RGB Original")
    axes[0].axis("off")

    im1 = axes[1].imshow(ndvi, cmap="RdYlGn", vmin=-1, vmax=1)
    axes[1].set_title("NDVI (aprox.)")
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    im2 = axes[2].imshow(ndwi, cmap="Blues", vmin=-1, vmax=1)
    axes[2].set_title("NDWI (aprox.)")
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    im3 = axes[3].imshow(bright, cmap="gray", vmin=0, vmax=1)
    axes[3].set_title("Brilho médio")
    axes[3].axis("off")
    plt.colorbar(im3, ax=axes[3], fraction=0.046)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✓ Figura salva: {save_path}")
    plt.close()


def plot_feature_importance(
    feature_names: list[str],
    importances: np.ndarray,
    top_n: int = 20,
    save_path: Path | None = None,
) -> None:
    """
    Plota importância das features do Random Forest.
    """
    idx = np.argsort(importances)[::-1][:top_n]
    top_names = [feature_names[i] for i in idx]
    top_vals  = importances[idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(range(top_n), top_vals[::-1], color="#3498DB", alpha=0.85)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_names[::-1], fontsize=9)
    ax.set_xlabel("Importância (Gini)")
    ax.set_title(f"Top {top_n} Features mais Importantes — Random Forest", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✓ Figura salva: {save_path}")
    plt.close()


def build_feature_names() -> list[str]:
    """Retorna lista de nomes das features na mesma ordem de extract_features."""
    names = []
    bands = ["Red", "Green", "Blue"]
    stats = ["mean", "std", "min", "max", "median"]

    # Estatísticas espectrais
    for b in bands:
        for s in stats:
            names.append(f"{b}_{s}")

    # Histograma
    for b in bands:
        for i in range(8):
            names.append(f"hist_{b}_bin{i}")

    # Índices
    for idx_name in ["NDVI", "NDWI", "Brilho", "RI"]:
        for s in stats:
            names.append(f"{idx_name}_{s}")

    # Textura
    names.extend(["tex_var_mean", "tex_var_std", "tex_var_max", "tex_energy"])

    return names


if __name__ == "__main__":
    print("feature_extraction.py — módulo de extração de características")
    names = build_feature_names()
    print(f"  Vetor de features: {len(names)} dimensões")
    for i, n in enumerate(names):
        print(f"    [{i:02d}] {n}")
