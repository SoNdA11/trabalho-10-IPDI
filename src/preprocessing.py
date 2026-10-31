"""
preprocessing.py — Pré-processamento de imagens de satélite

Etapas implementadas:
  1. Normalização min-max por banda (reflectância → [0, 1])
  2. Tratamento de valores inválidos (NaN, negativos, saturados)
  3. Composição RGB (bandas espectrais → imagem visível)
  4. Redução de ruído (filtro gaussiano)
  5. Alinhamento espacial (redimensionamento para tamanho padrão)

No EuroSAT RGB, as imagens já são fornecidas como JPEG 8-bit (0–255).
O pré-processamento normaliza para [0,1] e extrai canais individuais.
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ─── Normalização ─────────────────────────────────────────────────────────────

def normalize_minmax(img: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Normalização min-max por imagem completa.

    Mapeia cada canal para [0, 1] usando seus valores mínimo e máximo.
    Finalidade: remover diferenças de iluminação entre cenas, permitindo
    que o classificador compare características espectrais relativas.

    Parâmetros
    ----------
    img : ndarray (H, W, C) ou (H, W)
        Imagem com valores em qualquer escala.
    eps : float
        Valor pequeno para evitar divisão por zero.

    Retorna
    -------
    ndarray com valores em [0, 1].
    """
    img = img.astype(np.float32)
    if img.ndim == 2:
        mn, mx = img.min(), img.max()
        return (img - mn) / (mx - mn + eps)
    # Por canal
    out = np.empty_like(img)
    for c in range(img.shape[-1]):
        ch = img[..., c]
        mn, mx = ch.min(), ch.max()
        out[..., c] = (ch - mn) / (mx - mn + eps)
    return out


def normalize_global(X: np.ndarray) -> np.ndarray:
    """
    Normalização global sobre todo o dataset.

    Calcula média e desvio padrão do dataset completo por canal e aplica
    standardization (z-score). Usado antes do SVM/k-NN quando os valores
    precisam ter mesma escala entre diferentes imagens.

    Parâmetros
    ----------
    X : ndarray (N, H, W, C)
        Dataset completo.

    Retorna
    -------
    X_norm : ndarray normalizado
    mean : ndarray com médias por canal
    std : ndarray com desvios por canal
    """
    X = X.astype(np.float32)
    mean = X.mean(axis=(0, 1, 2), keepdims=True)
    std  = X.std(axis=(0, 1, 2), keepdims=True) + 1e-8
    return (X - mean) / std, mean.squeeze(), std.squeeze()


# ─── Tratamento de valores inválidos ──────────────────────────────────────────

def fix_invalid_values(img: np.ndarray) -> np.ndarray:
    """
    Corrige valores inválidos: NaN, infinitos, negativos e saturação.

    Finalidade: imagens de satélite podem conter:
      - NaN ou inf: erros de sensor ou bordas do swath
      - Valores negativos: ruído de fundo na reflectância de superfície
      - Saturação (>1.0 após normalização): nuvens muito brilhantes

    Estratégia:
      - NaN/inf → substituídos pela mediana do canal
      - Negativos → 0.0
      - > 1.0 após normalização → 1.0
    """
    img = img.copy().astype(np.float32)

    # Substitui NaN e inf pela mediana do canal
    for c in range(img.shape[-1]) if img.ndim == 3 else [None]:
        ch = img[..., c] if c is not None else img
        mask = ~np.isfinite(ch)
        if mask.any():
            ch[mask] = np.nanmedian(ch[~mask]) if (~mask).any() else 0.0
        if c is not None:
            img[..., c] = ch

    # Clipa para [0, 1] após normalização
    np.clip(img, 0.0, 1.0, out=img)
    return img


# ─── Composição colorida ──────────────────────────────────────────────────────

def compose_rgb(img_array: np.ndarray, r=0, g=1, b=2) -> np.ndarray:
    """
    Composição colorida verdadeira (RGB natural) ou falsa cor.

    Para Sentinel-2 com 13 bandas:
      - RGB verdadeiro:  R=B4(red), G=B3(green), B=B2(blue)
      - Falsa cor NIR:   R=B8(NIR), G=B4(red), B=B3(green)

    No dataset EuroSAT RGB, as bandas 0,1,2 já são B4, B3, B2.

    Parâmetros
    ----------
    img_array : ndarray (H, W, C)
        Imagem com C ≥ 3 canais.
    r, g, b : int
        Índices de canal para cada cor.

    Retorna
    -------
    ndarray (H, W, 3) com valores em [0, 1].
    """
    rgb = np.stack([img_array[..., r], img_array[..., g], img_array[..., b]], axis=-1)
    return normalize_minmax(rgb)


# ─── Redução de ruído ─────────────────────────────────────────────────────────

def denoise_gaussian(img: np.ndarray, sigma: float = 0.8) -> np.ndarray:
    """
    Suavização gaussiana para redução de ruído de sensor.

    Finalidade: imagens de satélite podem apresentar ruído de quantização
    e interferência atmosférica. Um filtro gaussiano leve (sigma≈0.8-1.5)
    suaviza ruído de alta frequência sem borrar bordas importantes.

    Não aplicado ao dataset EuroSAT (já preprocessado), mas disponível
    para imagens brutas ou com artefatos visíveis.

    Parâmetros
    ----------
    img : ndarray (H, W) ou (H, W, C)
    sigma : float
        Desvio padrão do kernel gaussiano.

    Retorna
    -------
    ndarray suavizado.
    """
    if img.ndim == 2:
        return gaussian_filter(img, sigma=sigma)
    out = np.empty_like(img)
    for c in range(img.shape[-1]):
        out[..., c] = gaussian_filter(img[..., c], sigma=sigma)
    return out


# ─── Redimensionamento / alinhamento ──────────────────────────────────────────

def resize_to(img: np.ndarray, size: tuple[int, int] = (64, 64)) -> np.ndarray:
    """
    Redimensiona imagem para tamanho padrão.

    Finalidade: alinhamento espacial entre bandas com diferentes resoluções
    (ex: Sentinel-2 B2/B3/B4 = 10m, B5-B7 = 20m, B1 = 60m). Todas as
    bandas são redimensionadas para o tamanho da banda de maior resolução.

    Parâmetros
    ----------
    img : ndarray (H, W) ou (H, W, C)
    size : tuple (height, width)

    Retorna
    -------
    ndarray redimensionado.
    """
    pil_img = Image.fromarray((img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8))
    pil_img = pil_img.resize((size[1], size[0]), Image.LANCZOS)
    return np.array(pil_img).astype(np.float32) / 255.0


# ─── Pipeline completo ────────────────────────────────────────────────────────

def preprocess_dataset(X: np.ndarray) -> np.ndarray:
    """
    Aplica pipeline completo de pré-processamento sobre o dataset.

    Etapas:
      1. Conversão para float32
      2. Correção de valores inválidos
      3. Normalização min-max por imagem

    Parâmetros
    ----------
    X : ndarray (N, H, W, C) com valores em [0, 255]

    Retorna
    -------
    ndarray (N, H, W, C) normalizado em [0, 1]
    """
    X = X.astype(np.float32)
    # Se os valores estiverem em [0, 255], normaliza para [0, 1]
    if X.max() > 1.0:
        X = X / 255.0
    out = np.empty_like(X)
    for i in range(len(X)):
        img = fix_invalid_values(X[i])
        out[i] = img
    return out


# ─── Visualização diagnóstica ─────────────────────────────────────────────────

def plot_preprocessing_demo(
    img_raw: np.ndarray,
    class_name: str,
    save_path: Path | None = None,
) -> None:
    """
    Gera figura mostrando as etapas de pré-processamento.

    Mostra: original → normalizado → com ruído → sem ruído
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle(f"Pré-processamento — {class_name}", fontsize=14, fontweight="bold")

    # 1. Original (uint8)
    raw = (img_raw * 255).astype(np.uint8) if img_raw.max() <= 1.0 else img_raw.astype(np.uint8)
    axes[0].imshow(raw)
    axes[0].set_title("1. Original")
    axes[0].axis("off")

    # 2. Normalizado [0,1]
    normalized = normalize_minmax(img_raw.astype(np.float32) / 255.0)
    axes[1].imshow(normalized)
    axes[1].set_title("2. Normalizado [0,1]")
    axes[1].axis("off")

    # 3. Com ruído simulado
    noisy = normalized + np.random.normal(0, 0.05, normalized.shape)
    noisy = np.clip(noisy, 0, 1)
    axes[2].imshow(noisy)
    axes[2].set_title("3. Com ruído (simulado)")
    axes[2].axis("off")

    # 4. Após filtro gaussiano
    denoised = denoise_gaussian(noisy, sigma=1.0)
    denoised = np.clip(denoised, 0, 1)
    axes[3].imshow(denoised)
    axes[3].set_title("4. Filtrado (gaussiano)")
    axes[3].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✓ Figura salva: {save_path}")
    plt.close()


def plot_band_histograms(
    img: np.ndarray,
    class_name: str,
    band_names: list[str] | None = None,
    save_path: Path | None = None,
) -> None:
    """
    Plota histograma de intensidade por canal de cor.

    Permite verificar a distribuição espectral de cada classe,
    evidenciando diferenças que o classificador aprenderá.
    """
    n_bands = img.shape[-1]
    if band_names is None:
        band_names = [f"Canal {i+1}" for i in range(n_bands)]
    colors = ["#E74C3C", "#2ECC71", "#3498DB"][:n_bands]

    fig, axes = plt.subplots(1, n_bands, figsize=(4 * n_bands, 3.5), sharey=True)
    if n_bands == 1:
        axes = [axes]
    fig.suptitle(f"Histograma de Bandas — {class_name}", fontsize=13, fontweight="bold")

    for i, (ax, color) in enumerate(zip(axes, colors)):
        data = img[..., i].flatten()
        ax.hist(data, bins=50, color=color, alpha=0.85, edgecolor="none")
        ax.set_title(band_names[i])
        ax.set_xlabel("Intensidade")
        ax.set_xlim(0, 1)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("Frequência")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✓ Figura salva: {save_path}")
    plt.close()


if __name__ == "__main__":
    print("preprocessing.py — módulo de pré-processamento")
    print("  normalize_minmax   : normalização min-max por canal")
    print("  fix_invalid_values : trata NaN, inf, negativos")
    print("  denoise_gaussian   : suavização por filtro gaussiano")
    print("  resize_to          : alinhamento espacial entre bandas")
    print("  preprocess_dataset : pipeline completo")
