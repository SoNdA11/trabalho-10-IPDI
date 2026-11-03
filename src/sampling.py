"""
sampling.py — Etapa 5: preparação das amostras e divisão treino/validação/teste.

O ponto central é evitar **vazamento espacial**: pixels (ou recortes) vizinhos
são quase idênticos. Se um vai para o treino e o vizinho para o teste, o teste
mede memorização, não generalização. Por isso a divisão é feita por **blocos**
geográficos, não por amostra.

* EuroSAT: cada recorte tem coordenada. Agrupamos em células de 0,25° (~28 km)
  e dividimos os *grupos* com StratifiedGroupKFold (mantém a proporção das classes).
* Cenas do INPE: a cena é cortada em blocos de 1,28 km; cada bloco inteiro vai
  para treino, validação ou teste. Uma faixa de 2 px na borda dos blocos é
  descartada para que a janela 5×5 das características não "enxergue" o vizinho.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

TRAIN, VAL, TEST = 0, 1, 2


def geo_groups(lonlat: np.ndarray, cell_deg: float = 0.25) -> np.ndarray:
    """Rótulo de grupo = célula de uma grade regular em graus."""
    gx = np.floor(lonlat[:, 0] / cell_deg).astype(np.int64)
    gy = np.floor(lonlat[:, 1] / cell_deg).astype(np.int64)
    _, g = np.unique(gx * 100000 + gy, return_inverse=True)
    return g


def spatial_split(y: np.ndarray, groups: np.ndarray, seed: int = 42, n_folds: int = 7):
    """Divide por grupo: 1 dobra para teste, 1 para validação, o resto treino (~71/14/14)."""
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = [te for _, te in sgkf.split(np.zeros(len(y)), y, groups)]
    test, val = folds[0], folds[1]
    train = np.setdiff1d(np.arange(len(y)), np.concatenate([test, val]))
    return train, val, test


def random_split(y: np.ndarray, seed: int = 42, val_frac: float = 1 / 7, test_frac: float = 1 / 7):
    """Divisão aleatória estratificada nas mesmas proporções (para comparação)."""
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=test_frac, stratify=y, random_state=seed)
    tr, va = train_test_split(tr, test_size=val_frac / (1 - test_frac), stratify=y[tr],
                              random_state=seed)
    return tr, va, te


def block_split_map(shape: tuple[int, int], block: int, seed: int = 42,
                    fractions=(0.6, 0.2, 0.2), buffer: int = 2) -> np.ndarray:
    """Mapa (H, W) com 0 = treino, 1 = validação, 2 = teste, −1 = faixa de exclusão."""
    h, w = shape
    nby, nbx = -(-h // block), -(-w // block)
    rng = np.random.default_rng(seed)
    blocks = rng.choice(3, size=(nby, nbx), p=fractions)
    full = np.kron(blocks, np.ones((block, block), dtype=np.int64))[:h, :w]
    out = np.full(shape, -1, dtype=np.int8)
    for s in (TRAIN, VAL, TEST):
        region = full == s
        if buffer:
            region = ndimage.binary_erosion(region, iterations=buffer, border_value=1)
        out[region] = s
    return out


def sample_pixels_sqrt(labels: np.ndarray, split: np.ndarray, which: int, total: int,
                       n_classes: int, min_per_class: int = 1500, seed: int = 42) -> np.ndarray:
    """Amostra proporcional à RAIZ da frequência de cada classe.

    Meio-termo testado no experimento: amostra balanceada (mesmo número por
    classe) faz o modelo superestimar as classes raras (precisão de "Área
    agrícola" ≈ 0,19); amostra proporcional quase ignora as raras. Com √f,
    a acurácia global e o F1 das classes raras sobem juntos.
    """
    rng = np.random.default_rng(seed + which)
    flat_l, flat_s = labels.ravel(), split.ravel()
    pools = [np.flatnonzero((flat_s == which) & (flat_l == k)) for k in range(n_classes)]
    freq = np.array([p.size for p in pools], dtype=np.float64)
    w = np.sqrt(freq) / np.sqrt(freq).sum()
    out = []
    for k, pool in enumerate(pools):
        n = int(min(pool.size, max(min_per_class, round(total * w[k]))))
        out.append(rng.choice(pool, n, replace=False))
    return np.sort(np.concatenate(out))


def sample_random(labels: np.ndarray, split: np.ndarray, which: int, total: int,
                  seed: int = 42) -> np.ndarray:
    """Amostra aleatória simples (preserva a proporção real das classes)."""
    rng = np.random.default_rng(seed + 10 + which)
    pool = np.flatnonzero((split.ravel() == which) & (labels.ravel() >= 0))
    return np.sort(rng.choice(pool, min(total, pool.size), replace=False))


def sample_pixels(labels: np.ndarray, split: np.ndarray, which: int, per_class: int | None,
                  n_classes: int, seed: int = 42) -> np.ndarray:
    """Índices lineares de pixels do conjunto `which`, até `per_class` por classe.

    `labels` usa −1 para pixel sem rótulo/inválido. Com per_class=None devolve todos.
    """
    rng = np.random.default_rng(seed + which)
    flat_l, flat_s = labels.ravel(), split.ravel()
    out = []
    for k in range(n_classes):
        idx = np.flatnonzero((flat_s == which) & (flat_l == k))
        if per_class is not None and idx.size > per_class:
            idx = rng.choice(idx, per_class, replace=False)
        out.append(idx)
    return np.sort(np.concatenate(out))
