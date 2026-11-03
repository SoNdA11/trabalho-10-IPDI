"""A divisão espacial não pode misturar grupos/blocos entre treino e teste."""
import numpy as np

import sampling as S


def test_spatial_split_groups_disjoint():
    rng = np.random.default_rng(0)
    lonlat = np.column_stack([rng.uniform(-10, 30, 3000), rng.uniform(35, 60, 3000)])
    y = rng.integers(0, 5, 3000)
    g = S.geo_groups(lonlat, 2.0)
    tr, va, te = S.spatial_split(y, g)
    assert not set(g[tr]) & set(g[te])
    assert not set(g[va]) & set(g[te])
    assert len(tr) + len(va) + len(te) == 3000


def test_block_split_has_buffer_and_disjoint_sets():
    m = S.block_split_map((200, 300), 32, buffer=2)
    assert set(np.unique(m)) <= {-1, 0, 1, 2}
    # nenhum pixel de treino encosta (vizinhança 4) num pixel de teste
    tr, te = m == 0, m == 2
    shifted = np.zeros_like(te)
    shifted[1:] |= te[:-1]; shifted[:-1] |= te[1:]; shifted[:, 1:] |= te[:, :-1]; shifted[:, :-1] |= te[:, 1:]
    assert not (tr & shifted).any()


def test_sqrt_sampling_between_balanced_and_proportional():
    labels = np.repeat(np.arange(3), [9000, 900, 100]).reshape(100, 100)
    split = np.zeros_like(labels)
    idx = S.sample_pixels_sqrt(labels, split, 0, 3000, 3, min_per_class=50)
    counts = np.bincount(labels.ravel()[idx], minlength=3)
    assert counts[0] > counts[1] > counts[2] >= 50
    assert counts[0] / counts[2] < 9000 / 100
