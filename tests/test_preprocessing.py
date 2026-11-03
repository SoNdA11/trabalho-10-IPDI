"""Radiometria, máscaras e filtros."""
import numpy as np

import config as C
import preprocessing as P


def test_l1c_offset_harmonization():
    dn = np.array([[1000, 1500, 11000]], dtype=np.uint16)
    assert P.harmonize_l1c(dn).tolist() == [[0, 500, 10000]]
    refl = P.to_reflectance(dn, "sentinel2", offset=C.S2_RADIO_ADD_OFFSET)
    assert np.allclose(refl, [[0.0, 0.05, 1.0]])


def test_cbers_scale_and_mask():
    dn = np.array([[[100, 200, 300, 2000], [C.CBERS_NODATA] * 4]], dtype=np.int16)
    cmask = np.array([[127, 127]], dtype=np.uint8)
    assert np.allclose(P.to_reflectance(dn, "cbers4_mux")[0, 0], [0.01, 0.02, 0.03, 0.2])
    assert P.valid_mask(dn, "cbers4_mux", cmask).tolist() == [[True, False]]
    cmask[0, 0] = 255  # nuvem
    assert P.valid_mask(dn, "cbers4_mux", cmask).tolist() == [[False, False]]


def test_fix_invalid_replaces_nan_and_clips():
    x = np.array([[[np.nan], [0.2], [3.0], [-0.1]]], dtype=np.float32)
    y = P.fix_invalid(x)
    assert np.isfinite(y).all() and y.min() >= 0 and y.max() <= 1.5


def test_majority_filter_removes_isolated_pixel():
    lab = np.zeros((5, 5), dtype=np.uint8)
    lab[2, 2] = 3
    assert P.majority_filter(lab, 5)[2, 2] == 0


def test_stretch_uses_common_limits():
    a = np.random.default_rng(0).uniform(0, 1, (10, 10, 3))
    _, lim = P.stretch(a)
    b, lim2 = P.stretch(a * 0.5, limits=lim)
    assert lim2 is lim and b.max() < 200
