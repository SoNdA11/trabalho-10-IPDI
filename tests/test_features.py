"""Índices espectrais, vetores de características e checagem de bandas."""
import numpy as np
import pytest

import config as C
import feature_extraction as FE


def synthetic(sensor, value_by_role, shape=(8, 8)):
    names = C.SENSOR_BANDS[sensor]["names"]
    roles = C.SENSOR_BANDS[sensor]["roles"]
    img = np.full(shape + (len(names),), 0.1, dtype=np.float32)
    for role, v in value_by_role.items():
        img[..., names.index(roles[role])] = v
    return img


def test_ndvi_formula_vegetation():
    img = synthetic("sentinel2", {"nir": 0.5, "red": 0.1})
    assert np.allclose(FE.spectral_index(img, "sentinel2", "NDVI"), (0.5 - 0.1) / 0.6, atol=1e-4)


def test_water_indices_positive_for_water():
    img = synthetic("sentinel2", {"green": 0.08, "nir": 0.02, "swir1": 0.01})
    assert (FE.spectral_index(img, "sentinel2", "NDWI") > 0).all()
    assert (FE.spectral_index(img, "sentinel2", "MNDWI") > 0).all()


def test_ndbi_positive_for_built_up():
    img = synthetic("sentinel2", {"swir1": 0.35, "nir": 0.25})
    assert (FE.spectral_index(img, "sentinel2", "NDBI") > 0).all()


def test_indices_require_bands():
    assert FE.available_indices("sentinel2") == ["NDVI", "NDWI", "MNDWI", "NDBI", "BSI"]
    assert FE.available_indices("cbers4_mux") == ["NDVI", "NDWI"]   # sem SWIR
    assert FE.available_indices("rgb") == []                        # sem NIR: nenhum índice
    with pytest.raises(ValueError):
        FE.spectral_index(np.zeros((4, 4, 3)), "rgb", "NDVI")
    with pytest.raises(ValueError):
        FE.spectral_index(np.zeros((4, 4, 4)), "cbers4_mux", "NDBI")


@pytest.mark.parametrize("sensor", ["sentinel2", "rgb", "cbers4_mux"])
def test_patch_features_shape_and_names(sensor):
    nb = len(C.SENSOR_BANDS[sensor]["names"])
    X = np.random.default_rng(0).uniform(0, 0.5, (3, 64, 64, nb)).astype(np.float32)
    F = FE.patch_features(X, sensor)
    assert F.shape == (3, len(FE.patch_feature_names(sensor)))
    assert np.isfinite(F).all()


@pytest.mark.parametrize("sensor", ["sentinel2", "cbers4_mux"])
def test_pixel_features_shape(sensor):
    nb = len(C.SENSOR_BANDS[sensor]["names"])
    img = np.random.default_rng(1).uniform(0, 0.5, (20, 30, nb)).astype(np.float32)
    F = FE.pixel_features(img, sensor)
    assert F.shape == (20, 30, len(FE.pixel_feature_names(sensor)))
    assert np.isfinite(F).all()


def test_glcm_homogeneous_vs_noisy():
    flat = np.ones((1, 32, 32), dtype=np.float32) + np.linspace(0, 1e-3, 32)[None, None]
    noisy = np.random.default_rng(2).uniform(0, 1, (1, 32, 32)).astype(np.float32)
    g_flat, g_noisy = FE.glcm_features(flat)[0], FE.glcm_features(noisy)[0]
    assert g_noisy[0] > g_flat[0]  # contraste
    assert g_flat[1] > g_noisy[1]  # homogeneidade
