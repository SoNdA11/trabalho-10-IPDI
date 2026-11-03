"""
exportar_dados.py — gera presentation/data/dados.js a partir dos resultados reais.

Todos os números e gráficos dos slides vêm daqui (results/metrics/*.json e dos
caches de características). Rode depois de treinar:

    .venv/bin/python presentation/scripts/exportar_dados.py
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import config as C  # noqa: E402
import feature_extraction as FE  # noqa: E402


def load(name):
    return json.loads((C.METRICS / f"{name}.json").read_text())


def eurosat_signatures():
    """Reflectância média por banda e classe (13 bandas, ordenadas por comprimento de onda)."""
    d = np.load(C.DATA_PROCESSED / "eurosat_features.npz")
    names = FE.patch_feature_names("sentinel2")
    wl = C.SENSOR_BANDS["sentinel2"]["wavelength_nm"]
    bands = sorted([b for b in C.EUROSAT_BANDS if b not in ("B10",)], key=lambda b: wl[b])
    cols = [names.index(f"{b}_mean") for b in bands]
    out = {}
    for k, c in enumerate(C.EUROSAT_CLASSES):
        out[c] = [round(float(v), 4) for v in d["X_ms"][d["y"] == k][:, cols].mean(axis=0)]
    ndvi = names.index("NDVI_mean")
    ndvi_by_class = {c: [round(float(v), 3) for v in np.percentile(d["X_ms"][d["y"] == k, ndvi], [5, 25, 50, 75, 95])]
                     for k, c in enumerate(C.EUROSAT_CLASSES)}
    return {"bands": bands, "wavelength": [wl[b] for b in bands], "mean": out, "ndvi_box": ndvi_by_class}


def scene_signatures():
    """Reflectância média por classe na cena do INPE (amostra de pixels)."""
    import data_loader as D
    import preprocessing as P
    from train import scene_labels
    s = D.load_scene("s2")
    refl, valid = P.scene_reflectance(s)
    lab = scene_labels(s["ref"], valid)
    wl = C.SENSOR_BANDS["sentinel2"]["wavelength_nm"]
    # só bandas de superfície: B01 (aerossol), B09 (vapor d'água) e B10 (cirros) medem a atmosfera
    bands = sorted([b for b in C.EUROSAT_BANDS if b not in ("B01", "B09", "B10")], key=lambda b: wl[b])
    idx = [C.EUROSAT_BANDS.index(b) for b in bands]
    rng = np.random.default_rng(0)
    out, ind = {}, {}
    for k, c in enumerate(C.LANDCOVER_CLASSES):
        pix = np.flatnonzero(lab.ravel() == k)
        pix = rng.choice(pix, min(20000, pix.size), replace=False)
        r = refl.reshape(-1, refl.shape[-1])[pix]
        out[c] = [round(float(v), 4) for v in r[:, idx].mean(axis=0)]
        ind[c] = {}
        for name in FE.available_indices("sentinel2"):
            v = FE.spectral_index(r[None], "sentinel2", name)[0]
            ind[c][name] = [round(float(x), 3) for x in np.percentile(v, [5, 25, 50, 75, 95])]
    return {"bands": bands, "wavelength": [wl[b] for b in bands], "mean": out, "indices": ind}


def main():
    eu, s2, cb = load("eurosat"), load("s2_pixel"), load("cbers_pixel")
    region = load("s2_region")
    errors = json.loads((C.METRICS / "eurosat_error_examples.json").read_text())
    dados = {
        "eurosat": {k: eu[k] for k in ("classes", "class_counts", "n_total", "n_train", "n_val", "n_test",
                                       "n_groups", "params", "tuning", "feature_dim", "feature_dim_rgb",
                                       "train_seconds", "predict_ms_per_image", "test", "test_macro",
                                       "random_split_test", "comparisons", "feature_importance")},
        "s2": {k: s2[k] for k in ("shape", "resolution_m", "feature_names", "feature_dim", "n_train", "n_val",
                                  "n_test", "class_pixels_total", "params", "tuning", "train_seconds",
                                  "map_seconds", "test", "test_majority3x3", "sampling_study",
                                  "random_split_test", "area_km2_pred", "area_km2_ref", "feature_importance",
                                  "scene")},
        "cbers": {k: cb[k] for k in ("shape", "resolution_m", "feature_names", "feature_dim", "n_test",
                                     "test", "test_majority3x3", "random_split_test", "feature_importance",
                                     "scene", "map_seconds")},
        "region": region,
        "errors": errors,
        "pt": C.EUROSAT_PT,
        "colors": {"eurosat": C.EUROSAT_COLORS, "landcover": C.LANDCOVER_COLORS},
        "macro": C.EUROSAT_TO_MACRO,
        "sig_eurosat": eurosat_signatures(),
        "sig_scene": scene_signatures(),
    }
    out = C.PRESENTATION_DIR / "data" / "dados.js"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("/* Gerado por presentation/scripts/exportar_dados.py — não editar à mão. */\n"
                   "window.DADOS = " + json.dumps(dados, ensure_ascii=False) + ";\n")
    print(f"✓ {out.relative_to(ROOT)} ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
