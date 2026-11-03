"""
train.py — Etapa 6: treinamento dos classificadores.

Três experimentos, um para cada nível de classificação:

    eurosat   classificação da IMAGEM INTEIRA: cada recorte 64×64 do EuroSAT MS
              recebe uma de 10 classes. Random Forest + comparação com SVM e k-NN,
              divisão aleatória × espacial e ablação "só RGB".
    s2        classificação POR PIXEL da cena Sentinel-2 L1C do INPE (Mossoró/RN),
              5 classes, rótulos do ESA WorldCover 2021, divisão por blocos.
    cbers     o mesmo para a cena CBERS-4 MUX do INPE (4 bandas, 20 m).

A classificação POR REGIÃO (janela deslizante com o modelo do EuroSAT sobre a cena
do INPE) fica em predict.py, porque usa o modelo `eurosat` já treinado.

Uso:
    python src/train.py            # todos
    python src/train.py eurosat    # um só
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).parent))

import config as C  # noqa: E402
import data_loader as D  # noqa: E402
import feature_extraction as FE  # noqa: E402
import preprocessing as P  # noqa: E402
import sampling as S  # noqa: E402
from evaluation import compute_metrics, map_to_macro  # noqa: E402

EUROSAT_CACHE = C.DATA_PROCESSED / "eurosat_features.npz"
DEMO_SAMPLES = C.DATA_PROCESSED / "eurosat_demo_samples.npz"

RF_GRID = [
    {"max_features": "sqrt", "min_samples_leaf": 1},
    {"max_features": "sqrt", "min_samples_leaf": 3},
    {"max_features": 0.3, "min_samples_leaf": 1},
    {"max_features": 0.3, "min_samples_leaf": 3},
]


def rf(n_estimators: int = 300, class_weight="balanced", **kw) -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1, random_state=C.SEED,
                                  class_weight=class_weight, **kw)


def tune_rf(Xtr, ytr, Xva, yva, n_estimators=300, class_weight="balanced") -> tuple[dict, list[dict]]:
    """Escolhe hiperparâmetros pelo F1-macro na VALIDAÇÃO (o teste não participa)."""
    log = []
    for params in RF_GRID:
        t0 = time.time()
        m = rf(n_estimators, class_weight=class_weight, **params).fit(Xtr, ytr)
        f = f1_score(yva, m.predict(Xva), average="macro")
        log.append({**params, "val_macro_f1": float(f), "seconds": round(time.time() - t0, 1)})
        print(f"    {params} → F1-macro val = {f:.4f}")
    best = max(log, key=lambda d: d["val_macro_f1"])
    return {k: best[k] for k in ("max_features", "min_samples_leaf")}, log


def save_bundle(name: str, model, **meta) -> Path:
    C.MODELS.mkdir(parents=True, exist_ok=True)
    path = C.MODELS / f"{name}.joblib"
    joblib.dump({"model": model, **meta}, path, compress=3)
    print(f"  ✓ modelo: {path.relative_to(C.ROOT)} ({path.stat().st_size / 1e6:.1f} MB)")
    return path


def save_json(name: str, data: dict) -> None:
    C.METRICS.mkdir(parents=True, exist_ok=True)
    (C.METRICS / f"{name}.json").write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"  ✓ métricas: results/metrics/{name}.json")


# ═══ Experimento 1 — EuroSAT (imagem inteira) ═════════════════════════════════

def build_eurosat_features() -> dict:
    """Lê os 27.000 recortes, converte para reflectância e extrai as características."""
    if EUROSAT_CACHE.exists():
        return dict(np.load(EUROSAT_CACHE, allow_pickle=False))
    D.ensure_eurosat_ms()
    files = D.eurosat_files()
    print(f"  extraindo características de {len(files)} recortes …")
    rgb_idx = [C.EUROSAT_BANDS.index(b) for b in C.SENSOR_BANDS["rgb"]["names"]]
    X_ms, X_rgb, y, lonlat, names = [], [], [], [], []
    batch, labels, coords = [], [], []
    t0 = time.time()

    def flush():
        refl = P.fix_invalid(P.to_reflectance(np.stack(batch), "sentinel2"))
        X_ms.append(FE.patch_features(refl, "sentinel2"))
        X_rgb.append(FE.patch_features(refl[..., rgb_idx], "rgb"))
        batch.clear()

    for i, (path, label) in enumerate(files):
        arr, ll = D.read_eurosat_tif(path)
        batch.append(arr)
        y.append(label)
        lonlat.append(ll)
        names.append(f"{path.parent.name}/{path.name}")
        if len(batch) == 1000:
            flush()
            print(f"    {i + 1}/{len(files)}  ({time.time() - t0:.0f} s)")
    if batch:
        flush()
    data = {"X_ms": np.concatenate(X_ms), "X_rgb": np.concatenate(X_rgb),
            "y": np.array(y, dtype=np.int64), "lonlat": np.array(lonlat, dtype=np.float64),
            "files": np.array(names)}
    np.savez_compressed(EUROSAT_CACHE, **data)
    print(f"  ✓ cache: {EUROSAT_CACHE.relative_to(C.ROOT)}  X_ms={data['X_ms'].shape}")
    return data


def save_demo_samples(data: dict, test_idx: np.ndarray, per_class: int = 24) -> None:
    """Guarda recortes do TESTE (nunca vistos no treino) para a interface web."""
    rng = np.random.default_rng(C.SEED)
    chosen = []
    for k in range(len(C.EUROSAT_CLASSES)):
        idx = test_idx[data["y"][test_idx] == k]
        chosen += sorted(rng.choice(idx, min(per_class, len(idx)), replace=False).tolist())
    chips = np.stack([D.read_eurosat_tif(C.EUROSAT_MS_DIR / data["files"][i])[0] for i in chosen])
    np.savez_compressed(DEMO_SAMPLES, chips=chips, index=np.array(chosen),
                        y=data["y"][chosen], files=data["files"][chosen])
    print(f"  ✓ amostras de demonstração (teste): {len(chosen)}")


def calibrate_rgb_jpg(n_per_class: int = 20) -> dict | None:
    """Ajusta reflectância ≈ a·JPG + b com os pares EuroSAT RGB × MS (mesmos recortes).

    Necessário para a interface aceitar JPG/PNG comuns no modelo "só RGB".
    """
    rgb_dir = C.DATA_RAW / "EuroSAT_RGB"
    if not rgb_dir.exists():
        return None
    from PIL import Image
    rng = np.random.default_rng(C.SEED)
    xs, ys = [[], [], []], [[], [], []]
    for cls in C.EUROSAT_CLASSES:
        for i in rng.choice(np.arange(1, 2000), n_per_class, replace=False):
            ms, jpg = C.EUROSAT_MS_DIR / cls / f"{cls}_{i}.tif", rgb_dir / cls / f"{cls}_{i}.jpg"
            if not (ms.exists() and jpg.exists()):
                continue
            a = P.to_reflectance(D.read_eurosat_tif(ms)[0][..., [3, 2, 1]], "sentinel2")
            j = np.asarray(Image.open(jpg).convert("RGB"), dtype=np.float32)
            for b in range(3):
                keep = j[..., b] < 250  # pixels saturados no JPG ficam de fora
                xs[b].append(j[..., b][keep])
                ys[b].append(a[..., b][keep])
    cal = {"slope": [], "intercept": [], "r": []}
    for b in range(3):
        x, y = np.concatenate(xs[b]), np.concatenate(ys[b])
        a, c = np.polyfit(x, y, 1)
        cal["slope"].append(float(a))
        cal["intercept"].append(float(c))
        cal["r"].append(float(np.corrcoef(x, y)[0, 1]))
    (C.DATA_PROCESSED / "rgb_jpg_calibration.json").write_text(json.dumps(cal, indent=2))
    print(f"  ✓ calibração JPG→reflectância: r = {[round(v, 4) for v in cal['r']]}")
    return cal


def train_eurosat() -> dict:
    print("\n" + "═" * 64 + "\n  Experimento 1 — EuroSAT MS: classificação da imagem inteira\n" + "═" * 64)
    data = build_eurosat_features()
    X, Xrgb, y = data["X_ms"], data["X_rgb"], data["y"]
    names_ms, names_rgb = FE.patch_feature_names("sentinel2"), FE.patch_feature_names("rgb")
    classes = C.EUROSAT_CLASSES

    groups = S.geo_groups(data["lonlat"])
    tr, va, te = S.spatial_split(y, groups, seed=C.SEED)
    print(f"  divisão espacial: treino {len(tr)} · validação {len(va)} · teste {len(te)} "
          f"({len(np.unique(groups))} células de 0,25°)")
    assert not (set(groups[tr]) & set(groups[te])), "grupo repetido entre treino e teste"

    print("  [1] ajuste de hiperparâmetros do Random Forest (validação)")
    best, grid_log = tune_rf(X[tr], y[tr], X[va], y[va])
    print(f"      melhor: {best}")

    trva = np.concatenate([tr, va])
    t0 = time.time()
    model = rf(**best).fit(X[trva], y[trva])
    t_train = time.time() - t0
    t0 = time.time()
    proba = model.predict_proba(X[te])
    t_pred = time.time() - t0
    pred = proba.argmax(axis=1)
    m = compute_metrics(y[te], pred, classes)
    print(f"  [2] RF (13 bandas) teste: acurácia {m['accuracy']:.4f}  F1-macro {m['macro_f1']:.4f}")

    # Solo exposto não existe no EuroSAT: remove a linha/coluna vazia
    macro_classes = [c for c in C.LANDCOVER_CLASSES if c != "Solo exposto"]
    lut = {C.LANDCOVER_CLASSES.index(c): i for i, c in enumerate(macro_classes)}
    yt_m = np.vectorize(lut.get)(map_to_macro(y[te]))
    yp_m = np.vectorize(lut.get)(map_to_macro(pred))
    macro = compute_metrics(yt_m, yp_m, macro_classes)
    print(f"      agrupado em 4 macro-classes: acurácia {macro['accuracy']:.4f}")

    print("  [3] comparações (mesmo teste espacial)")
    comparisons = {}

    def run(label, est, Xa, fit_idx):
        t0 = time.time()
        est.fit(Xa[fit_idx], y[fit_idx])
        tt = time.time() - t0
        t1 = time.time()
        pred_c = est.predict(Xa[te])
        tp = time.time() - t1
        mm = compute_metrics(y[te], pred_c, classes)
        comparisons[label] = {"accuracy": mm["accuracy"], "macro_f1": mm["macro_f1"],
                              "kappa": mm["kappa"], "train_seconds": round(tt, 1),
                              "predict_ms_per_1000": round(tp / len(te) * 1e6, 1),
                              "per_class_f1": {c: v["f1"] for c, v in mm["per_class"].items()}}
        print(f"      {label:34s} acc {mm['accuracy']:.4f}  F1 {mm['macro_f1']:.4f}  ({tt:.0f} s)")
        return est

    comparisons["Random Forest — 13 bandas"] = {
        "accuracy": m["accuracy"], "macro_f1": m["macro_f1"], "kappa": m["kappa"],
        "train_seconds": round(t_train, 1), "predict_ms_per_1000": round(t_pred / len(te) * 1e6, 1),
        "per_class_f1": {c: v["f1"] for c, v in m["per_class"].items()}}
    run("SVM (RBF) — 13 bandas",
        make_pipeline(StandardScaler(), SVC(C=10, gamma="scale", random_state=C.SEED)), X, trva)
    run("k-NN (k=7) — 13 bandas",
        make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7, n_jobs=-1)), X, trva)
    rgb_model = run("Random Forest — só RGB", rf(**best), Xrgb, trva)

    # vazamento: mesmo modelo, divisão aleatória
    rtr, rva, rte = S.random_split(y, seed=C.SEED)
    leak = rf(**best).fit(X[np.concatenate([rtr, rva])], y[np.concatenate([rtr, rva])])
    m_rand = compute_metrics(y[rte], leak.predict(X[rte]), classes)
    print(f"  [4] divisão ALEATÓRIA: acurácia {m_rand['accuracy']:.4f} "
          f"(espacial: {m['accuracy']:.4f}) → diferença = otimismo por vazamento")

    imp = model.feature_importances_
    order = np.argsort(imp)[::-1]
    save_bundle("rf_eurosat_ms", model, feature_names=names_ms, classes=classes, sensor="sentinel2",
                params=best, n_estimators=300)
    save_bundle("rf_eurosat_rgb", rgb_model, feature_names=names_rgb, classes=classes, sensor="rgb",
                params=best, n_estimators=300)
    np.savez_compressed(C.METRICS / "eurosat_test_predictions.npz", index=te, y_true=y[te],
                        y_pred=pred, proba=proba.astype(np.float32))
    save_demo_samples(data, te)
    cal = calibrate_rgb_jpg()

    counts = np.bincount(y, minlength=len(classes))
    result = {
        "experiment": "eurosat", "level": "imagem inteira",
        "dataset": "EuroSAT MS (Sentinel-2 L1C, 13 bandas, 64×64 px)",
        "classes": classes, "class_counts": counts.tolist(),
        "n_total": int(len(y)), "n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(te)),
        "split": "espacial — StratifiedGroupKFold por células de 0,25°",
        "n_groups": int(len(np.unique(groups))),
        "model": "RandomForestClassifier", "n_estimators": 300, "params": best,
        "tuning": grid_log, "feature_dim": int(X.shape[1]), "feature_dim_rgb": int(Xrgb.shape[1]),
        "train_seconds": round(t_train, 2), "predict_ms_per_image": round(t_pred / len(te) * 1000, 3),
        "test": m, "test_macro": macro, "random_split_test": {
            "accuracy": m_rand["accuracy"], "macro_f1": m_rand["macro_f1"], "kappa": m_rand["kappa"]},
        "comparisons": comparisons, "rgb_jpg_calibration": cal,
        "feature_importance": [{"name": names_ms[i], "importance": float(imp[i])} for i in order[:25]],
    }
    save_json("eurosat", result)
    return result


# ═══ Experimentos 2 e 3 — cenas do INPE (por pixel) ═══════════════════════════

def scene_labels(ref: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """WorldCover → índice da classe do trabalho (−1 = sem classe ou inválido)."""
    lut = np.full(256, -1, dtype=np.int64)
    for code, cls in C.WORLDCOVER_TO_CLASS.items():
        lut[code] = C.LANDCOVER_CLASSES.index(cls)
    lab = lut[ref]
    lab[~valid] = -1
    return lab


def prepare_scene_features(name: str) -> dict:
    scene = D.load_scene(name)
    refl, valid = P.scene_reflectance(scene)
    feats = FE.pixel_features(refl, scene["sensor"])
    labels = scene_labels(scene["ref"], valid)
    return dict(scene=scene, refl=refl, valid=valid, feats=feats, labels=labels)


def train_scene(name: str) -> dict:
    title = {"s2": "Sentinel-2 L1C (INPE)", "cbers": "CBERS-4 MUX (INPE)"}[name]
    print("\n" + "═" * 64 + f"\n  Experimento por pixel — {title} — {C.AOI_NAME}\n" + "═" * 64)
    d = prepare_scene_features(name)
    scene, feats, labels, valid = d["scene"], d["feats"], d["labels"], d["valid"]
    sensor, res = scene["sensor"], scene["res"]
    classes = C.LANDCOVER_CLASSES
    names = FE.pixel_feature_names(sensor)
    h, w, nf = feats.shape

    block = int(round(1280 / res))  # blocos de 1,28 km
    split = S.block_split_map((h, w), block, seed=C.SEED)
    np.save(C.DATA_PROCESSED / f"{name}_split.npy", split)
    Xf = feats.reshape(-1, nf)
    # treino: proporcional à raiz da frequência; validação: amostra aleatória
    # (mesma distribuição do teste); teste: TODOS os pixels dos blocos de teste
    tr = S.sample_pixels_sqrt(labels, split, S.TRAIN, 40000, len(classes))
    va = S.sample_random(labels, split, S.VAL, 20000)
    te = S.sample_pixels(labels, split, S.TEST, None, len(classes))
    yl = labels.ravel()
    print(f"  {h}×{w} px de {res:.0f} m · {nf} características · blocos de {block} px")
    print(f"  amostras: treino {len(tr)} · validação {len(va)} · teste {len(te)} (todos os pixels)")

    print("  [1] ajuste de hiperparâmetros (validação)")
    best, grid_log = tune_rf(Xf[tr], yl[tr], Xf[va], yl[va], n_estimators=200, class_weight=None)
    t0 = time.time()
    model = rf(200, class_weight=None, **best).fit(Xf[tr], yl[tr])
    t_train = time.time() - t0

    t0 = time.time()
    pred_test = model.predict(Xf[te])
    t_pred = time.time() - t0
    m = compute_metrics(yl[te], pred_test, classes)
    print(f"  [2] teste: acurácia {m['accuracy']:.4f}  kappa {m['kappa']:.4f}  F1-macro {m['macro_f1']:.4f}")

    sampling_study = _sampling_study(Xf, yl, labels, split, te, best) if name == "s2" else None
    leakage = _leakage_study(Xf, yl, labels, best, m)

    # mapa completo + pós-processamento por moda
    print("  [3] classificando a cena inteira …")
    t0 = time.time()
    flat_valid = valid.ravel()
    full = np.zeros(h * w, dtype=np.uint8)
    full[flat_valid] = _predict_chunks(model, Xf[flat_valid])
    full = full.reshape(h, w)
    t_map = time.time() - t0
    smooth = P.majority_filter(full, len(classes), size=3, valid=valid)
    m_smooth = compute_metrics(yl[te], smooth.ravel()[te], classes)
    print(f"      {valid.sum()} pixels em {t_map:.0f} s · com filtro de moda 3×3: "
          f"acurácia {m_smooth['accuracy']:.4f}  F1-macro {m_smooth['macro_f1']:.4f}")

    # áreas (km²) por classe: mapa do modelo × referência
    px_km2 = (res * res) / 1e6
    area_pred = {c: float(((smooth == i) & valid).sum() * px_km2) for i, c in enumerate(classes)}
    area_ref = {c: float((labels == i).sum() * px_km2) for i, c in enumerate(classes)}

    C.PREDICTIONS.mkdir(parents=True, exist_ok=True)
    np.save(C.PREDICTIONS / f"{name}_map.npy", smooth)
    np.save(C.PREDICTIONS / f"{name}_map_raw.npy", full)
    _write_map_tif(C.PREDICTIONS / f"{name}_map.tif", smooth, valid, scene)

    imp = model.feature_importances_
    order = np.argsort(imp)[::-1]
    save_bundle(f"rf_{name}_pixel", model, feature_names=names, classes=classes, sensor=sensor,
                params=best, n_estimators=200)
    info = D.scene_info()[name]
    result = {
        "experiment": name, "level": "pixel", "scene": info, "aoi": C.AOI_NAME,
        "aoi_bbox": C.AOI_BBOX, "shape": [h, w], "resolution_m": res,
        "reference": "ESA WorldCover 2021 v200 (reclassificado em 5 classes)",
        "classes": classes, "feature_names": names, "feature_dim": nf,
        "split": f"blocos de {block * res / 1000:.2f} km (60/20/20) com faixa de exclusão de 2 px",
        "sampling": "treino ∝ √(frequência da classe), mínimo 1500/classe; validação aleatória; "
                    "teste = todos os pixels dos blocos de teste",
        "n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(te)),
        "class_pixels_total": {c: int((labels == i).sum()) for i, c in enumerate(classes)},
        "model": "RandomForestClassifier", "n_estimators": 200, "params": best, "tuning": grid_log,
        "train_seconds": round(t_train, 2), "map_seconds": round(t_map, 1),
        "predict_us_per_pixel": round(t_pred / len(te) * 1e6, 2),
        "test": m, "test_majority3x3": m_smooth, "sampling_study": sampling_study,
        "random_split_test": leakage,
        "area_km2_pred": area_pred, "area_km2_ref": area_ref,
        "feature_importance": [{"name": names[i], "importance": float(imp[i])} for i in order],
    }
    save_json(f"{name}_pixel", result)
    return result


def _sampling_study(Xf, yl, labels, split, te, best) -> dict:
    """Mesmo modelo, três formas de amostrar o treino — avaliado no mesmo teste."""
    print("  [extra] estudo da amostragem do treino")
    n = len(C.LANDCOVER_CLASSES)
    variants = {
        "balanceada (8000/classe)": S.sample_pixels(labels, split, S.TRAIN, 8000, n),
        "proporcional (aleatória)": S.sample_random(labels, split, S.TRAIN, 40000),
        "raiz da frequência (usada)": S.sample_pixels_sqrt(labels, split, S.TRAIN, 40000, n),
    }
    out = {}
    for label, idx in variants.items():
        m = rf(200, class_weight=None, **best).fit(Xf[idx], yl[idx])
        mm = compute_metrics(yl[te], m.predict(Xf[te]), C.LANDCOVER_CLASSES)
        out[label] = {"n_train": int(len(idx)), "accuracy": mm["accuracy"], "kappa": mm["kappa"],
                      "macro_f1": mm["macro_f1"],
                      "f1": {c: v["f1"] for c, v in mm["per_class"].items()}}
        print(f"      {label:28s} acc {mm['accuracy']:.4f}  F1-macro {mm['macro_f1']:.4f}")
    return out


def _leakage_study(Xf, yl, labels, best, m_block) -> dict:
    """Mesmo modelo com divisão ALEATÓRIA de pixels: treino e teste ficam vizinhos."""
    rng = np.random.default_rng(C.SEED + 99)
    pool = np.flatnonzero(labels.ravel() >= 0)
    test = rng.choice(pool, 200_000, replace=False)
    rest = np.zeros(labels.shape, dtype=np.int8).ravel()
    rest[test] = -1
    tr = S.sample_pixels_sqrt(labels, rest.reshape(labels.shape), 0, 40000, len(C.LANDCOVER_CLASSES))
    model = rf(200, class_weight=None, **best).fit(Xf[tr], yl[tr])
    mm = compute_metrics(yl[test], model.predict(Xf[test]), C.LANDCOVER_CLASSES)
    print(f"  [extra] divisão ALEATÓRIA de pixels: acurácia {mm['accuracy']:.4f} · F1-macro "
          f"{mm['macro_f1']:.4f}  (blocos: {m_block['accuracy']:.4f} · {m_block['macro_f1']:.4f})")
    return {"accuracy": mm["accuracy"], "macro_f1": mm["macro_f1"], "kappa": mm["kappa"],
            "n_test": int(len(test))}


def _predict_chunks(model, X: np.ndarray, chunk: int = 400_000) -> np.ndarray:
    return np.concatenate([model.predict(X[i:i + chunk]) for i in range(0, len(X), chunk)])


def _write_map_tif(path: Path, labels: np.ndarray, valid: np.ndarray, scene: dict) -> None:
    import rasterio
    from matplotlib.colors import to_rgb
    out = labels.astype(np.uint8) + 1
    out[~valid] = 0
    with rasterio.open(path, "w", driver="GTiff", height=out.shape[0], width=out.shape[1], count=1,
                       dtype="uint8", crs=scene["crs"], transform=scene["transform"], nodata=0,
                       compress="deflate") as dst:
        dst.write(out, 1)
        dst.write_colormap(1, {i + 1: tuple(int(v * 255) for v in to_rgb(C.LANDCOVER_COLORS[c])) + (255,)
                               for i, c in enumerate(C.LANDCOVER_CLASSES)})
        dst.update_tags(classes=json.dumps({i + 1: c for i, c in enumerate(C.LANDCOVER_CLASSES)},
                                           ensure_ascii=False))


if __name__ == "__main__":
    which = sys.argv[1:] or ["eurosat", "s2", "cbers"]
    for w in which:
        if w == "eurosat":
            train_eurosat()
        elif w in ("s2", "cbers"):
            train_scene(w)
        else:
            raise SystemExit(f"experimento desconhecido: {w}")
    print("\n✓ treinamento concluído")
