"""
predict.py — Etapa 8: inferência e geração dos mapas de classificação.

Três granularidades, todas usando os modelos realmente treinados:

* imagem inteira  `classify_chip`: um recorte 64×64 → UMA classe (EuroSAT).
* por pixel       `classify_window` / `classify_scene`: cada pixel da cena do
                  INPE → uma classe; o resultado é um mapa temático.
* por região      `region_map`: janela deslizante de 64×64 px sobre a cena
                  Sentinel-2; cada janela é classificada pelo modelo do EuroSAT
                  (treinado na Europa) → mapa em blocos.

Uso:
    python src/predict.py      # gera o mapa por região e o compara à referência
"""

from __future__ import annotations

import base64
import io
import json
import sys
import time
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

import config as C  # noqa: E402
import data_loader as D  # noqa: E402
import feature_extraction as FE  # noqa: E402
import preprocessing as P  # noqa: E402

RGB_JPG_CAL = C.DATA_PROCESSED / "rgb_jpg_calibration.json"


# ─── Modelos ──────────────────────────────────────────────────────────────────

@lru_cache(maxsize=None)
def load_bundle(name: str) -> dict:
    path = C.MODELS / f"{name}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Modelo {path.name} não encontrado. Execute: python src/train.py")
    return joblib.load(path)


def model_available(name: str) -> bool:
    return (C.MODELS / f"{name}.joblib").exists()


# ─── Imagem inteira (EuroSAT) ─────────────────────────────────────────────────

def classify_chip(dn: np.ndarray, top_features: int = 12) -> dict:
    """Classifica um recorte (H, W, 13) em DN do Sentinel-2 L1C (escala EuroSAT)."""
    bundle = load_bundle("rf_eurosat_ms")
    refl = P.fix_invalid(P.to_reflectance(dn, "sentinel2"))
    t0 = time.perf_counter()
    vec = FE.patch_features(refl, "sentinel2")
    proba = bundle["model"].predict_proba(vec)[0]
    ms = (time.perf_counter() - t0) * 1000
    return _chip_result(bundle, vec[0], proba, ms, top_features, refl, "sentinel2")


def classify_rgb(img_uint8: np.ndarray, top_features: int = 12) -> dict:
    """Classifica uma imagem RGB comum (JPG/PNG) com o modelo "só RGB".

    O JPG não traz reflectância: convertemos os níveis 0–255 com a calibração
    linear estimada nos pares EuroSAT RGB × EuroSAT MS (mesmos recortes).
    """
    bundle = load_bundle("rf_eurosat_rgb")
    cal = json.loads(RGB_JPG_CAL.read_text()) if RGB_JPG_CAL.exists() else {
        "slope": [1 / 255 * 0.275] * 3, "intercept": [0.0] * 3}
    x = img_uint8[..., :3].astype(np.float32)
    refl = np.stack([x[..., i] * cal["slope"][i] + cal["intercept"][i] for i in range(3)], axis=-1)
    refl = P.fix_invalid(refl)
    t0 = time.perf_counter()
    vec = FE.patch_features(refl, "rgb")
    proba = bundle["model"].predict_proba(vec)[0]
    ms = (time.perf_counter() - t0) * 1000
    return _chip_result(bundle, vec[0], proba, ms, top_features, refl, "rgb")


def _chip_result(bundle, vec, proba, ms, top_features, refl, sensor) -> dict:
    classes = bundle["classes"]
    k = int(np.argmax(proba))
    names = bundle["feature_names"]
    imp = bundle["model"].feature_importances_
    order = np.argsort(imp)[::-1][:top_features]
    cls = classes[k]
    out = {
        "model": "rf_eurosat_ms" if sensor == "sentinel2" else "rf_eurosat_rgb",
        "class_name": cls, "class_pt": C.EUROSAT_PT[cls], "macro_class": C.EUROSAT_TO_MACRO[cls],
        "color": C.EUROSAT_COLORS[cls], "confidence": float(proba[k]),
        "probabilities": {c: float(p) for c, p in zip(classes, proba)},
        "inference_ms": round(ms, 2), "n_features": len(names),
        "features": [{"name": names[i], "value": float(vec[i]), "importance": float(imp[i])}
                     for i in order],
    }
    if sensor == "sentinel2":
        for idx in FE.available_indices(sensor):
            out.setdefault("indices", {})[idx] = float(FE.spectral_index(refl, sensor, idx).mean())
    return out


# ─── Renderização ─────────────────────────────────────────────────────────────

def hex_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def labels_to_rgb(labels: np.ndarray, classes: list[str], colors: dict,
                  valid: np.ndarray | None = None, background=(20, 24, 33)) -> np.ndarray:
    lut = np.array([hex_rgb(colors[c]) for c in classes] + [background], dtype=np.uint8)
    lab = labels.astype(np.int64).copy()
    if valid is not None:
        lab[~valid] = len(classes)
    lab[(lab < 0) | (lab > len(classes))] = len(classes)
    return lut[lab]


def png_b64(arr: np.ndarray, max_side: int | None = None, nearest: bool = False) -> str:
    """PNG em base64; `nearest=True` para mapas temáticos (não mistura cores de classes)."""
    im = Image.fromarray(arr)
    if max_side and max(im.size) > max_side:
        im.thumbnail((max_side, max_side), Image.NEAREST if nearest else Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


# ─── Cenas do INPE (por pixel) ────────────────────────────────────────────────

@lru_cache(maxsize=2)
def scene_cache(name: str) -> dict:
    """Reflectância tratada + rótulos de referência, em memória (para a API)."""
    scene = D.load_scene(name)
    refl, valid = P.scene_reflectance(scene)
    from train import scene_labels
    labels = scene_labels(scene["ref"], valid)
    tc, limits = P.composite(refl, scene["sensor"], "cor_verdadeira", valid=valid)
    return dict(scene=scene, refl=refl, valid=valid, labels=labels, truecolor=tc,
                tc_limits=limits, sensor=scene["sensor"])


def classify_window(name: str, x0: int, y0: int, size: int = 256, majority: bool = True) -> dict:
    """Classifica pixel a pixel uma janela da cena (em tempo real)."""
    sc = scene_cache(name)
    bundle = load_bundle(f"rf_{name}_pixel")
    h, w = sc["valid"].shape
    size = int(min(size, h, w))
    x0, y0 = int(np.clip(x0, 0, w - size)), int(np.clip(y0, 0, h - size))
    m = 4  # margem para a janela 5×5 e o filtro de moda
    ya, yb, xa, xb = max(0, y0 - m), min(h, y0 + size + m), max(0, x0 - m), min(w, x0 + size + m)
    t0 = time.perf_counter()
    feats = FE.pixel_features(sc["refl"][ya:yb, xa:xb], sc["sensor"])
    valid = sc["valid"][ya:yb, xa:xb]
    lab = np.zeros(valid.shape, dtype=np.uint8)
    lab[valid] = bundle["model"].predict(feats[valid])
    if majority:
        lab = P.majority_filter(lab, len(C.LANDCOVER_CLASSES), 3, valid)
    ms = (time.perf_counter() - t0) * 1000
    sl = (slice(y0 - ya, y0 - ya + size), slice(x0 - xa, x0 - xa + size))
    lab, valid, feats = lab[sl], valid[sl], feats[sl]
    ref = sc["labels"][y0:y0 + size, x0:x0 + size]
    has_ref = ref >= 0
    agree = float((lab[has_ref] == ref[has_ref]).mean()) if has_ref.any() else None
    return dict(x0=x0, y0=y0, size=size, labels=lab, valid=valid, ref=ref, feats=feats,
                agreement=agree, ms=round(ms, 1), feature_names=bundle["feature_names"])


def classify_scene(name: str, majority: bool = True) -> dict:
    """Classifica todos os pixels válidos da cena (≈30 s no Sentinel-2)."""
    sc = scene_cache(name)
    bundle = load_bundle(f"rf_{name}_pixel")
    t0 = time.perf_counter()
    h, w = sc["valid"].shape
    lab = np.zeros((h, w), dtype=np.uint8)
    rows = 256  # em faixas, para não estourar a memória com o cubo de características
    for y in range(0, h, rows):
        ya, yb = max(0, y - 4), min(h, y + rows + 4)
        feats = FE.pixel_features(sc["refl"][ya:yb], sc["sensor"])
        v = sc["valid"][ya:yb]
        part = np.zeros(v.shape, dtype=np.uint8)
        part[v] = bundle["model"].predict(feats[v])
        lab[y:min(h, y + rows)] = part[y - ya:y - ya + min(rows, h - y)]
    if majority:
        lab = P.majority_filter(lab, len(C.LANDCOVER_CLASSES), 3, sc["valid"])
    secs = time.perf_counter() - t0
    has_ref = sc["labels"] >= 0
    agree = float((lab[has_ref] == sc["labels"][has_ref]).mean())
    return dict(labels=lab, valid=sc["valid"], seconds=round(secs, 1), agreement=agree,
                n_pixels=int(sc["valid"].sum()))


# ─── Por região: janela deslizante com o modelo do EuroSAT ───────────────────

def region_map(window: int = 64, stride: int = 32) -> dict:
    """Classifica a cena S2 em janelas 64×64 (640 m) com o modelo do EuroSAT.

    O DN do L1C atual é harmonizado (−1000) para a escala do EuroSAT. Cada
    janela recebe uma classe; a comparação com a referência usa as macro-classes
    e a classe WorldCover majoritária da janela.
    """
    bundle = load_bundle("rf_eurosat_ms")
    scene = D.load_scene("s2")
    dn = P.harmonize_l1c(scene["dn"])
    valid = P.valid_mask(scene["dn"], "sentinel2")
    from train import scene_labels
    ref = scene_labels(scene["ref"], valid)
    h, w, _ = dn.shape
    ys = list(range(0, h - window + 1, stride))
    xs = list(range(0, w - window + 1, stride))
    t0 = time.perf_counter()
    chips, pos = [], []
    for y in ys:
        for x in xs:
            chips.append(dn[y:y + window, x:x + window])
            pos.append((y, x))
    preds, probas = [], []
    for i in range(0, len(chips), 2000):
        refl = P.fix_invalid(P.to_reflectance(np.stack(chips[i:i + 2000]), "sentinel2"))
        pr = bundle["model"].predict_proba(FE.patch_features(refl, "sentinel2"))
        probas.append(pr.max(axis=1))
        preds.append(pr.argmax(axis=1))
    pred, conf = np.concatenate(preds), np.concatenate(probas)
    secs = time.perf_counter() - t0

    # cada janela pinta o seu bloco central stride×stride
    grid = np.full((h, w), -1, dtype=np.int64)
    off = (window - stride) // 2
    for (y, x), k in zip(pos, pred):
        grid[y + off:y + off + stride, x + off:x + off + stride] = k

    macro_names = [c for c in C.LANDCOVER_CLASSES if c != "Solo exposto"]
    to_macro = np.array([macro_names.index(C.EUROSAT_TO_MACRO[c]) for c in C.EUROSAT_CLASSES])
    lc_to_macro = {C.LANDCOVER_CLASSES.index(c): macro_names.index(c) for c in macro_names}
    agree, total, conf_m = 0, 0, np.zeros((4, 4), dtype=np.int64)
    for (y, x), k in zip(pos, pred):
        r = ref[y:y + window, x:x + window]
        r = r[r >= 0]
        if r.size < window * window * 0.5:
            continue
        maj = int(np.bincount(r, minlength=5).argmax())
        if maj not in lc_to_macro:  # solo exposto: sem classe equivalente no EuroSAT
            continue
        t, p = lc_to_macro[maj], int(to_macro[k])
        conf_m[t, p] += 1
        agree += int(t == p)
        total += 1
    counts = np.bincount(pred, minlength=10)
    return dict(grid=grid, pred=pred, conf=conf, pos=pos, seconds=round(secs, 1),
                n_windows=len(pred), agreement_macro=agree / max(total, 1), n_compared=total,
                confusion_macro=conf_m.tolist(), macro_names=macro_names,
                class_counts={c: int(n) for c, n in zip(C.EUROSAT_CLASSES, counts)})


def main() -> None:
    print("=" * 64 + "\n  Classificação por região — EuroSAT (Europa) → Mossoró/RN\n" + "=" * 64)
    r = region_map()
    print(f"  {r['n_windows']} janelas de 640 m em {r['seconds']} s")
    print(f"  concordância com a referência (4 macro-classes): {r['agreement_macro']:.1%} "
          f"em {r['n_compared']} janelas")
    for c, n in sorted(r["class_counts"].items(), key=lambda kv: -kv[1]):
        print(f"    {c:22s} {n:5d}")
    C.PREDICTIONS.mkdir(parents=True, exist_ok=True)
    np.save(C.PREDICTIONS / "s2_region_grid.npy", r["grid"])
    out = {k: v for k, v in r.items() if k not in ("grid", "pred", "conf", "pos")}
    out.update(window_px=64, window_m=640, stride_px=32, mean_confidence=float(r["conf"].mean()),
               model="rf_eurosat_ms", scene=D.scene_info()["s2"],
               note="Modelo treinado no EuroSAT (Europa) aplicado sem ajuste à caatinga: "
                    "mede a transferência entre domínios, não a acurácia do classificador.")
    (C.METRICS / "s2_region.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print("  ✓ results/metrics/s2_region.json")


if __name__ == "__main__":
    main()
