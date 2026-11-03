"""
app.py — Etapa 9: backend Flask da interface de demonstração.

Todas as respostas de classificação são calculadas NA HORA pelos modelos
treinados (nenhum resultado fixo): o tempo de inferência volta em cada resposta.

Endpoints
  GET  /                                   interface web (web/index.html)
  GET  /api/status                         modelos, dados e cenas disponíveis
  GET  /api/metrics                        resumo das métricas de teste
  GET  /api/metrics/<nome>                 métricas completas (eurosat, s2_pixel, cbers_pixel, s2_region)
  GET  /api/eurosat/samples                recortes do conjunto de TESTE (nunca vistos no treino)
  GET  /api/eurosat/classify/<id>          classifica um recorte de teste
  POST /api/eurosat/upload                 classifica um arquivo: .tif 13 bandas (EuroSAT) ou .jpg/.png (RGB)
  GET  /api/scenes                         cenas do INPE (metadados + prévia)
  POST /api/scenes/<cena>/window           classifica pixel a pixel uma janela {x, y, size}
  GET  /api/scenes/<cena>/pixel?x=&y=      características e classe de um pixel
  POST /api/scenes/<cena>/full             classifica a cena inteira (≈30 s no Sentinel-2)

Uso:  python src/app.py   →  http://localhost:5000
"""

from __future__ import annotations

import io
import json
import logging
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

import config as C  # noqa: E402
import data_loader as D  # noqa: E402
import feature_extraction as FE  # noqa: E402
import predict as PR  # noqa: E402
import preprocessing as P  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("trab10")

app = Flask(__name__, static_folder=None)
app.json.sort_keys = False  # mantém a ordem das classes nas respostas
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # uploads de até 20 MB

DEMO = C.DATA_PROCESSED / "eurosat_demo_samples.npz"
PREVIEW_SIDE = 900


def error(msg: str, code: int = 400):
    return jsonify({"error": msg}), code


# ─── Arquivos estáticos ───────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(C.WEB_DIR, "index.html")


@app.route("/web/<path:name>")
def web_static(name):
    return send_from_directory(C.WEB_DIR, name)


@app.route("/vendor/<path:name>")
def vendor(name):
    return send_from_directory(C.PRESENTATION_DIR / "vendor", name)


# ─── Status e métricas ────────────────────────────────────────────────────────

MODELS = ["rf_eurosat_ms", "rf_eurosat_rgb", "rf_s2_pixel", "rf_cbers_pixel"]


@app.route("/api/status")
def status():
    return jsonify({
        "models": {m: PR.model_available(m) for m in MODELS},
        "demo_samples": DEMO.exists(),
        "scenes": {"s2": C.S2_STACK.exists(), "cbers": C.CBERS_STACK.exists()},
        "metrics": sorted(p.stem for p in C.METRICS.glob("*.json")),
    })


@app.route("/api/metrics")
def metrics_summary():
    out = {}
    for name in ("eurosat", "s2_pixel", "cbers_pixel"):
        p = C.METRICS / f"{name}.json"
        if p.exists():
            m = json.loads(p.read_text())
            t = m["test"]
            out[name] = {"level": m["level"], "accuracy": t["accuracy"], "kappa": t["kappa"],
                         "macro_f1": t["macro_f1"], "n_test": t["n"], "classes": m["classes"],
                         "per_class": t["per_class"], "confusion_matrix": t["confusion_matrix"],
                         "split": m["split"], "params": m["params"],
                         "feature_importance": m["feature_importance"][:10]}
            if name == "eurosat":
                out[name]["comparisons"] = m["comparisons"]
                out[name]["random_split_test"] = m["random_split_test"]
    p = C.METRICS / "s2_region.json"
    if p.exists():
        out["s2_region"] = json.loads(p.read_text())
    return jsonify(out)


@app.route("/api/metrics/<name>")
def metrics_full(name):
    p = C.METRICS / f"{name}.json"
    if not p.exists() or p.parent != C.METRICS:
        return error(f"métricas '{name}' não encontradas", 404)
    return jsonify(json.loads(p.read_text()))


# ─── Imagem inteira (EuroSAT) ─────────────────────────────────────────────────

@lru_cache(maxsize=1)
def demo_samples() -> dict:
    if not DEMO.exists():
        raise FileNotFoundError("Execute python src/train.py eurosat")
    d = np.load(DEMO)
    return {k: d[k] for k in d.files}


def chip_views(dn: np.ndarray) -> dict:
    """Composições do recorte para a tela (realce fixo: comparável entre recortes)."""
    refl = P.to_reflectance(dn, "sentinel2")
    up = lambda a: np.asarray(Image.fromarray(a).resize((256, 256), Image.NEAREST))  # noqa: E731
    tc, _ = P.stretch(refl[..., [3, 2, 1]], limits=(np.zeros(3), np.full(3, 0.30)), gamma=1.2)
    fc, _ = P.stretch(refl[..., [7, 3, 2]], limits=(np.zeros(3), np.array([0.45, 0.25, 0.25])))
    sw, _ = P.stretch(refl[..., [11, 7, 3]], limits=(np.zeros(3), np.array([0.35, 0.45, 0.25])))
    from figures import colormap
    nd = colormap(FE.spectral_index(refl, "sentinel2", "NDVI"), "RdYlGn", -0.2, 0.8)
    return {k: PR.png_b64(up(v)) for k, v in
            (("cor_verdadeira", tc), ("falsa_cor", fc), ("swir", sw), ("ndvi", nd))}


@app.route("/api/eurosat/samples")
def eurosat_samples():
    try:
        d = demo_samples()
    except FileNotFoundError as e:
        return error(str(e), 503)
    items = []
    for i, (chip, y) in enumerate(zip(d["chips"], d["y"])):
        refl = P.to_reflectance(chip, "sentinel2")
        tc, _ = P.stretch(refl[..., [3, 2, 1]], limits=(np.zeros(3), np.full(3, 0.30)), gamma=1.2)
        cls = C.EUROSAT_CLASSES[int(y)]
        items.append({"id": i, "class": cls, "class_pt": C.EUROSAT_PT[cls],
                      "file": str(d["files"][i]), "thumb": PR.png_b64(tc)})
    return jsonify({"samples": items, "note": "recortes do conjunto de teste (divisão espacial)"})


@app.route("/api/eurosat/classify/<int:sid>")
def eurosat_classify(sid: int):
    try:
        d = demo_samples()
    except FileNotFoundError as e:
        return error(str(e), 503)
    if not 0 <= sid < len(d["y"]):
        return error("amostra inexistente", 404)
    chip = d["chips"][sid]
    res = PR.classify_chip(chip)
    true = C.EUROSAT_CLASSES[int(d["y"][sid])]
    res.update(true_class=true, true_pt=C.EUROSAT_PT[true], correct=res["class_name"] == true,
               file=str(d["files"][sid]), views=chip_views(chip))
    return jsonify(res)


@app.route("/api/eurosat/upload", methods=["POST"])
def eurosat_upload():
    f = request.files.get("file")
    if f is None or not f.filename:
        return error("envie um arquivo no campo 'file'")
    ext = Path(f.filename).suffix.lower()
    data = f.read()
    if not data:
        return error("arquivo vazio")
    try:
        if ext in (".tif", ".tiff"):
            import rasterio
            with rasterio.MemoryFile(data) as mem, mem.open() as src:
                if src.count != 13:
                    return error(f"GeoTIFF com {src.count} bandas; o modelo espera as 13 bandas "
                                 "do Sentinel-2 na ordem do EuroSAT MS")
                dn = np.transpose(src.read(), (1, 2, 0))
            res = PR.classify_chip(dn)
            res["views"] = chip_views(dn)
            res["input"] = "GeoTIFF 13 bandas → modelo multiespectral"
        elif ext in (".jpg", ".jpeg", ".png"):
            img = Image.open(io.BytesIO(data)).convert("RGB")
            resized = img.size != (64, 64)
            if resized:
                img = img.resize((64, 64), Image.LANCZOS)
            arr = np.asarray(img)
            res = PR.classify_rgb(arr)
            res["views"] = {"cor_verdadeira": PR.png_b64(np.asarray(img.resize((256, 256), Image.NEAREST)))}
            res["input"] = ("JPG/PNG → modelo só RGB (sem NIR/SWIR, sem índices espectrais)"
                            + (" · redimensionada para 64×64" if resized else ""))
        else:
            return error(f"formato não suportado: {ext} (use .tif, .jpg ou .png)")
    except FileNotFoundError as e:
        return error(str(e), 503)
    except Exception as e:  # arquivo corrompido etc.
        log.exception("upload")
        return error(f"não foi possível ler o arquivo: {e}")
    return jsonify(res)


# ─── Cenas do INPE (por pixel) ────────────────────────────────────────────────

def _scene_or_404(name):
    if name not in ("s2", "cbers"):
        return None
    return PR.scene_cache(name)


@lru_cache(maxsize=2)
def _preview(name: str) -> dict:
    sc = PR.scene_cache(name)
    h, w = sc["valid"].shape
    return {"truecolor": PR.png_b64(sc["truecolor"], PREVIEW_SIDE),
            "reference": PR.png_b64(PR.labels_to_rgb(sc["labels"], C.LANDCOVER_CLASSES,
                                                     C.LANDCOVER_COLORS, sc["labels"] >= 0),
                                    PREVIEW_SIDE, nearest=True),
            "height": h, "width": w}


@app.route("/api/scenes")
def scenes():
    info = D.scene_info()
    out = []
    for name in ("s2", "cbers"):
        if not PR.model_available(f"rf_{name}_pixel"):
            continue
        sc = PR.scene_cache(name)
        out.append({"id": name, **info[name], **_preview(name), "aoi": C.AOI_NAME,
                    "bands_list": sc["scene"]["bands"],
                    "indices": FE.available_indices(sc["sensor"]),
                    "valid_fraction": float(sc["valid"].mean())})
    return jsonify({"scenes": out, "classes": [{"name": c, "color": C.LANDCOVER_COLORS[c]}
                                               for c in C.LANDCOVER_CLASSES]})


def _fractions(lab, valid):
    n = max(int(valid.sum()), 1)
    return {c: float(((lab == i) & valid).sum() / n) for i, c in enumerate(C.LANDCOVER_CLASSES)}


@app.route("/api/scenes/<name>/window", methods=["POST"])
def scene_window(name):
    sc = _scene_or_404(name)
    if sc is None:
        return error("cena inexistente", 404)
    body = request.get_json(silent=True) or {}
    size = int(body.get("size", 256))
    size = max(32, min(size, 512))
    r = PR.classify_window(name, int(body.get("x", 0)), int(body.get("y", 0)), size,
                           majority=bool(body.get("majority", True)))
    sl = (slice(r["y0"], r["y0"] + r["size"]), slice(r["x0"], r["x0"] + r["size"]))
    up = lambda a: np.asarray(Image.fromarray(a).resize((512, 512), Image.NEAREST))  # noqa: E731
    lab_rgb = PR.labels_to_rgb(r["labels"], C.LANDCOVER_CLASSES, C.LANDCOVER_COLORS, r["valid"])
    ref_rgb = PR.labels_to_rgb(r["ref"], C.LANDCOVER_CLASSES, C.LANDCOVER_COLORS, r["ref"] >= 0)
    return jsonify({
        "x": r["x0"], "y": r["y0"], "size": r["size"], "ms": r["ms"],
        "n_pixels": int(r["valid"].sum()), "agreement": r["agreement"],
        "fractions": _fractions(r["labels"], r["valid"]),
        "truecolor": PR.png_b64(up(sc["truecolor"][sl])),
        "map": PR.png_b64(up(lab_rgb)), "reference": PR.png_b64(up(ref_rgb)),
        "resolution_m": sc["scene"]["res"],
    })


@app.route("/api/scenes/<name>/pixel")
def scene_pixel(name):
    sc = _scene_or_404(name)
    if sc is None:
        return error("cena inexistente", 404)
    h, w = sc["valid"].shape
    x = int(np.clip(int(request.args.get("x", 0)), 0, w - 1))
    y = int(np.clip(int(request.args.get("y", 0)), 0, h - 1))
    if not sc["valid"][y, x]:
        return jsonify({"x": x, "y": y, "valid": False,
                        "reason": "pixel sem dado ou coberto por nuvem (máscara)"})
    bundle = PR.load_bundle(f"rf_{name}_pixel")
    ya, yb, xa, xb = max(0, y - 3), min(h, y + 4), max(0, x - 3), min(w, x + 4)
    t0 = time.perf_counter()
    feats = FE.pixel_features(sc["refl"][ya:yb, xa:xb], sc["sensor"])[y - ya, x - xa]
    proba = bundle["model"].predict_proba(feats[None])[0]
    ms = (time.perf_counter() - t0) * 1000
    k = int(proba.argmax())
    ref = int(sc["labels"][y, x])
    code = int(sc["scene"]["ref"][y, x])
    return jsonify({
        "x": x, "y": y, "valid": True, "ms": round(ms, 2),
        "class_name": C.LANDCOVER_CLASSES[k], "color": C.LANDCOVER_COLORS[C.LANDCOVER_CLASSES[k]],
        "probabilities": {c: float(p) for c, p in zip(C.LANDCOVER_CLASSES, proba)},
        "reference": C.LANDCOVER_CLASSES[ref] if ref >= 0 else None,
        "worldcover": f"{code} — {C.WORLDCOVER_NAMES.get(code, '?')}",
        "features": [{"name": n, "value": float(v)} for n, v in zip(bundle["feature_names"], feats)],
    })


@app.route("/api/scenes/<name>/full", methods=["POST"])
def scene_full(name):
    sc = _scene_or_404(name)
    if sc is None:
        return error("cena inexistente", 404)
    r = PR.classify_scene(name)
    rgb = PR.labels_to_rgb(r["labels"], C.LANDCOVER_CLASSES, C.LANDCOVER_COLORS, r["valid"])
    px_km2 = sc["scene"]["res"] ** 2 / 1e6
    fr = _fractions(r["labels"], r["valid"])
    return jsonify({"seconds": r["seconds"], "agreement": r["agreement"], "n_pixels": r["n_pixels"],
                    "fractions": fr,
                    "area_km2": {c: v * r["n_pixels"] * px_km2 for c, v in fr.items()},
                    "map": PR.png_b64(rgb, PREVIEW_SIDE, nearest=True)})


if __name__ == "__main__":
    print("=" * 56 + "\n  Classificador de cobertura do solo — Trabalho 10 PDI\n" + "=" * 56)
    for m in MODELS:
        print(f"  {'✓' if PR.model_available(m) else '✗'} {m}")
    print("\n  Interface: http://localhost:5000\n")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
