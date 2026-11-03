"""
figures.py — Etapas 7–9: figuras de resultado (visualização e análise de erros).

Gera em results/figures/ (e copia para presentation/img/):
  * cenas: cor verdadeira, falsa cor, SWIR, CBERS, índices NDVI/MNDWI/NDBI
  * pré-processamento: banda de 20 m antes/depois do alinhamento, máscara de nuvem
  * amostragem: mapa dos blocos de treino/validação/teste
  * mapas temáticos: por pixel (S2 e CBERS), referência, mapa de erros, por região
  * EuroSAT: amostras por classe (teste) e exemplos de erros (teste)
  * gráficos matplotlib: matrizes de confusão, métricas por classe, importância

Uso:  python src/figures.py   (ou python src/evaluation.py)
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))

import config as C  # noqa: E402
import data_loader as D  # noqa: E402
import feature_extraction as FE  # noqa: E402
import preprocessing as P  # noqa: E402
from predict import labels_to_rgb, scene_cache  # noqa: E402

FIG = C.FIGURES
SLIDE_IMG = C.PRESENTATION_DIR / "img"
# recorte de detalhe: centro de Mossoró, Rio Apodi-Mossoró e entorno (pixels S2)
ZOOM_S2 = (slice(820, 1620), slice(700, 1500))


def save(img: np.ndarray, name: str, width: int | None = 1280, nearest: bool = False) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    im = Image.fromarray(img)
    if width and im.width > width:
        im = im.resize((width, round(im.height * width / im.width)),
                       Image.NEAREST if nearest else Image.LANCZOS)
    im.save(FIG / name, optimize=True)
    print(f"  ✓ {name}")


def colormap(arr: np.ndarray, cmap: str, vmin: float, vmax: float, valid=None) -> np.ndarray:
    rgba = plt.get_cmap(cmap)(np.clip((arr - vmin) / (vmax - vmin), 0, 1))
    out = (rgba[..., :3] * 255).astype(np.uint8)
    if valid is not None:
        out[~valid] = (20, 24, 33)
    return out


def scene_figures() -> None:
    s2 = scene_cache("s2")
    refl, valid = s2["refl"], s2["valid"]
    save(s2["truecolor"], "s2_cor_verdadeira.png")
    save(s2["truecolor"][ZOOM_S2], "s2_cor_verdadeira_zoom.png", width=800)
    for kind in ("falsa_cor", "swir"):
        im, _ = P.composite(refl, "sentinel2", kind, valid=valid)
        save(im, f"s2_{kind}.png")
        save(im[ZOOM_S2], f"s2_{kind}_zoom.png", width=800)
    for name, cmap, lo, hi in [("NDVI", "RdYlGn", -0.1, 0.6), ("MNDWI", "RdBu", -0.6, 0.4),
                               ("NDBI", "PuOr", -0.4, 0.3), ("NDWI", "RdBu", -0.6, 0.3)]:
        idx = FE.spectral_index(refl, "sentinel2", name)
        save(colormap(idx, cmap, lo, hi, valid), f"s2_{name.lower()}.png")
        save(colormap(idx, cmap, lo, hi, valid)[ZOOM_S2], f"s2_{name.lower()}_zoom.png", width=800)

    cb = scene_cache("cbers")
    tc = cb["truecolor"].copy()
    save(tc, "cbers_cor_verdadeira.png")
    im, _ = P.composite(cb["refl"], "cbers4_mux", "falsa_cor", valid=cb["valid"])
    save(im, "cbers_falsa_cor.png")
    # máscara de nuvem do INPE sobreposta
    over = tc.copy()
    cloud = cb["scene"]["cmask"] == 255
    over[cloud] = (0.35 * over[cloud] + 0.65 * np.array([239, 79, 95])).astype(np.uint8)
    save(over, "cbers_mascara_nuvem.png")

    # alinhamento: B11 (20 m) lida crua × reamostrada para 10 m — recorte pequeno
    with __import__("rasterio").open(C.S2_STACK) as src:
        b11 = src.read(C.EUROSAT_BANDS.index("B11") + 1)[1150:1270, 1050:1170].astype(np.float32)
    blocky = b11[::2, ::2].repeat(2, axis=0).repeat(2, axis=1)
    lo, hi = np.percentile(b11, [2, 98])
    for arr, nm in ((blocky, "alinhamento_b11_20m.png"), (b11, "alinhamento_b11_10m.png")):
        g = (np.clip((arr - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
        save(np.stack([g] * 3, -1), nm, width=None)
        Image.open(FIG / nm).resize((480, 480), Image.NEAREST).save(FIG / nm)


def map_figures() -> None:
    classes, colors = C.LANDCOVER_CLASSES, C.LANDCOVER_COLORS
    for name in ("s2", "cbers"):
        sc = scene_cache(name)
        lab = np.load(C.PREDICTIONS / f"{name}_map.npy")
        save(labels_to_rgb(lab, classes, colors, sc["valid"]), f"{name}_mapa.png", nearest=True)
        save(labels_to_rgb(sc["labels"], classes, colors, sc["labels"] >= 0),
             f"{name}_referencia.png", nearest=True)
        # erros: só nos blocos de TESTE
        split = np.load(C.DATA_PROCESSED / f"{name}_split.npy")
        err = np.full(lab.shape + (3,), (20, 24, 33), dtype=np.uint8)
        test = (split == 2) & (sc["labels"] >= 0)
        base = (sc["truecolor"] * 0.35).astype(np.uint8)
        err[:] = base
        err[test & (lab == sc["labels"])] = (52, 211, 153)
        err[test & (lab != sc["labels"])] = (239, 79, 95)
        save(err, f"{name}_erros.png", nearest=True)
        if name == "s2":
            raw = np.load(C.PREDICTIONS / "s2_map_raw.npy")
            for arr, nm in ((raw, "s2_mapa_sem_filtro_zoom.png"), (lab, "s2_mapa_zoom.png")):
                save(labels_to_rgb(arr, classes, colors, sc["valid"])[ZOOM_S2], nm, width=800, nearest=True)
            save(labels_to_rgb(sc["labels"], classes, colors, sc["labels"] >= 0)[ZOOM_S2],
                 "s2_referencia_zoom.png", width=800, nearest=True)
            save(err[ZOOM_S2], "s2_erros_zoom.png", width=800, nearest=True)
            # blocos de amostragem
            sp = np.full(split.shape + (3,), (60, 66, 80), dtype=np.uint8)
            sp[split == 0], sp[split == 1], sp[split == 2] = (47, 107, 216), (233, 184, 36), (214, 58, 74)
            blend = (0.55 * sp + 0.45 * sc["truecolor"]).astype(np.uint8)
            save(blend, "s2_blocos_divisao.png")

    # por região (EuroSAT → Mossoró), em macro-classes
    grid_p = C.PREDICTIONS / "s2_region_grid.npy"
    if grid_p.exists():
        grid = np.load(grid_p)
        sc = scene_cache("s2")
        names = C.EUROSAT_CLASSES
        rgb = labels_to_rgb(np.where(grid >= 0, grid, 0), names, C.EUROSAT_COLORS, grid >= 0)
        save(rgb, "s2_regiao_eurosat.png", nearest=True)
        macro = np.array([C.LANDCOVER_CLASSES.index(C.EUROSAT_TO_MACRO[c]) for c in names])
        mg = np.where(grid >= 0, macro[np.clip(grid, 0, None)], -1)
        save(labels_to_rgb(mg, classes, colors, grid >= 0), "s2_regiao_macro.png", nearest=True)
        save(labels_to_rgb(mg, classes, colors, grid >= 0)[ZOOM_S2], "s2_regiao_macro_zoom.png",
             width=800, nearest=True)


def chip_rgb(dn: np.ndarray) -> np.ndarray:
    refl = P.to_reflectance(dn, "sentinel2")
    tc, _ = P.stretch(refl[..., [3, 2, 1]], limits=(np.full(3, 0.0), np.full(3, 0.30)), gamma=1.2)
    return tc


def eurosat_figures() -> None:
    p = C.DATA_PROCESSED / "eurosat_demo_samples.npz"
    if not p.exists():
        return
    d = np.load(p)
    chips, y = d["chips"], d["y"]
    # grade 10 classes × 6 amostras (cor verdadeira, ampliadas 2×)
    tile, gap, n = 128, 8, 6
    canvas = Image.new("RGB", (n * (tile + gap) - gap, 10 * (tile + gap) - gap), (255, 255, 255))
    for k in range(10):
        idx = np.flatnonzero(y == k)[:n]
        for j, i in enumerate(idx):
            im = Image.fromarray(chip_rgb(chips[i])).resize((tile, tile), Image.NEAREST)
            canvas.paste(im, (j * (tile + gap), k * (tile + gap)))
    canvas.save(FIG / "eurosat_amostras.png", optimize=True)
    # uma por classe em linha (capa/slide de classes)
    row = Image.new("RGB", (10 * (tile + gap) - gap, tile), (255, 255, 255))
    for k in range(10):
        i = np.flatnonzero(y == k)[1]
        row.paste(Image.fromarray(chip_rgb(chips[i])).resize((tile, tile), Image.NEAREST),
                  (k * (tile + gap), 0))
    row.save(FIG / "eurosat_uma_por_classe.png", optimize=True)
    for k in range(10):  # chips individuais para o slide de classes
        i = np.flatnonzero(y == k)[1]
        Image.fromarray(chip_rgb(chips[i])).resize((192, 192), Image.NEAREST).save(
            FIG / f"eurosat_{C.EUROSAT_CLASSES[k]}.png")
    # composições do mesmo recorte: cor verdadeira, falsa cor, NDVI
    i = np.flatnonzero(y == C.EUROSAT_CLASSES.index("River"))[2]
    refl = P.to_reflectance(chips[i], "sentinel2")
    fc, _ = P.stretch(refl[..., [7, 3, 2]], limits=(np.zeros(3), np.array([0.4, 0.25, 0.25])))
    nd = colormap(FE.spectral_index(refl, "sentinel2", "NDVI"), "RdYlGn", -0.2, 0.8)
    for arr, nm in ((chip_rgb(chips[i]), "chip_cor_verdadeira.png"), (fc, "chip_falsa_cor.png"),
                    (nd, "chip_ndvi.png")):
        Image.fromarray(arr).resize((256, 256), Image.NEAREST).save(FIG / nm)
    print("  ✓ eurosat_amostras.png, chips")
    error_examples(d)


def error_examples(d) -> None:
    """Erros reais do conjunto de TESTE: os pares de classes mais confundidos."""
    pr = np.load(C.METRICS / "eurosat_test_predictions.npz")
    pos = {int(g): j for j, g in enumerate(pr["index"])}
    pairs = {}
    for j, g in enumerate(d["index"]):
        k = pos[int(g)]
        t, p = int(pr["y_true"][k]), int(pr["y_pred"][k])
        if t != p:
            pairs.setdefault((t, p), []).append((j, float(pr["proba"][k].max())))
    top = sorted(pairs.items(), key=lambda kv: -len(kv[1]))[:4]
    tile = 150
    out = []
    for (t, p), items in top:
        for j, conf in items[:1]:
            out.append({"true": C.EUROSAT_CLASSES[t], "pred": C.EUROSAT_CLASSES[p], "conf": conf,
                        "file": f"erro_{len(out)}.png"})
            Image.fromarray(chip_rgb(d["chips"][j])).resize((tile, tile), Image.NEAREST).save(
                FIG / out[-1]["file"])
    (C.METRICS / "eurosat_error_examples.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"  ✓ {len(out)} exemplos de erro (teste)")


def chart_figures() -> None:
    """Versões estáticas (matplotlib) das métricas, para o README e os documentos."""
    for name, title in (("eurosat", "EuroSAT — imagem inteira"), ("s2_pixel", "Sentinel-2 — pixel"),
                        ("cbers_pixel", "CBERS-4 — pixel")):
        p = C.METRICS / f"{name}.json"
        if not p.exists():
            continue
        m = json.loads(p.read_text())["test"]
        cm = np.array(m["confusion_matrix"], dtype=float)
        cmn = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
        labels = [C.EUROSAT_PT.get(c, c) for c in m["labels"]]
        fig, ax = plt.subplots(figsize=(8.5, 7) if len(labels) > 5 else (6.5, 5.5))
        ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if cmn[i, j] > 0.55 else "#0e1f3d")
        ax.set_xticks(range(len(labels)), labels, rotation=40, ha="right")
        ax.set_yticks(range(len(labels)), labels)
        ax.set_xlabel("classe prevista")
        ax.set_ylabel("classe de referência")
        ax.set_title(f"Matriz de confusão normalizada (teste) — {title}")
        fig.tight_layout()
        fig.savefig(FIG / f"confusao_{name}.png", dpi=130)
        plt.close(fig)
        print(f"  ✓ confusao_{name}.png")


def copy_to_slides() -> None:
    """Copia para presentation/img só as figuras que o deck usa (mantém o repositório leve)."""
    import re
    SLIDE_IMG.mkdir(parents=True, exist_ok=True)
    html = (C.PRESENTATION_DIR / "index.html").read_text()
    used = set(re.findall(r"img/([\w-]+\.png)", html))
    n = 0
    for f in FIG.glob("*.png"):
        if f.name in used or f.name.startswith("erro_"):
            shutil.copy2(f, SLIDE_IMG / f.name)
            n += 1
    print(f"  ✓ {n} figuras copiadas para presentation/img/")


def main() -> None:
    print("=" * 64 + "\n  Figuras de resultado\n" + "=" * 64)
    scene_figures()
    map_figures()
    eurosat_figures()
    chart_figures()
    copy_to_slides()


if __name__ == "__main__":
    main()
