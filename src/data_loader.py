"""
data_loader.py — Etapa 1: aquisição e leitura dos dados.

Três fontes, todas públicas e com procedência documentada:

1. EuroSAT MS (Zenodo): 27.000 recortes Sentinel-2 de 64×64 px, 13 bandas,
   10 classes rotuladas. Usado para o experimento de *imagem inteira*.
2. INPE — Brazil Data Cube (STAC): uma cena Sentinel-2 L1C e uma cena
   CBERS-4 MUX sobre Mossoró/RN. Usadas nos experimentos *por pixel* e *por região*.
3. ESA WorldCover 2021 (10 m): mapa de referência usado como rótulo dos pixels.

Uso:
    python src/data_loader.py            # baixa e prepara tudo
    python src/data_loader.py eurosat    # só o EuroSAT
    python src/data_loader.py inpe       # só as cenas do INPE + referência
"""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).parent))

import config as C  # noqa: E402

try:
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.transform import from_origin
    from rasterio.warp import reproject, transform_bounds
    from rasterio.windows import from_bounds
except ImportError as exc:  # pragma: no cover - mensagem amigável
    raise SystemExit("rasterio é necessário: pip install -r requirements.txt") from exc


GDAL_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "GDAL_HTTP_MAX_RETRY": "4",
    "GDAL_HTTP_RETRY_DELAY": "2",
}


# ─── Download ─────────────────────────────────────────────────────────────────

def download(url: str, dest: Path, desc: str | None = None) -> Path:
    """Baixa `url` para `dest` (pula se já existe com o tamanho anunciado)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    head = requests.head(url, allow_redirects=True, timeout=60)
    size = int(head.headers.get("content-length", 0))
    if dest.exists() and (size == 0 or dest.stat().st_size == size):
        print(f"  ✓ já existe: {dest.relative_to(C.ROOT)}")
        return dest
    print(f"  ↓ {desc or dest.name} ({size / 1e6:.0f} MB)")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        done = 0
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                done += len(chunk)
                if size:
                    print(f"\r    {done / size:6.1%}", end="", flush=True)
    print()
    tmp.rename(dest)
    return dest


def stac_item(collection: str, item_id: str) -> dict:
    """Busca um item no catálogo STAC do INPE (Brazil Data Cube)."""
    url = f"{C.INPE_STAC}/collections/{collection}/items/{item_id}"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.json()


# ─── EuroSAT MS ───────────────────────────────────────────────────────────────

def ensure_eurosat_ms() -> Path:
    """Baixa e extrai o EuroSAT MS (≈2 GB) em data/raw/EuroSAT_MS/<Classe>/*.tif."""
    if C.EUROSAT_MS_DIR.exists() and any(C.EUROSAT_MS_DIR.glob("*/*.tif")):
        return C.EUROSAT_MS_DIR
    download(C.EUROSAT_MS_URL, C.EUROSAT_MS_ZIP, "EuroSAT_MS.zip")
    print("  ⇲ extraindo EuroSAT_MS.zip …")
    with zipfile.ZipFile(C.EUROSAT_MS_ZIP) as zf:
        zf.extractall(C.DATA_RAW)
    # o zip pode trazer uma pasta com outro nome; normaliza para EuroSAT_MS/
    if not C.EUROSAT_MS_DIR.exists():
        for d in C.DATA_RAW.iterdir():
            if d.is_dir() and (d / "Forest").exists() and any((d / "Forest").glob("*.tif")):
                d.rename(C.EUROSAT_MS_DIR)
                break
    n = len(list(C.EUROSAT_MS_DIR.glob("*/*.tif")))
    print(f"  ✓ EuroSAT MS: {n} imagens")
    return C.EUROSAT_MS_DIR


def eurosat_files() -> list[tuple[Path, int]]:
    """Lista (caminho, rótulo) de todas as imagens do EuroSAT MS, em ordem fixa."""
    out = []
    for label, cls in enumerate(C.EUROSAT_CLASSES):
        files = sorted((C.EUROSAT_MS_DIR / cls).glob("*.tif"),
                       key=lambda p: int(p.stem.split("_")[-1]))
        out += [(p, label) for p in files]
    return out


def read_eurosat_tif(path: Path) -> tuple[np.ndarray, tuple[float, float]]:
    """Lê um recorte EuroSAT MS → (64, 64, 13) uint16 e o centro (lon, lat)."""
    with rasterio.open(path) as src:
        arr = src.read()  # (13, 64, 64)
        b = src.bounds
        cx, cy = (b.left + b.right) / 2, (b.bottom + b.top) / 2
        lon, lat = transform_bounds(src.crs, "EPSG:4326", cx, cy, cx, cy)[:2]
    return np.transpose(arr, (1, 2, 0)), (lon, lat)


# ─── Grade da área de estudo ──────────────────────────────────────────────────

def aoi_grid(res: float) -> tuple[rasterio.Affine, int, int]:
    """Grade regular em UTM 24S cobrindo a AOI, alinhada a múltiplos de 20 m.

    S2 (10 m) e CBERS (20 m) usam a mesma origem: um pixel CBERS cobre
    exatamente 2×2 pixels Sentinel-2, o que permite comparar os mapas.
    """
    left, bottom, right, top = transform_bounds("EPSG:4326", C.AOI_CRS, *C.AOI_BBOX)
    left, top = np.floor(left / 20) * 20, np.ceil(top / 20) * 20
    right, bottom = np.ceil(right / 20) * 20, np.floor(bottom / 20) * 20
    width, height = int(round((right - left) / res)), int(round((top - bottom) / res))
    return from_origin(left, top, res, res), width, height


def _write_stack(path: Path, arr: np.ndarray, transform, dtype, nodata, names, tags=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = dict(driver="GTiff", height=arr.shape[1], width=arr.shape[2], count=arr.shape[0],
                   dtype=dtype, crs=C.AOI_CRS, transform=transform, nodata=nodata,
                   compress="deflate", tiled=True, blockxsize=256, blockysize=256)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr.astype(dtype))
        for i, n in enumerate(names, start=1):
            dst.set_band_description(i, n)
        if tags:
            dst.update_tags(**tags)


# ─── Sentinel-2 L1C (INPE/BDC) ────────────────────────────────────────────────

def ensure_s2_zip() -> Path:
    item = stac_item(C.S2_COLLECTION, C.S2_ITEM)
    href = item["assets"]["asset"]["href"]
    return download(href, C.S2_ZIP, f"Sentinel-2 L1C {C.S2_ITEM[:24]}… (INPE)")


def prepare_s2_scene() -> Path:
    """Recorta as 13 bandas do L1C na AOI e reamostra todas para 10 m.

    É o *alinhamento espacial das bandas*: B05–B07, B8A, B11 e B12 têm 20 m e
    B01, B09 e B10 têm 60 m. Reamostragem bilinear para a grade de 10 m.
    """
    if C.S2_STACK.exists():
        return C.S2_STACK
    zpath = ensure_s2_zip()
    with zipfile.ZipFile(zpath) as zf:
        jp2 = {Path(n).stem.split("_")[-1]: n for n in zf.namelist()
               if "/IMG_DATA/" in n and n.endswith(".jp2")}
    transform, w, h = aoi_grid(10)
    out = np.zeros((len(C.EUROSAT_BANDS), h, w), dtype=np.uint16)
    with rasterio.Env(**GDAL_ENV):
        for i, band in enumerate(C.EUROSAT_BANDS):
            with rasterio.open(f"/vsizip/{zpath}/{jp2[band]}") as src:
                # lê só a janela da AOI (com margem) e reprojeta para a grade
                left, bottom, right, top = rasterio.transform.array_bounds(h, w, transform)
                win = from_bounds(left - 120, bottom - 120, right + 120, top + 120, src.transform)
                data = src.read(1, window=win)
                reproject(data, out[i], src_transform=src.window_transform(win), src_crs=src.crs,
                          dst_transform=transform, dst_crs=C.AOI_CRS,
                          resampling=Resampling.nearest if src.res[0] == 10 else Resampling.bilinear,
                          src_nodata=0, dst_nodata=0)
                print(f"    {band}: {src.res[0]:.0f} m → 10 m")
    _write_stack(C.S2_STACK, out, transform, "uint16", 0, C.EUROSAT_BANDS,
                 tags={"source": f"INPE BDC {C.S2_COLLECTION} / {C.S2_ITEM}",
                       "radio_add_offset": C.S2_RADIO_ADD_OFFSET})
    print(f"  ✓ {C.S2_STACK.relative_to(C.ROOT)}  ({h}×{w} px, 13 bandas)")
    return C.S2_STACK


# ─── CBERS-4 MUX (INPE/BDC) ───────────────────────────────────────────────────

def prepare_cbers_scene() -> Path:
    """Lê as 4 bandas MUX + a máscara de nuvem direto dos COGs do INPE (janela da AOI)."""
    if C.CBERS_STACK.exists():
        return C.CBERS_STACK
    item = stac_item(C.CBERS_COLLECTION, C.CBERS_ITEM)
    transform, w, h = aoi_grid(20)
    names = C.CBERS_BANDS + ["CMASK"]
    out = np.full((len(names), h, w), C.CBERS_NODATA, dtype=np.int16)
    with rasterio.Env(**GDAL_ENV):
        for i, band in enumerate(names):
            href = item["assets"][band]["href"]
            with rasterio.open(f"/vsicurl/{href}") as src:
                left, bottom, right, top = rasterio.transform.array_bounds(h, w, transform)
                win = from_bounds(left - 60, bottom - 60, right + 60, top + 60, src.transform)
                data = src.read(1, window=win).astype(np.int16)
                nod = 0 if band == "CMASK" else C.CBERS_NODATA
                if band == "CMASK":
                    out[i] = 0
                reproject(data, out[i], src_transform=src.window_transform(win), src_crs=src.crs,
                          dst_transform=transform, dst_crs=C.AOI_CRS,
                          resampling=Resampling.nearest, src_nodata=nod, dst_nodata=nod)
            print(f"    {band} ✓")
    _write_stack(C.CBERS_STACK, out, transform, "int16", None, names,
                 tags={"source": f"INPE BDC {C.CBERS_COLLECTION} / {C.CBERS_ITEM}"})
    print(f"  ✓ {C.CBERS_STACK.relative_to(C.ROOT)}  ({h}×{w} px, 4 bandas + CMASK)")
    return C.CBERS_STACK


# ─── Referência: ESA WorldCover 2021 ──────────────────────────────────────────

def prepare_reference(stack_path: Path, out_path: Path) -> Path:
    """Reprojeta o WorldCover para a grade de uma cena.

    10 m → 10 m: vizinho mais próximo. 10 m → 20 m: *moda* (classe mais
    frequente dos 2×2 pixels), que é a forma correta de agregar rótulos.
    """
    if out_path.exists():
        return out_path
    download(C.WORLDCOVER_URL, C.WORLDCOVER_TIF, "ESA WorldCover 2021 (tile S06W039)")
    with rasterio.open(stack_path) as ref:
        transform, h, w, res = ref.transform, ref.height, ref.width, ref.res[0]
    out = np.zeros((1, h, w), dtype=np.uint8)
    with rasterio.open(C.WORLDCOVER_TIF) as src:
        bounds = transform_bounds(C.AOI_CRS, src.crs, *rasterio.transform.array_bounds(h, w, transform))
        win = from_bounds(bounds[0] - 0.002, bounds[1] - 0.002, bounds[2] + 0.002, bounds[3] + 0.002,
                          src.transform)
        data = src.read(1, window=win)
        reproject(data, out[0], src_transform=src.window_transform(win), src_crs=src.crs,
                  dst_transform=transform, dst_crs=C.AOI_CRS,
                  resampling=Resampling.nearest if res <= 10 else Resampling.mode,
                  src_nodata=0, dst_nodata=0)
    _write_stack(out_path, out, transform, "uint8", 0, ["worldcover_2021"],
                 tags={"source": "ESA WorldCover 10m 2021 v200"})
    print(f"  ✓ {out_path.relative_to(C.ROOT)}")
    return out_path


# ─── Leitura das cenas preparadas ─────────────────────────────────────────────

def load_scene(name: str) -> dict:
    """Carrega uma cena preparada: DN, rótulos de referência e georreferência.

    Retorna dict com: dn (H, W, B), ref (H, W) com o código WorldCover,
    bands, sensor, transform, crs, res, cmask (só CBERS).
    """
    if name == "s2":
        stack, refp, sensor = prepare_s2_scene(), C.S2_REF, "sentinel2"
        prepare_reference(stack, refp)
    elif name == "cbers":
        stack, refp, sensor = prepare_cbers_scene(), C.CBERS_REF, "cbers4_mux"
        prepare_reference(stack, refp)
    else:
        raise ValueError(f"cena desconhecida: {name}")
    with rasterio.open(stack) as src:
        arr = src.read()
        meta = dict(transform=src.transform, crs=str(src.crs), res=src.res[0],
                    bands=list(src.descriptions))
    with rasterio.open(refp) as src:
        ref = src.read(1)
    out = dict(name=name, sensor=sensor, ref=ref, **meta)
    if name == "cbers":
        out["cmask"] = arr[-1].astype(np.uint8)
        out["dn"] = np.transpose(arr[:-1], (1, 2, 0))
        out["bands"] = meta["bands"][:-1]
    else:
        out["dn"] = np.transpose(arr, (1, 2, 0))
    return out


def scene_info() -> dict:
    """Metadados documentados das cenas (para README, API e slides)."""
    return {
        "s2": {"satellite": "Sentinel-2B", "sensor": "MSI", "level": "L1C (TOA)",
               "date": C.S2_DATE, "item": C.S2_ITEM, "collection": C.S2_COLLECTION,
               "bands": 13, "resolution_m": 10, "provider": "INPE — Brazil Data Cube"},
        "cbers": {"satellite": "CBERS-4", "sensor": "MUX", "level": "L4 SR",
                  "date": C.CBERS_DATE, "item": C.CBERS_ITEM, "collection": C.CBERS_COLLECTION,
                  "bands": 4, "resolution_m": 20, "provider": "INPE — Brazil Data Cube"},
    }


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    print("=" * 64)
    print("  Aquisição dos dados — Trabalho 10 PDI")
    print("=" * 64)
    if what in ("all", "eurosat"):
        print("\n[EuroSAT MS]")
        ensure_eurosat_ms()
    if what in ("all", "inpe"):
        print("\n[INPE — Sentinel-2 L1C]")
        prepare_reference(prepare_s2_scene(), C.S2_REF)
        print("\n[INPE — CBERS-4 MUX]")
        prepare_reference(prepare_cbers_scene(), C.CBERS_REF)
        with open(C.DATA_PROCESSED / "scenes.json", "w") as f:
            json.dump(scene_info(), f, indent=2, ensure_ascii=False)
    print("\n✓ dados prontos")
