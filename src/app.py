"""
app.py — Backend Flask para a interface de demonstração

Endpoints:
  GET  /                          → Serve a interface web
  POST /api/classify              → Classifica imagem enviada
  GET  /api/metrics               → Retorna métricas do modelo
  GET  /api/samples               → Retorna amostras por classe em base64
  GET  /api/figures               → Lista figuras disponíveis
  GET  /api/figures/<name>        → Retorna figura em base64
  GET  /api/classes               → Retorna lista de classes e cores

Uso:
  python src/app.py
  Acesse: http://localhost:5000
"""

import json
import base64
import io
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("trab10")
from pathlib import Path

from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import numpy as np
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import CLASS_NAMES, CLASS_COLORS, MACRO_CLASSES, MACRO_COLORS, EUROSAT_RGB_DIR
from predict import classify_from_bytes, load_model, load_metrics

# ─── Configuração ─────────────────────────────────────────────────────────────

ROOT        = Path(__file__).parent.parent
STATIC_DIR  = ROOT / "presentation"
FIGURES_DIR = ROOT / "results" / "figures"
PREDS_DIR   = ROOT / "results" / "predictions"
METRICS_PATH = ROOT / "results" / "metrics" / "metrics.json"

app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="")
CORS(app)


# ─── Utilitários ──────────────────────────────────────────────────────────────

def _img_to_b64(pil_img: Image.Image, fmt: str = "PNG") -> str:
    """Converte imagem PIL para string base64."""
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


def _file_to_b64(path: Path) -> str:
    """Lê arquivo e retorna base64."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


# ─── Rotas ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve a interface de demonstração principal."""
    return send_from_directory(str(STATIC_DIR), "index.html")


@app.route("/api/classify", methods=["POST"])
def api_classify():
    """
    Classifica uma imagem enviada como multipart/form-data.

    Body: file=<image>
    Retorna:
      {
        class_name: str,
        macro_class: str,
        color: str,
        probabilities: {class: float},
        image_b64: str (PNG base64)
      }
    """
    if "file" not in request.files:
        return jsonify({"error": "Nenhum arquivo enviado. Use o campo 'file'."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Nome de arquivo vazio."}), 400

    # Validação de formato
    allowed = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext and ext not in allowed:
        return jsonify({"error": f"Formato não suportado: {ext}"}), 400

    try:
        file_bytes = file.read()
        logger.info("classify request: %s (%d bytes)", file.filename, len(file_bytes))
        if len(file_bytes) == 0:
            return jsonify({"error": "Arquivo vazio."}), 400
        result = classify_from_bytes(file_bytes, img_size=64)
        return jsonify(result)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        return jsonify({"error": f"Erro ao classificar: {e}"}), 500


@app.route("/api/classify_batch", methods=["POST"])
def api_classify_batch():
    """
    Classifica múltiplas imagens enviadas como multipart/form-data.
    Campo: files (múltiplos).
    """
    if "files" not in request.files and "file" not in request.files:
        return jsonify({"error": "Nenhum arquivo enviado. Use o campo 'files'."}), 400

    files = request.files.getlist("files") or request.files.getlist("file")
    if not files:
        return jsonify({"error": "Lista de arquivos vazia."}), 400

    results = []
    for f in files:
        if not f or f.filename == "":
            results.append({"error": "Arquivo inválido"})
            continue
        try:
            fb = f.read()
            r = classify_from_bytes(fb, img_size=64)
            r["filename"] = f.filename
            results.append(r)
        except Exception as e:
            results.append({"filename": f.filename, "error": str(e)})

    return jsonify({"results": results, "count": len(results)})


@app.route("/api/classify_sample/<class_name>/<int:sample_idx>", methods=["GET"])
def api_classify_sample(class_name: str, sample_idx: int):
    """
    Classifica uma das imagens de exemplo do dataset.

    Parâmetros:
      class_name : nome da classe EuroSAT
      sample_idx : índice da imagem (0-4)
    """
    class_dir = EUROSAT_RGB_DIR / class_name
    if not class_dir.exists():
        return jsonify({"error": f"Classe não encontrada: {class_name}"}), 404

    files = sorted(class_dir.glob("*.jpg"))
    if sample_idx >= len(files):
        return jsonify({"error": f"Índice fora do range. Max: {len(files)-1}"}), 400

    try:
        result = classify_from_bytes(files[sample_idx].read_bytes(), img_size=64)
        result["true_class"] = class_name
        result["is_correct"] = result["class_name"] == class_name
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/metrics", methods=["GET"])
def api_metrics():
    """Retorna métricas do modelo treinado."""
    if not METRICS_PATH.exists():
        return jsonify({"error": "Modelo não treinado. Execute: python src/train.py"}), 503
    with open(METRICS_PATH) as f:
        return jsonify(json.load(f))


@app.route("/api/classes", methods=["GET"])
def api_classes():
    """Retorna lista de classes com cores e macro-categorias."""
    metrics = load_metrics()
    class_names = metrics.get("class_names", CLASS_NAMES)
    return jsonify({
        "classes": [
            {
                "name": cls,
                "color": CLASS_COLORS.get(cls, "#95A5A6"),
                "macro": MACRO_CLASSES.get(cls, "Outro"),
            }
            for cls in class_names
        ],
        "macro_classes": [
            {"name": m, "color": MACRO_COLORS.get(m, "#95A5A6")}
            for m in sorted(set(MACRO_CLASSES.values()))
        ],
    })


@app.route("/api/samples", methods=["GET"])
def api_samples():
    """
    Retorna amostras de imagens por classe em base64.

    Query params:
      n_per_class: int (padrão: 5)
    """
    n = int(request.args.get("n_per_class", 5))

    if not EUROSAT_RGB_DIR.exists():
        return jsonify({"error": "Dataset não encontrado. Execute: python src/data_loader.py"}), 503

    metrics = load_metrics()
    class_names = metrics.get("class_names", CLASS_NAMES)

    result = {}
    import numpy as np
    rng = np.random.default_rng(42)

    for cls in class_names:
        class_dir = EUROSAT_RGB_DIR / cls
        if not class_dir.exists():
            continue
        files = sorted(class_dir.glob("*.jpg"))
        if not files:
            continue
        selected = rng.choice(len(files), min(n, len(files)), replace=False)
        imgs = []
        for idx in selected:
            pil = Image.open(files[idx]).convert("RGB").resize((64, 64))
            imgs.append({
                "b64": _img_to_b64(pil),
                "filename": files[idx].name,
            })
        result[cls] = {
            "images": imgs,
            "color": CLASS_COLORS.get(cls, "#95A5A6"),
            "macro": MACRO_CLASSES.get(cls, "Outro"),
        }

    return jsonify(result)


@app.route("/api/figures", methods=["GET"])
def api_figures():
    """Lista todas as figuras geradas disponíveis."""
    if not FIGURES_DIR.exists():
        return jsonify({"figures": []})
    figs = [f.name for f in sorted(FIGURES_DIR.glob("*.png"))]
    preds = [f"predictions/{f.name}" for f in sorted(PREDS_DIR.glob("*.png"))] if PREDS_DIR.exists() else []
    return jsonify({"figures": figs + preds})


@app.route("/api/figures/<path:name>", methods=["GET"])
def api_figure(name: str):
    """Retorna figura específica como PNG."""
    if name.startswith("predictions/"):
        path = PREDS_DIR / name[len("predictions/"):]
    else:
        path = FIGURES_DIR / name

    if not path.exists():
        return jsonify({"error": f"Figura não encontrada: {name}"}), 404
    return send_file(str(path), mimetype="image/png")


@app.route("/api/figures/<path:name>/b64", methods=["GET"])
def api_figure_b64(name: str):
    """Retorna figura em base64."""
    if name.startswith("predictions/"):
        path = PREDS_DIR / name[len("predictions/"):]
    else:
        path = FIGURES_DIR / name

    if not path.exists():
        return jsonify({"error": f"Figura não encontrada: {name}"}), 404
    return jsonify({"b64": _file_to_b64(path), "name": name})


@app.route("/api/status", methods=["GET"])
def api_status():
    """Verifica status do sistema."""
    model_ready = (ROOT / "models" / "random_forest.joblib").exists()
    dataset_ready = EUROSAT_RGB_DIR.exists()
    metrics_ready = METRICS_PATH.exists()

    acc = None
    if metrics_ready:
        with open(METRICS_PATH) as f:
            m = json.load(f)
            acc = m.get("test_accuracy")

    return jsonify({
        "model_ready": model_ready,
        "dataset_ready": dataset_ready,
        "metrics_ready": metrics_ready,
        "test_accuracy": acc,
        "figures_count": len(list(FIGURES_DIR.glob("*.png"))) if FIGURES_DIR.exists() else 0,
    })


# ─── Inicialização ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  Satellite Classifier API — Trabalho 10 PDI")
    print("=" * 55)
    print(f"  Interface: http://localhost:5000")
    print(f"  API docs:  http://localhost:5000/api/status")
    print()

    # Pré-carrega modelo
    try:
        load_model()
        m = load_metrics()
        print(f"  ✓ Modelo carregado — Acurácia: {m.get('test_accuracy', '?'):.1%}")
    except FileNotFoundError:
        print("  ⚠ Modelo não encontrado. Execute: python src/train.py")

    app.run(debug=False, port=5000, host="0.0.0.0")
