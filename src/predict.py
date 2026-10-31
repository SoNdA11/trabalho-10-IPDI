"""
predict.py — Inferência e geração de mapas de classificação

Funções:
  - classify_image      : classifica uma imagem individual
  - classify_batch      : classifica lote de imagens
  - generate_class_map  : gera mapa colorido de classificação
  - classify_from_path  : carrega e classifica imagem a partir do caminho
"""

import json
import base64
import io
import numpy as np
import joblib
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import sys
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import CLASS_NAMES, CLASS_COLORS, MACRO_CLASSES, MACRO_COLORS
from preprocessing import preprocess_dataset, normalize_minmax
from feature_extraction import extract_features

# ─── Caminhos ─────────────────────────────────────────────────────────────────

ROOT    = Path(__file__).parent.parent
MODELS  = ROOT / "models"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
PREDS   = RESULTS / "predictions"

MODEL_PATH = MODELS / "random_forest.joblib"
METRICS_PATH = ROOT / "results" / "metrics" / "metrics.json"


# ─── Carregamento do modelo ───────────────────────────────────────────────────

_model = None
_metrics = None


def load_model():
    """Carrega o modelo Random Forest salvo em disco (singleton)."""
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Modelo não encontrado: {MODEL_PATH}\n"
                "Execute primeiro: python src/train.py"
            )
        _model = joblib.load(MODEL_PATH)
    return _model


def load_metrics() -> dict:
    """Carrega métricas salvas em disco."""
    global _metrics
    if _metrics is None:
        if not METRICS_PATH.exists():
            return {}
        with open(METRICS_PATH) as f:
            _metrics = json.load(f)
    return _metrics


# ─── Classificação ────────────────────────────────────────────────────────────

def classify_image(img: np.ndarray) -> dict:
    """
    Classifica uma imagem (H, W, 3) ou (H, W, C).

    Parâmetros
    ----------
    img : ndarray — imagem normalizada em [0, 1]

    Retorna
    -------
    dict com:
        class_index    : int — índice previsto
        class_name     : str — nome da classe
        macro_class    : str — macro-categoria
        probabilities  : dict {class_name: prob}
        color          : str — cor hex da classe
    """
    clf = load_model()
    metrics = load_metrics()
    active_classes = metrics.get("class_names", CLASS_NAMES)

    # Garante 3 canais (RGB)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    if img.shape[-1] > 3:
        img = img[..., :3]

    from preprocessing import fix_invalid_values
    img_norm = fix_invalid_values(img.astype(np.float32))
    fv = extract_features(img_norm).reshape(1, -1)

    pred_idx = int(clf.predict(fv)[0])
    probas = clf.predict_proba(fv)[0]

    # Mapeia índice de treino → nome de classe
    pred_name = CLASS_NAMES[pred_idx] if pred_idx < len(CLASS_NAMES) else "Desconhecido"

    # Probas por classe ativa (as usadas no treino)
    proba_dict = {}
    for i, cls in enumerate(active_classes):
        proba_dict[cls] = float(probas[i]) if i < len(probas) else 0.0

    return {
        "class_index": pred_idx,
        "class_name": pred_name,
        "macro_class": MACRO_CLASSES.get(pred_name, "Desconhecido"),
        "probabilities": proba_dict,
        "color": CLASS_COLORS.get(pred_name, "#95A5A6"),
    }


def classify_batch(X: np.ndarray) -> list[dict]:
    """Classifica lote de imagens."""
    return [classify_image(img) for img in X]


def classify_from_path(img_path: str | Path, img_size: int = 64) -> dict:
    """
    Carrega imagem do disco e classifica.

    Parâmetros
    ----------
    img_path : str ou Path — caminho para imagem PNG/JPG
    img_size : tamanho para redimensionamento

    Retorna
    -------
    dict com resultado da classificação + imagem base64
    """
    pil = Image.open(img_path).convert("RGB")
    if pil.size != (img_size, img_size):
        pil = pil.resize((img_size, img_size), Image.LANCZOS)

    img = np.array(pil).astype(np.float32) / 255.0
    result = classify_image(img)

    # Codifica imagem em base64 para a API
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    result["image_b64"] = base64.b64encode(buf.getvalue()).decode()
    return result


def classify_from_bytes(file_bytes: bytes, img_size: int = 64) -> dict:
    """
    Classifica imagem a partir de bytes (upload HTTP).

    Parâmetros
    ----------
    file_bytes : bytes — conteúdo do arquivo de imagem
    img_size   : tamanho para redimensionamento

    Retorna
    -------
    dict com resultado da classificação + imagem base64
    """
    pil = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    if pil.size != (img_size, img_size):
        pil = pil.resize((img_size, img_size), Image.LANCZOS)

    img = np.array(pil).astype(np.float32) / 255.0
    result = classify_image(img)

    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    result["image_b64"] = base64.b64encode(buf.getvalue()).decode()
    return result


# ─── Mapa de classificação ────────────────────────────────────────────────────

def generate_class_map(
    X: np.ndarray,
    y_true: np.ndarray | None = None,
    n_cols: int = 15,
    save_path: Path | None = None,
) -> np.ndarray:
    """
    Gera mapa visual de classificação para uma grade de imagens.

    Para cada imagem, pinta um quadrado com a cor da classe prevista.
    Se y_true for fornecido, marca erros com borda vermelha.

    Parâmetros
    ----------
    X       : ndarray (N, H, W, 3)
    y_true  : ndarray (N,) — rótulos verdadeiros (opcional)
    n_cols  : colunas na grade
    save_path : onde salvar a figura

    Retorna
    -------
    ndarray (H_grid, W_grid, 3) — imagem RGB do mapa
    """
    clf = load_model()
    metrics = load_metrics()
    active_classes = metrics.get("class_names", CLASS_NAMES)

    # Features
    from feature_extraction import extract_features_batch
    F = extract_features_batch(X, verbose=False)
    y_pred = clf.predict(F)

    n = len(X)
    img_h, img_w = X.shape[1], X.shape[2]
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.2, n_rows * 1.5))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for row in range(n_rows):
        for col in range(n_cols):
            idx = row * n_cols + col
            ax = axes[row, col]
            ax.axis("off")
            if idx >= n:
                continue

            pred_name = CLASS_NAMES[y_pred[idx]] if y_pred[idx] < len(CLASS_NAMES) else "?"
            color = CLASS_COLORS.get(pred_name, "#95A5A6")

            ax.imshow(X[idx])
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(3)
                if y_true is not None and y_true[idx] != y_pred[idx]:
                    spine.set_edgecolor("#E74C3C")  # erro = vermelho
                else:
                    spine.set_edgecolor(color)

    # Legenda
    legend_elements = [
        mpatches.Patch(facecolor=CLASS_COLORS.get(cls, "#95A5A6"), label=cls)
        for cls in active_classes
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=min(5, len(active_classes)),
               fontsize=8, framealpha=0.9, bbox_to_anchor=(0.5, -0.02))

    title = "Mapa de Classificação — Random Forest"
    if y_true is not None:
        n_errors = int(np.sum(y_true != y_pred))
        title += f" (erros: {n_errors}/{n})"
    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0.05, 1, 0.97])

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"✓ Mapa salvo: {save_path}")

    # Converte fig para array
    fig.canvas.draw()
    w_fig, h_fig = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf = buf.reshape(h_fig, w_fig, 3)
    plt.close()
    return buf


def generate_thematic_map(
    class_names_sample: list[str],
    save_path: Path | None = None,
) -> None:
    """
    Gera mapa temático compacto por macro-classe com paleta de cores.
    Simula um mapa de cobertura territorial.
    """
    PREDS.mkdir(parents=True, exist_ok=True)

    from data_loader import load_samples_by_class, EUROSAT_RGB_DIR
    if not EUROSAT_RGB_DIR.exists():
        print("  ⚠ Dataset não encontrado. Execute data_loader.py primeiro.")
        return

    samples = load_samples_by_class(n_per_class=8)

    fig, axes = plt.subplots(4, 8, figsize=(18, 9))
    fig.patch.set_facecolor("#1A1A2E")
    fig.suptitle("Mapa Temático de Cobertura do Solo — EuroSAT", 
                 fontsize=15, fontweight="bold", color="white", y=0.98)

    classes_to_show = [c for c in CLASS_NAMES if c in samples]
    colors = [CLASS_COLORS.get(c, "#95A5A6") for c in classes_to_show]

    row_idx = 0
    for cls in classes_to_show:
        if row_idx >= 4:
            break
        imgs = samples[cls]
        for col_idx, img in enumerate(imgs[:8]):
            ax = axes[row_idx, col_idx]
            ax.imshow(img)
            ax.axis("off")
            # Borda colorida por classe
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(2)
                spine.set_edgecolor(CLASS_COLORS.get(cls, "#fff"))
        row_idx += 1

    # Esconde eixos vazios
    for r in range(row_idx, 4):
        for c in range(8):
            axes[r, c].axis("off")

    # Legenda
    legend_elements = [
        mpatches.Patch(facecolor=CLASS_COLORS.get(cls, "#95A5A6"), label=cls, edgecolor="white")
        for cls in classes_to_show
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=5,
               fontsize=9, framealpha=0.8, facecolor="#2D2D44", 
               labelcolor="white", edgecolor="#555",
               bbox_to_anchor=(0.5, 0.0))

    plt.tight_layout(rect=[0, 0.06, 1, 0.97])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#1A1A2E")
        print(f"✓ Mapa temático salvo: {save_path}")
    plt.close()


# ─── Execução direta ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("predict.py — testando carregamento do modelo e classificação")
    try:
        metrics = load_metrics()
        print(f"  Modelo: {metrics.get('model', '?')}")
        print(f"  Acurácia (teste): {metrics.get('test_accuracy', '?'):.1%}")
        print(f"  Classes: {metrics.get('class_names', [])}")
    except Exception as e:
        print(f"  ⚠ {e}")

    # Gera mapa temático
    print("\nGerando mapa temático...")
    PREDS.mkdir(parents=True, exist_ok=True)
    generate_thematic_map(CLASS_NAMES, save_path=PREDS / "thematic_map.png")
    print("✓ predict.py OK")
