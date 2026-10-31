"""
evaluation.py — Avaliação completa do classificador e geração de figuras

Gera todas as visualizações para a apresentação:
  1. Matriz de confusão (normalizada e por contagem)
  2. Métricas por classe
  3. Importância das features
  4. Exemplos de classificações corretas e incorretas
  5. Distribuição de probabilidades previstas
  6. Mapa de classificação de uma cena completa
  7. Comparação imagem original vs. classificada
"""

import json
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
)
from sklearn.preprocessing import label_binarize

import sys
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import (
    load_processed,
    download_eurosat_rgb,
    load_images_rgb,
    CLASS_NAMES,
    CLASS_COLORS,
    MACRO_COLORS,
)
from preprocessing import preprocess_dataset
from feature_extraction import extract_features_batch, build_feature_names
from predict import load_model, load_metrics

# ─── Caminhos ─────────────────────────────────────────────────────────────────

ROOT    = Path(__file__).parent.parent
FIGURES = ROOT / "results" / "figures"
METRICS_DIR = ROOT / "results" / "metrics"
PREDS   = ROOT / "results" / "predictions"
MODELS  = ROOT / "models"

for d in [FIGURES, METRICS_DIR, PREDS]:
    d.mkdir(parents=True, exist_ok=True)


# ─── Análise de erros ─────────────────────────────────────────────────────────

def find_error_examples(
    X: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    n_per_pair: int = 3,
) -> dict:
    """
    Encontra exemplos de classificações incorretas por par de classes.

    Retorna
    -------
    dict {(true_class, pred_class): [indices]}
    """
    errors = {}
    for i in range(len(y_true)):
        if y_true[i] != y_pred[i]:
            key = (int(y_true[i]), int(y_pred[i]))
            errors.setdefault(key, []).append(i)

    # Mantém apenas os pares com mais erros
    sorted_pairs = sorted(errors.items(), key=lambda x: -len(x[1]))
    return {k: v[:n_per_pair] for k, v in sorted_pairs[:8]}


def plot_error_analysis(
    X: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    save_path: Path | None = None,
) -> None:
    """
    Plota exemplos de classificações incorretas com destaque visual.
    """
    error_pairs = find_error_examples(X, y_true, y_pred, class_names)
    if not error_pairs:
        print("  Nenhum erro encontrado no subconjunto visualizado.")
        return

    n_pairs = min(6, len(error_pairs))
    n_examples = 3
    fig, axes = plt.subplots(n_pairs, n_examples, figsize=(n_examples * 2.5, n_pairs * 2.8))
    if n_pairs == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle("Análise de Erros — Classificações Incorretas", 
                 fontsize=13, fontweight="bold")

    for row, ((true_idx, pred_idx), indices) in enumerate(list(error_pairs.items())[:n_pairs]):
        true_name = class_names[true_idx] if true_idx < len(class_names) else str(true_idx)
        pred_name = class_names[pred_idx] if pred_idx < len(class_names) else str(pred_idx)
        true_color = CLASS_COLORS.get(true_name, "#2ECC71")
        pred_color = CLASS_COLORS.get(pred_name, "#E74C3C")

        for col in range(n_examples):
            ax = axes[row, col]
            if col < len(indices):
                ax.imshow(X[indices[col]])
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_linewidth(3)
                    spine.set_edgecolor("#E74C3C")
                if col == 0:
                    ax.set_ylabel(
                        f"Real: {true_name}\n→ Previsto: {pred_name}",
                        fontsize=7.5, color="#C0392B", labelpad=5
                    )
            else:
                ax.set_facecolor("#F8F8F8")
            ax.axis("off")
            ax.tick_params(left=False, bottom=False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  ✓ Erro analysis salva: {save_path}")
    plt.close()


def plot_confidence_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probas: np.ndarray,
    save_path: Path | None = None,
) -> None:
    """
    Plota distribuição de confiança do modelo para predições corretas e incorretas.

    Classifique bem com alta confiança = bom modelo.
    """
    max_proba = probas.max(axis=1)
    correct_mask = (y_true == y_pred)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(max_proba[correct_mask], bins=25, alpha=0.75, color="#2ECC71",
            label=f"Corretas (n={correct_mask.sum()})", density=True)
    ax.hist(max_proba[~correct_mask], bins=25, alpha=0.75, color="#E74C3C",
            label=f"Incorretas (n={(~correct_mask).sum()})", density=True)

    ax.axvline(0.5, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Confiança máxima da predição")
    ax.set_ylabel("Densidade")
    ax.set_title("Distribuição de Confiança — Corretas vs. Incorretas", fontweight="bold")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  ✓ Figura salva: {save_path}")
    plt.close()


def plot_comparison_grid(
    X: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    n_samples: int = 30,
    save_path: Path | None = None,
) -> None:
    """
    Grade de imagens mostrando imagem original colorida pela classe prevista.
    Verde = correto, Vermelho = incorreto.
    """
    rng = np.random.default_rng(0)
    indices = rng.choice(len(X), min(n_samples, len(X)), replace=False)

    n_cols = 10
    n_rows = (len(indices) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.4, n_rows * 1.8))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle("Resultados — Grade de Classificação (verde=correto, vermelho=erro)",
                 fontsize=11, fontweight="bold")

    for i, idx in enumerate(indices):
        row, col = divmod(i, n_cols)
        ax = axes[row, col]
        ax.imshow(X[idx])
        ax.axis("off")

        correct = y_true[idx] == y_pred[idx]
        pred_name = class_names[y_pred[idx]] if y_pred[idx] < len(class_names) else "?"
        border_color = "#2ECC71" if correct else "#E74C3C"

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(3)
            spine.set_edgecolor(border_color)

        ax.set_xlabel(pred_name[:10], fontsize=6.5, labelpad=2)
        ax.xaxis.label.set_color(CLASS_COLORS.get(pred_name, "#666"))

    for i in range(len(indices), n_rows * n_cols):
        row, col = divmod(i, n_cols)
        axes[row, col].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  ✓ Grade salva: {save_path}")
    plt.close()


def plot_spectral_signatures(
    X: np.ndarray,
    y: np.ndarray,
    class_names: list[str],
    save_path: Path | None = None,
) -> None:
    """
    Plota assinaturas espectrais médias por classe (média dos canais RGB).

    Assinaturas espectrais são "impressões digitais" de cada tipo de
    cobertura — vegetação absorve vermelho e reflete verde/NIR,
    água absorve vermelho e azul, solo reflete fortemente no vermelho.
    """
    band_names = ["Red (B4)", "Green (B3)", "Blue (B2)"]
    fig, ax = plt.subplots(figsize=(10, 6))

    for cls_idx, cls_name in enumerate(class_names):
        mask = y == cls_idx
        if mask.sum() == 0:
            continue
        samples = X[mask]  # (N, H, W, 3)
        # Média por canal sobre todos os pixels e todas as amostras
        means = samples.reshape(-1, 3).mean(axis=0)
        stds  = samples.reshape(-1, 3).std(axis=0)
        color = CLASS_COLORS.get(cls_name, "#95A5A6")

        ax.plot(range(3), means, "o-", color=color, label=cls_name, linewidth=2, markersize=6)
        ax.fill_between(range(3),
                        means - stds, means + stds,
                        color=color, alpha=0.12)

    ax.set_xticks(range(3))
    ax.set_xticklabels(band_names, fontsize=11)
    ax.set_ylabel("Reflectância média normalizada", fontsize=11)
    ax.set_title("Assinaturas Espectrais por Classe — EuroSAT", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.85)
    ax.set_ylim(0, 1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  ✓ Assinaturas espectrais salvas: {save_path}")
    plt.close()


def plot_ndvi_distributions(
    X: np.ndarray,
    y: np.ndarray,
    class_names: list[str],
    save_path: Path | None = None,
) -> None:
    """
    Distribuição do NDVI aproximado por classe.
    """
    from feature_extraction import ndvi_approx

    fig, ax = plt.subplots(figsize=(11, 6))
    for cls_idx, cls_name in enumerate(class_names):
        mask = y == cls_idx
        if mask.sum() == 0:
            continue
        ndvi_vals = np.array([ndvi_approx(X[i]).mean() for i in np.where(mask)[0]])
        color = CLASS_COLORS.get(cls_name, "#95A5A6")
        ax.hist(ndvi_vals, bins=30, alpha=0.6, color=color, label=cls_name, density=True)

    ax.set_xlabel("NDVI médio por imagem (aproximado)")
    ax.set_ylabel("Densidade")
    ax.set_title("Distribuição do NDVI Aproximado por Classe", fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  ✓ NDVI distributions salvas: {save_path}")
    plt.close()


# ─── Pipeline completo ────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 65)
    print("  Avaliação Completa — Trabalho 10 PDI")
    print("=" * 65)

    # Carrega dados
    print("\n[1/4] Carregando dados e modelo...")
    download_eurosat_rgb()
    X, y, found_classes = load_images_rgb(max_per_class=500)
    X = preprocess_dataset(X)

    cached = load_processed("features")
    if cached is not None:
        F, y = cached
    else:
        F = extract_features_batch(X, verbose=True)

    clf = load_model()
    metrics = load_metrics()
    active_classes = metrics.get("class_names", found_classes)

    print(f"  Classes: {active_classes}")
    present = sorted(set(int(v) for v in y))

    # Re-faz predições no conjunto completo para análise
    print("\n[2/4] Gerando predições...")
    y_pred = clf.predict(F)
    probas = clf.predict_proba(F)

    print(f"  Acurácia total: {accuracy_score(y, y_pred):.4f}")

    # ── Figura: assinaturas espectrais ─────────────────────────────────────────
    print("\n[3/4] Gerando figuras...")
    plot_spectral_signatures(X, y, active_classes, FIGURES / "spectral_signatures.png")
    plot_ndvi_distributions(X, y, active_classes, FIGURES / "ndvi_distributions.png")
    plot_error_analysis(X, y, y_pred, active_classes, FIGURES / "error_analysis.png")
    plot_confidence_distribution(y, y_pred, probas, FIGURES / "confidence_distribution.png")
    plot_comparison_grid(X, y, y_pred, active_classes, save_path=FIGURES / "classification_grid.png")

    # ── Mapa temático ──────────────────────────────────────────────────────────
    print("\n[4/4] Gerando mapa temático...")
    from predict import generate_thematic_map
    generate_thematic_map(active_classes, save_path=PREDS / "thematic_map.png")

    print("\n✓ Avaliação concluída. Todas as figuras em results/figures/")
    print("  Arquivos gerados:")
    for f in sorted(FIGURES.glob("*.png")):
        print(f"    {f.name}")


if __name__ == "__main__":
    main()
