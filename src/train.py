"""
train.py — Treinamento do classificador Random Forest

Pipeline completo:
  1. Carrega e prepara o dataset EuroSAT
  2. Extrai features de cada imagem
  3. Divide em treino (70%), validação (15%), teste (15%)
  4. Treina Random Forest com otimização de hiperparâmetros
  5. Avalia no conjunto de teste
  6. Salva modelo, métricas e figuras
"""

import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# Importações locais
import sys
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import (
    download_eurosat_rgb,
    load_images_rgb,
    load_processed,
    save_processed,
    CLASS_NAMES,
    CLASS_COLORS,
    MACRO_CLASSES,
    MACRO_CLASS_NAMES,
    MACRO_COLORS,
)
from preprocessing import preprocess_dataset
from feature_extraction import (
    extract_features_batch,
    build_feature_names,
    plot_feature_importance,
)

# ─── Caminhos ─────────────────────────────────────────────────────────────────

ROOT     = Path(__file__).parent.parent
MODELS   = ROOT / "models"
RESULTS  = ROOT / "results"
FIGURES  = RESULTS / "figures"
METRICS  = RESULTS / "metrics"

for d in [MODELS, FIGURES, METRICS]:
    d.mkdir(parents=True, exist_ok=True)


# ─── Utilitários de visualização ──────────────────────────────────────────────

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    title: str = "Matriz de Confusão",
    save_path: Path | None = None,
    normalize: bool = True,
) -> None:
    """Plota e salva a matriz de confusão."""
    if normalize:
        cm_plot = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)
        fmt = ".2f"
        vmax = 1.0
    else:
        cm_plot = cm
        fmt = "d"
        vmax = cm.max()

    fig, ax = plt.subplots(figsize=(max(8, len(class_names)), max(7, len(class_names) - 1)))
    sns.heatmap(
        cm_plot,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        vmin=0,
        vmax=vmax,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_xlabel("Classe Prevista", fontsize=12)
    ax.set_ylabel("Classe Real", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    plt.xticks(rotation=35, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  ✓ Figura salva: {save_path}")
    plt.close()


def plot_class_distribution(y: np.ndarray, class_names: list[str], save_path: Path | None = None) -> None:
    """Plota distribuição das classes no dataset."""
    counts = [(class_names[i], np.sum(y == i)) for i in range(len(class_names))]
    counts.sort(key=lambda x: -x[1])
    names, vals = zip(*counts)

    colors = [CLASS_COLORS.get(n, "#95A5A6") for n in names]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(range(len(names)), vals, color=colors, alpha=0.9, edgecolor="white", linewidth=0.8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("Número de amostras")
    ax.set_title("Distribuição das Classes — EuroSAT", fontsize=13, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)

    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20, str(val),
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  ✓ Figura salva: {save_path}")
    plt.close()


def plot_sample_images(X: np.ndarray, y: np.ndarray, class_names: list[str], save_path: Path | None = None) -> None:
    """Plota grade de amostras por classe."""
    n_classes = len(class_names)
    n_samples = 5
    fig, axes = plt.subplots(n_classes, n_samples, figsize=(n_samples * 1.8, n_classes * 1.8))
    fig.suptitle("Amostras do Dataset EuroSAT — Uma por Classe", fontsize=13, fontweight="bold")

    rng = np.random.default_rng(42)
    for row, cls_name in enumerate(class_names):
        cls_idx = CLASS_NAMES.index(cls_name) if cls_name in CLASS_NAMES else row
        idxs = np.where(y == cls_idx)[0]
        selected = rng.choice(idxs, min(n_samples, len(idxs)), replace=False) if len(idxs) > 0 else []

        for col in range(n_samples):
            ax = axes[row, col]
            if col < len(selected):
                img = X[selected[col]]
                ax.imshow(img)
            else:
                ax.set_facecolor("#EEEEEE")
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(cls_name, fontsize=8, labelpad=3, rotation=0, ha="right", va="center")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  ✓ Figura salva: {save_path}")
    plt.close()


def plot_metrics_bar(report_dict: dict, class_names: list[str], save_path: Path | None = None) -> None:
    """Plota métricas por classe em gráfico de barras agrupadas."""
    metrics = ["precision", "recall", "f1-score"]
    x = np.arange(len(class_names))
    width = 0.25
    colors = ["#3498DB", "#2ECC71", "#E74C3C"]

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        vals = [report_dict.get(cls, {}).get(metric, 0) for cls in class_names]
        bars = ax.bar(x + i * width, vals, width, label=metric.capitalize(), color=color, alpha=0.85)

    ax.set_xlabel("Classe", fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Métricas por Classe — Random Forest", fontsize=13, fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels(class_names, rotation=35, ha="right", fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=10)
    ax.axhline(y=0.9, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  ✓ Figura salva: {save_path}")
    plt.close()


# ─── Pipeline principal ───────────────────────────────────────────────────────

def main() -> None:
    print("=" * 65)
    print("  Treinamento — Random Forest — Trabalho 10 PDI")
    print("=" * 65)

    # ── 1. Dataset ─────────────────────────────────────────────────────────────
    print("\n[1/6] Carregando dataset...")
    download_eurosat_rgb()

    # Tenta carregar features pré-computadas
    cached = load_processed("features")
    if cached is not None:
        F, y = cached
        print("  Carregando imagens para visualizações...")
        X, y_raw, found_classes = load_images_rgb(max_per_class=500)
        X = preprocess_dataset(X)
    else:
        X, y_raw, found_classes = load_images_rgb(max_per_class=500)
        print(f"  Pré-processando {len(X)} imagens...")
        X = preprocess_dataset(X)
        y = y_raw

        # ── 2. Extração de features ────────────────────────────────────────────
        print("\n[2/6] Extraindo características...")
        F = extract_features_batch(X, verbose=True)
        save_processed(F, y, "features")

    # Classes presentes
    present_classes = sorted(set(int(i) for i in y))
    active_class_names = [CLASS_NAMES[i] for i in present_classes]
    print(f"  Classes ativas: {active_class_names}")

    # ── 3. Visualizações exploratórias ─────────────────────────────────────────
    print("\n[3/6] Gerando visualizações...")
    plot_class_distribution(y, active_class_names, FIGURES / "class_distribution.png")
    plot_sample_images(X, y, active_class_names, FIGURES / "sample_images.png")

    # ── 4. Divisão treino/val/teste ────────────────────────────────────────────
    print("\n[4/6] Dividindo dados...")
    # 70% treino, 15% val, 15% teste (estratificado)
    F_train, F_temp, y_train, y_temp = train_test_split(
        F, y, test_size=0.30, random_state=42, stratify=y
    )
    F_val, F_test, y_val, y_test = train_test_split(
        F_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )
    print(f"  Treino:    {len(F_train):5d} amostras")
    print(f"  Validação: {len(F_val):5d} amostras")
    print(f"  Teste:     {len(F_test):5d} amostras")

    # ── 5. Treinamento ─────────────────────────────────────────────────────────
    print("\n[5/6] Treinando Random Forest...")
    print("  Parâmetros: n_estimators=200, max_depth=None, min_samples_leaf=2")
    t0 = time.time()

    clf = RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=2,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42,
        class_weight="balanced",
        oob_score=True,
    )
    clf.fit(F_train, y_train)
    elapsed = time.time() - t0
    print(f"  Tempo de treino: {elapsed:.1f}s")
    print(f"  OOB Score:       {clf.oob_score_:.4f}")

    # Validação
    y_val_pred = clf.predict(F_val)
    val_acc = accuracy_score(y_val, y_val_pred)
    print(f"  Acurácia (val):  {val_acc:.4f}")

    # ── 6. Avaliação no conjunto de teste ──────────────────────────────────────
    print("\n[6/6] Avaliando no conjunto de teste...")
    y_test_pred = clf.predict(F_test)
    y_test_proba = clf.predict_proba(F_test)

    test_acc = accuracy_score(y_test, y_test_pred)
    cm = confusion_matrix(y_test, y_test_pred, labels=present_classes)
    report = classification_report(
        y_test, y_test_pred,
        labels=present_classes,
        target_names=active_class_names,
        output_dict=True,
    )
    report_str = classification_report(
        y_test, y_test_pred,
        labels=present_classes,
        target_names=active_class_names,
    )

    print(f"\n  Acurácia (teste): {test_acc:.4f}")
    print("\n" + report_str)

    # Salva modelo
    model_path = MODELS / "random_forest.joblib"
    joblib.dump(clf, model_path)
    print(f"  ✓ Modelo salvo: {model_path}")

    # Salva scaler (identidade — RF não precisa, mas mantemos interface uniforme)
    scaler_path = MODELS / "scaler.joblib"
    joblib.dump(None, scaler_path)

    # Salva métricas em JSON
    metrics_data = {
        "model": "RandomForestClassifier",
        "n_estimators": 200,
        "dataset": "EuroSAT RGB",
        "n_classes": len(active_class_names),
        "class_names": active_class_names,
        "n_train": len(F_train),
        "n_val": len(F_val),
        "n_test": len(F_test),
        "oob_score": float(clf.oob_score_),
        "val_accuracy": float(val_acc),
        "test_accuracy": float(test_acc),
        "macro_avg_precision": report["macro avg"]["precision"],
        "macro_avg_recall": report["macro avg"]["recall"],
        "macro_avg_f1": report["macro avg"]["f1-score"],
        "weighted_avg_f1": report["weighted avg"]["f1-score"],
        "per_class": {
            cls: {
                "precision": report[cls]["precision"],
                "recall": report[cls]["recall"],
                "f1": report[cls]["f1-score"],
                "support": int(report[cls]["support"]),
            }
            for cls in active_class_names
        },
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": active_class_names,
        "training_time_seconds": float(elapsed),
        "feature_dim": int(F.shape[1]),
    }
    metrics_path = METRICS / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_data, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Métricas salvas: {metrics_path}")

    # ── Figuras de resultados ──────────────────────────────────────────────────
    print("\n  Gerando figuras de resultados...")

    # Matriz de confusão normalizada
    plot_confusion_matrix(
        cm, active_class_names,
        title="Matriz de Confusão — Random Forest (Teste)",
        save_path=FIGURES / "confusion_matrix.png",
        normalize=True,
    )
    # Matriz de confusão com contagens
    plot_confusion_matrix(
        cm, active_class_names,
        title="Matriz de Confusão — Contagens (Teste)",
        save_path=FIGURES / "confusion_matrix_counts.png",
        normalize=False,
    )
    # Métricas por classe
    plot_metrics_bar(report, active_class_names, FIGURES / "metrics_by_class.png")

    # Importância das features
    feature_names = build_feature_names()
    plot_feature_importance(
        feature_names, clf.feature_importances_,
        top_n=20,
        save_path=FIGURES / "feature_importance.png",
    )

    print("\n" + "=" * 65)
    print(f"  RESULTADO FINAL — Acurácia no Teste: {test_acc:.1%}")
    print(f"  F1 (macro):   {report['macro avg']['f1-score']:.1%}")
    print(f"  F1 (weighted): {report['weighted avg']['f1-score']:.1%}")
    print("=" * 65)
    print("\n✓ Treinamento concluído. Resultados em results/")


if __name__ == "__main__":
    main()
