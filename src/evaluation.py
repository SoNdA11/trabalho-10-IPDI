"""
evaluation.py — Etapa 7: avaliação (métricas) e figuras de resultado.

Todas as métricas e figuras de erro usam APENAS o conjunto de teste, que o
modelo nunca viu (nem para escolher hiperparâmetros).

Uso:
    python src/evaluation.py      # gera results/figures/*.png a partir dos modelos treinados
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import (accuracy_score, cohen_kappa_score, confusion_matrix,
                             precision_recall_fscore_support)

sys.path.insert(0, str(Path(__file__).parent))

import config as C  # noqa: E402


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> dict:
    """Acurácia, kappa, precisão/recall/F1 por classe e médias, matriz de confusão."""
    labels = list(range(len(class_names)))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    pm, rm, fm, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels,
                                                    average="macro", zero_division=0)
    _, _, fw, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels,
                                                  average="weighted", zero_division=0)
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "kappa": float(cohen_kappa_score(y_true, y_pred, labels=labels)),
        "macro_precision": float(pm),
        "macro_recall": float(rm),
        "macro_f1": float(fm),
        "weighted_f1": float(fw),
        "per_class": {
            c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]),
                "support": int(s[i])}
            for i, c in enumerate(class_names)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "labels": class_names,
    }


def map_to_macro(y: np.ndarray) -> np.ndarray:
    """Rótulos EuroSAT (0–9) → índice da macro-classe em LANDCOVER_CLASSES."""
    lut = np.array([C.LANDCOVER_CLASSES.index(C.EUROSAT_TO_MACRO[c]) for c in C.EUROSAT_CLASSES])
    return lut[y]


def load_metrics(name: str) -> dict:
    p = C.METRICS / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else {}


if __name__ == "__main__":
    import figures
    figures.main()
