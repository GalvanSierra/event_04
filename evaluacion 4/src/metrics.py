"""Métricas y gráficas de evaluación para el problema APS."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

COST_FP = 10   # revisión innecesaria en taller
COST_FN = 500  # falla APS no detectada


def aps_cost(y_true, y_pred, cost_fp: int = COST_FP, cost_fn: int = COST_FN) -> int:
    """Costo oficial del reto: 10·FP + 500·FN."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return int(cost_fp * fp + cost_fn * fn)


def evaluate(y_true, proba, threshold: float = 0.5) -> dict:
    """Métricas de clasificación para un umbral dado."""
    y_pred = (np.asarray(proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "pr_auc": average_precision_score(y_true, proba),
        "roc_auc": roc_auc_score(y_true, proba),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "cost": int(COST_FP * fp + COST_FN * fn),
    }


def cost_curve(y_true, proba, thresholds=None):
    """Costo total para una grilla de umbrales."""
    if thresholds is None:
        thresholds = np.linspace(0.001, 0.999, 999)
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    costs = np.array([aps_cost(y_true, (proba >= t).astype(int)) for t in thresholds])
    return thresholds, costs


def best_threshold_by_cost(y_true, proba, thresholds=None):
    """Umbral que minimiza el costo (calcular con probabilidades OOF de train)."""
    thresholds, costs = cost_curve(y_true, proba, thresholds)
    i = int(np.argmin(costs))
    return float(thresholds[i]), int(costs[i])


def plot_pr_curves(curves: dict, ax=None, title="Curvas Precision-Recall"):
    """``curves``: {nombre: (y_true, proba)}."""
    ax = ax or plt.subplots(figsize=(6, 5))[1]
    for name, (y_true, proba) in curves.items():
        p, r, _ = precision_recall_curve(y_true, proba)
        ap = average_precision_score(y_true, proba)
        ax.plot(r, p, label=f"{name} (AP={ap:.3f})")
    base = np.mean(next(iter(curves.values()))[0])
    ax.axhline(base, ls="--", c="grey", lw=1, label=f"Azar ({base:.3f})")
    ax.set(xlabel="Recall", ylabel="Precision", title=title, xlim=(0, 1), ylim=(0, 1.02))
    ax.legend(loc="lower left", fontsize=8)
    return ax


def plot_confusion(y_true, y_pred, ax=None, title="Matriz de confusión"):
    ax = ax or plt.subplots(figsize=(4.5, 4))[1]
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=["neg", "pos"], cmap="Blues", ax=ax, colorbar=False
    )
    ax.set_title(title)
    return ax
