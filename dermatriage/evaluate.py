"""Evaluación clínica: sensibilidad por clase, errores críticos y subgrupos."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score

from .config import ALTO, BAJO, CONCORDANCE_TARGET, RISK_LEVELS


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float | None, float | None]:
    """Intervalo de confianza de Wilson al 95 % para una proporción k/n."""
    if n == 0:
        return None, None
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def evaluate(
    y_true,
    y_pred,
    proba=None,
    classes: list[str] = RISK_LEVELS,
    concordance_target: float = CONCORDANCE_TARGET,
) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    k, n = len(classes), len(y_true)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(k)))
    per_class = {}
    for i, name in enumerate(classes):
        tp = int(cm[i, i])
        fn = int(cm[i].sum() - tp)
        fp = int(cm[:, i].sum() - tp)
        tn = int(cm.sum() - tp - fn - fp)
        lo, hi = wilson_interval(tp, tp + fn)
        auc = None
        positives = y_true == i
        if proba is not None and positives.any() and (~positives).any():
            auc = float(roc_auc_score(positives, np.asarray(proba)[:, i]))
        per_class[name] = {
            "n": tp + fn,
            "sensibilidad": tp / (tp + fn) if tp + fn else None,
            "sensibilidad_ic95": [lo, hi],
            "especificidad": tn / (tn + fp) if tn + fp else None,
            "vpp": tp / (tp + fp) if tp + fp else None,
            "auc": auc,
        }
    concordance = float(np.trace(cm) / n) if n else None
    return {
        "n": n,
        "clases": list(classes),
        "matriz_confusion": cm.tolist(),
        "concordancia": concordance,
        "meta_concordancia": concordance_target,
        "cumple_meta_concordancia": bool(concordance is not None and concordance >= concordance_target),
        "por_clase": per_class,
        "errores_criticos_alto_como_bajo": int(cm[ALTO, BAJO]),
        "n_alto": int(cm[ALTO].sum()),
    }


def subgroup_report(values, y_true, y_pred, min_n: int = 1) -> dict:
    """Concordancia y sensibilidad de 'alto' por subgrupo (fuente, Fitzpatrick...)."""
    df = pd.DataFrame({"g": pd.Series(values).fillna("desconocido").astype(str).to_numpy(),
                       "t": np.asarray(y_true), "p": np.asarray(y_pred)})
    out = {}
    for g, part in df.groupby("g"):
        if len(part) < min_n:
            continue
        alto = part[part.t == ALTO]
        out[g] = {
            "n": int(len(part)),
            "concordancia": float((part.t == part.p).mean()),
            "n_alto": int(len(alto)),
            "sensibilidad_alto": float((alto.p == ALTO).mean()) if len(alto) else None,
        }
    return out


def fitzpatrick_group(value) -> str:
    try:
        v = int(value)
    except (TypeError, ValueError):
        return "desconocido"
    return {1: "I-II", 2: "I-II", 3: "III-IV", 4: "III-IV", 5: "V-VI", 6: "V-VI"}.get(v, "desconocido")


def plot_confusion(cm, classes: list[str], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes)
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Diagnóstico real")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_report(report: dict, out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    plot_confusion(report["matriz_confusion"], report["clases"], out_dir / "confusion_matrix.png")
