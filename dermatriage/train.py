"""Entrenamiento: regresión logística sobre embeddings + umbral clínico para 'alto'."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import __version__
from .config import ALTO, RISK_LEVELS, TARGET_SENSITIVITY_ALTO
from .embeddings import load_embeddings
from .evaluate import evaluate, fitzpatrick_group, save_report, subgroup_report
from .splits import make_splits


def decide(proba: np.ndarray, alto_threshold: float) -> np.ndarray:
    """Regla de decisión. 'alto' si p(alto) >= umbral. Si no, gana medio o bajo."""
    other = [i for i in range(proba.shape[1]) if i != ALTO]
    pred = np.array(other)[np.argmax(proba[:, other], axis=1)]
    pred[proba[:, ALTO] >= alto_threshold] = ALTO
    return pred


def choose_alto_threshold(p_alto: np.ndarray, is_alto: np.ndarray, target: float) -> float:
    """Umbral más alto que logra sensibilidad >= target en validación.

    Un umbral más alto significa menos falsos positivos.
    Si ningún umbral logra el objetivo, usa el más bajo posible.
    """
    if is_alto.sum() == 0:
        raise ValueError("La validación no tiene casos de riesgo alto. Revisa los datos.")
    candidates = np.unique(np.concatenate([p_alto, [0.0]]))[::-1]
    for t in candidates:
        if (p_alto[is_alto] >= t).mean() >= target:
            return float(t)
    return 0.0


def load_xy(manifest_csv: Path, embeddings_npz: Path) -> tuple[pd.DataFrame, np.ndarray]:
    manifest = pd.read_csv(manifest_csv)
    emb = load_embeddings(embeddings_npz)
    manifest = manifest[manifest.sample_id.isin(emb)].reset_index(drop=True)
    X = np.stack([emb[s] for s in manifest.sample_id]).astype(np.float32)
    return manifest, X


def run_training(
    manifest: pd.DataFrame,
    X: np.ndarray,
    out_dir: Path,
    seed: int = 42,
    target_sensitivity: float = TARGET_SENSITIVITY_ALTO,
    c_grid: tuple[float, ...] = (0.0005, 0.002, 0.01, 0.05, 0.2),
    embedder_name: str = "google/derm-foundation",
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = make_splits(manifest, seed=seed)
    y = df["risk"].map({r: i for i, r in enumerate(RISK_LEVELS)}).to_numpy()
    tr, va, te = (np.where(df.split == s)[0] for s in ("train", "val", "test"))
    for name, idx in (("train", tr), ("val", va), ("test", te)):
        if len(np.unique(y[idx])) < len(RISK_LEVELS):
            raise ValueError(f"La partición {name} no tiene las 3 clases. Faltan datos.")

    best = None
    for c in c_grid:
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=c, max_iter=3000, class_weight="balanced"),
        )
        model.fit(X[tr], y[tr])
        score = balanced_accuracy_score(y[va], model.predict(X[va]))
        if best is None or score > best[0]:
            best = (score, c, model)
    _, best_c, model = best

    p_val = model.predict_proba(X[va])
    threshold = choose_alto_threshold(p_val[:, ALTO], y[va] == ALTO, target_sensitivity)

    p_test = model.predict_proba(X[te])
    pred_test = decide(p_test, threshold)
    report = evaluate(y[te], pred_test, p_test)
    report["umbral_alto"] = threshold
    report["objetivo_sensibilidad_alto"] = target_sensitivity
    report["C"] = best_c
    report["por_fuente"] = subgroup_report(df.loc[te, "source"], y[te], pred_test)
    report["por_fitzpatrick"] = subgroup_report(
        df.loc[te, "fitzpatrick"].map(fitzpatrick_group), y[te], pred_test
    )
    report["tamanos"] = {s: int((df.split == s).sum()) for s in ("train", "val", "test")}
    report["advertencia"] = (
        "Evaluación offline con datos públicos. No equivale a validación clínica."
    )
    save_report(report, out_dir)
    df[["sample_id", "group_key", "risk", "source", "split"]].to_csv(out_dir / "splits.csv", index=False)

    joblib.dump(
        {
            "pipeline": model,
            "alto_threshold": threshold,
            "classes": RISK_LEVELS,
            "embedder": embedder_name,
            "version": __version__,
            "metrics": {k: report[k] for k in ("concordancia", "errores_criticos_alto_como_bajo", "n")},
        },
        out_dir / "model.joblib",
    )
    (out_dir / "model_card.json").write_text(
        json.dumps({"version": __version__, "embedder": embedder_name, "umbral_alto": threshold,
                    "tamanos": report["tamanos"], "concordancia_test": report["concordancia"]},
                   indent=2, ensure_ascii=False)
    )
    return report
