"""Inferencia: foto -> riesgo + recomendación."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from . import __version__
from .config import ALTO, DISCLAIMER, MIN_CONFIDENCE, REFERRAL_TEXT, RISK_LEVELS
from .train import decide


class RiskClassifier:
    def __init__(self, bundle: dict, embedder=None, min_confidence: float = MIN_CONFIDENCE):
        self.bundle = bundle
        self._embedder = embedder
        self.min_confidence = min_confidence

    @classmethod
    def load(cls, model_path: str | Path, embedder=None) -> "RiskClassifier":
        return cls(joblib.load(model_path), embedder=embedder)

    @property
    def embedder(self):
        if self._embedder is None:  # Carga perezosa: TensorFlow pesa y tarda.
            from .embeddings import DermFoundationEmbedder

            self._embedder = DermFoundationEmbedder()
        return self._embedder

    def classify_embedding(self, vec: np.ndarray) -> dict:
        proba = self.bundle["pipeline"].predict_proba(np.asarray(vec, dtype=np.float32).reshape(1, -1))
        threshold = self.bundle["alto_threshold"]
        idx = int(decide(proba, threshold)[0])
        level = RISK_LEVELS[idx]
        confidence = float(proba[0, idx])
        # Si el sistema sube a 'alto' por umbral, la confianza puede ser baja.
        # Eso es intencional: se prefiere un falso positivo a un falso negativo.
        needs_review = idx != ALTO and confidence < self.min_confidence
        return {
            "riesgo": level,
            "probabilidades": {name: float(proba[0, i]) for i, name in enumerate(RISK_LEVELS)},
            "confianza": confidence,
            "requiere_revision": bool(needs_review),
            "recomendacion": REFERRAL_TEXT[level],
            "umbral_alto": threshold,
            "version_modelo": self.bundle.get("version", __version__),
            "advertencia": DISCLAIMER,
        }

    def classify_bytes(self, image_bytes: bytes) -> dict:
        return self.classify_embedding(self.embedder.embed_bytes(image_bytes))
