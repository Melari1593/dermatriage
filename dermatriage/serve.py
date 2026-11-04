"""API HTTP para DermaTriage.

Ejecutar:  DERMA_MODEL_PATH=artifacts/model.joblib uvicorn dermatriage.serve:app --port 8000
El servidor no guarda las imágenes. No registra su contenido.
"""
from __future__ import annotations

import os
from functools import lru_cache

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .predict import RiskClassifier

MAX_BYTES = 10 * 1024 * 1024
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}

app = FastAPI(title="DermaTriage API", version="0.1.0")
origins = [o for o in os.getenv("DERMA_CORS_ORIGINS", "").split(",") if o]
if origins:
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["POST", "GET"], allow_headers=["*"])


@lru_cache(maxsize=1)
def get_classifier() -> RiskClassifier:
    path = os.getenv("DERMA_MODEL_PATH", "artifacts/model.joblib")
    if not os.path.exists(path):
        raise RuntimeError(f"No encontré el modelo en {path}. Define DERMA_MODEL_PATH.")
    return RiskClassifier.load(path)


@app.get("/health")
def health() -> dict:
    return {"estado": "ok"}


@app.post("/classify")
async def classify(file: UploadFile = File(...)) -> dict:
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(415, "Formato no admitido. Usa JPEG, PNG o WebP.")
    data = await file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "La imagen supera 10 MB.")
    if not data:
        raise HTTPException(400, "El archivo está vacío.")
    try:
        return get_classifier().classify_bytes(data)
    except (OSError, ValueError) as exc:  # Imagen ilegible.
        raise HTTPException(422, "No se pudo leer la imagen.") from exc
