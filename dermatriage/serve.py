"""API HTTP para DermaTriage.

Ejecutar:  DERMA_MODEL_PATH=artifacts/model.joblib uvicorn dermatriage.serve:app --port 8000
El servidor no guarda las imágenes. No registra su contenido.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .predict import RiskClassifier

MAX_BYTES = 10 * 1024 * 1024
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}
STATIC_DIR = Path(__file__).parent / "static"

SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "X-Frame-Options": "DENY",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
    "Content-Security-Policy": (
        "default-src 'self'; img-src 'self' blob:; style-src 'self'; script-src 'self'; "
        "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'"
    ),
}

app = FastAPI(title="DermaTriage API", version="0.1.0")
origins = [o for o in os.getenv("DERMA_CORS_ORIGINS", "").split(",") if o]
if origins:
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["POST", "GET"], allow_headers=["*"])


@app.middleware("http")
async def add_headers(request: Request, call_next):
    response = await call_next(request)
    for key, value in SECURITY_HEADERS.items():
        response.headers.setdefault(key, value)
    if request.url.path in {"/classify", "/health"}:
        response.headers["Cache-Control"] = "no-store"  # Nunca guardar resultados clínicos.
    return response


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
        classifier = get_classifier()
    except RuntimeError as exc:  # Falta el archivo del modelo.
        raise HTTPException(503, "El modelo no está disponible.") from exc
    try:
        return classifier.classify_bytes(data)
    except (OSError, ValueError) as exc:  # Imagen ilegible.
        raise HTTPException(422, "No se pudo leer la imagen.") from exc


# La página web va al final. Así no tapa las rutas de la API.
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="web")
