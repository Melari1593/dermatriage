"""Constantes y decisiones clínicas del proyecto.

Las decisiones clínicas viven aquí, en un solo lugar.
Cambia este archivo para cambiar el comportamiento. No cambies el código.
"""
from __future__ import annotations

import re

# Orden fijo de las clases. El índice de cada clase se usa en todo el código.
RISK_LEVELS = ["bajo", "medio", "alto"]
RISK_INDEX = {name: i for i, name in enumerate(RISK_LEVELS)}
ALTO = RISK_INDEX["alto"]
BAJO = RISK_INDEX["bajo"]

# Decisión de la Dra. Karla (2026-09-11): mapeo de PAD-UFES-20 a riesgo.
PAD_TO_RISK = {
    "MEL": "alto",
    "BCC": "medio",
    "SCC": "medio",  # PAD-UFES-20 ya incluye la enfermedad de Bowen en SCC.
    "ACK": "medio",
    "NEV": "bajo",
    "SEK": "bajo",
}

# Etiquetas de SCIN con equivalencia clara a las condiciones de PAD-UFES-20.
# Las claves están normalizadas con normalize_label().
# VERIFICAR contra el reporte de harmonize (lista de etiquetas sin mapear).
SCIN_TO_RISK = {
    "melanoma": "alto",
    "basalcellcarcinoma": "medio",
    "bcc": "medio",
    "sccsccis": "medio",
    "squamouscellcarcinoma": "medio",
    "actinickeratosis": "medio",
    "melanocyticnevus": "bajo",
    "nevus": "bajo",
    "seborrheickeratosis": "bajo",
    "skisk": "bajo",  # "SK/ISK" en SCIN.
}

# Calidad de imagen exigida en SCIN (valor del campo de gradabilidad).
SCIN_QUALITY_OK = "DEFAULT_YES_IMAGE_QUALITY_SUFFICIENT"

# Parámetros del modelo.
EMBEDDING_DIM = 6144  # Salida de Derm Foundation.
EMBEDDING_IMAGE_SIZE = 448  # Derm Foundation espera PNG de 448 x 448.
DERM_FOUNDATION_REPO = "google/derm-foundation"
SCIN_PRECOMPUTED_NPZ = "scin_dataset_precomputed_embeddings.npz"

# Objetivos de evaluación.
TARGET_SENSITIVITY_ALTO = 0.90  # Sensibilidad mínima para la clase "alto".
CONCORDANCE_TARGET = 0.80  # Meta del proyecto: concordancia >= 80 %.
MIN_CONFIDENCE = 0.50  # Bajo este valor, el resultado pide revisión humana.

# Texto de recomendación. Es una decisión clínica: edítalo.
REFERRAL_TEXT = {
    "alto": "Referir a dermatología de forma prioritaria.",
    "medio": "Referir a dermatología (cita programada).",
    "bajo": "Seguimiento en atención primaria. Reevaluar si la lesión cambia.",
}

DISCLAIMER = (
    "Herramienta de apoyo a la decisión. No reemplaza el juicio clínico "
    "ni el diagnóstico dermatológico."
)


def normalize_label(label: str) -> str:
    """Deja solo letras y números en minúscula. 'SK/ISK' -> 'skisk'."""
    return re.sub(r"[^a-z0-9]", "", str(label).lower())
