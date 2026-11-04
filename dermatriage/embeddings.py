"""Embeddings con Derm Foundation (imagen -> vector de 6144 valores).

El modelo es restringido en Hugging Face.
Acepta los términos en https://huggingface.co/google/derm-foundation
y define HF_TOKEN antes de usarlo.
"""
from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from .config import (
    DERM_FOUNDATION_REPO,
    EMBEDDING_DIM,
    EMBEDDING_IMAGE_SIZE,
    SCIN_PRECOMPUTED_NPZ,
)


class Embedder(Protocol):
    name: str

    def embed_bytes(self, image_bytes: bytes) -> np.ndarray: ...


def to_png_bytes(image_bytes: bytes, size: int = EMBEDDING_IMAGE_SIZE) -> bytes:
    """Convierte cualquier imagen a PNG RGB de size x size."""
    from PIL import Image

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((size, size))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class DermFoundationEmbedder:
    """Envoltura de Derm Foundation (SavedModel de TensorFlow).

    Interfaz (según la documentación pública del modelo):
    entrada = tf.Example serializado con 'image/encoded' (bytes PNG);
    salida = clave 'embedding' con forma (1, 6144).
    VERIFICAR en la primera ejecución: el modelo es restringido
    y no pude leer su ficha desde el entorno de desarrollo.
    """

    name = DERM_FOUNDATION_REPO

    def __init__(self, model_dir: str | None = None):
        import tensorflow as tf
        from huggingface_hub import snapshot_download

        self._tf = tf
        path = model_dir or snapshot_download(
            DERM_FOUNDATION_REPO, allow_patterns=["saved_model.pb", "variables/*", "fingerprint.pb"]
        )
        self._infer = tf.saved_model.load(path).signatures["serving_default"]

    def embed_bytes(self, image_bytes: bytes) -> np.ndarray:
        tf = self._tf
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "image/encoded": tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[to_png_bytes(image_bytes)])
                    )
                }
            )
        ).SerializeToString()
        out = self._infer(inputs=tf.constant([example]))
        vec = np.asarray(out["embedding"]).reshape(-1).astype(np.float32)
        if vec.shape[0] != EMBEDDING_DIM:
            raise RuntimeError(f"Embedding de {vec.shape[0]} valores. Esperaba {EMBEDDING_DIM}.")
        return vec


def load_precomputed_scin(npz_path: str | None = None) -> dict[str, np.ndarray]:
    """Embeddings de SCIN ya calculados por Google. Clave = ruta de imagen SCIN."""
    if npz_path is None:
        from huggingface_hub import hf_hub_download

        npz_path = hf_hub_download(DERM_FOUNDATION_REPO, SCIN_PRECOMPUTED_NPZ)
    out = {}
    with np.load(npz_path, allow_pickle=True) as data:
        for key in data.files:
            out[key] = np.asarray(data[key], dtype=np.float32).reshape(-1)
    return out


def build_embeddings(
    manifest: pd.DataFrame,
    out_path: Path,
    embedder: Embedder | None = None,
    precomputed: dict[str, np.ndarray] | None = None,
) -> dict:
    """Crea embeddings.npz (ids + X) y devuelve un reporte.

    Prioridad: embedding precalculado > calcular desde la imagen local.
    Las muestras sin ninguna de las dos se omiten y se reportan.
    """
    precomputed = precomputed or {}
    ids, vecs, missing, reused = [], [], [], 0
    for r in manifest.itertuples():
        if r.sample_id in precomputed:
            vec, reused = precomputed[r.sample_id], reused + 1
        elif isinstance(r.image_path, str) and r.image_path and Path(r.image_path).exists():
            if embedder is None:
                missing.append(r.sample_id)
                continue
            try:
                vec = embedder.embed_bytes(Path(r.image_path).read_bytes())
            except Exception as exc:  # Imagen corrupta: la omitimos y la reportamos.
                missing.append(f"{r.sample_id} ({type(exc).__name__})")
                continue
        else:
            missing.append(r.sample_id)
            continue
        ids.append(r.sample_id)
        vecs.append(vec)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not ids:
        raise RuntimeError("No se creó ningún embedding. Revisa las rutas de imágenes.")
    np.savez_compressed(out_path, ids=np.array(ids), X=np.stack(vecs).astype(np.float32))
    report = {"total": len(manifest), "con_embedding": len(ids), "precalculados": reused, "omitidos": len(missing)}
    (out_path.parent / "embeddings_report.json").write_text(
        json.dumps({**report, "omitidos_ids": missing[:200]}, indent=2, ensure_ascii=False)
    )
    return report


def load_embeddings(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return dict(zip(data["ids"].tolist(), data["X"]))
