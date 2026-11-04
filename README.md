# DermaTriage — modelo de clasificación de riesgo

Este paquete clasifica una foto de lesión de piel en riesgo **bajo**, **medio** o **alto**.
Usa embeddings de Derm Foundation (Google) y un clasificador ligero encima.
Los datos de entrenamiento son PAD-UFES-20 y SCIN.

> Es una herramienta de apoyo a la decisión. No reemplaza el juicio clínico.
> Una evaluación offline con datos públicos no equivale a validación clínica.

## Mapeo de riesgo (decisión de la Dra. Karla, 2026-09-11)

| Riesgo | Condiciones |
|--------|-------------|
| Alto   | MEL |
| Medio  | BCC, SCC, ACK |
| Bajo   | NEV, SEK |

El mapeo vive en `dermatriage/config.py`. Cambia ese archivo para cambiar el mapeo.

## Estructura

```
dermatriage/
  config.py       Mapeo de riesgo, objetivos y textos de recomendación
  harmonize.py    Fase 0: une PAD-UFES-20 y SCIN en manifest.csv
  embeddings.py   Imagen -> vector de 6144 valores (Derm Foundation)
  splits.py       train/val/test sin fuga entre pacientes
  train.py        Entrenamiento y umbral clínico para "alto"
  evaluate.py     Sensibilidad, errores críticos, subgrupos
  predict.py      Foto -> riesgo + recomendación
  serve.py        API HTTP (FastAPI)
  cli.py          Comandos: harmonize, embed, train
integration/
  classifyLesion.js   Reemplaza la función simulada en DermaTriage.jsx
tests/
  test_pipeline.py    16 pruebas con datos sintéticos
```

## Requisitos

1. Python 3.10 o superior. Ejecuta `pip install -r requirements.txt`.
2. Acceso a Derm Foundation. El modelo es restringido.
   Acepta los términos en https://huggingface.co/google/derm-foundation.
   Define `HF_TOKEN` con tu token de Hugging Face.
3. PAD-UFES-20: descárgalo de https://data.mendeley.com/datasets/zr7vgbcyr2.
   La carpeta debe tener `metadata.csv` y las imágenes.
4. SCIN: descarga solo los CSV.
   ```
   gsutil cp gs://dx-scin-public-data/dataset/scin_cases.csv  data/scin/
   gsutil cp gs://dx-scin-public-data/dataset/scin_labels.csv data/scin/
   ```
   No necesitas las imágenes de SCIN. Google publica sus embeddings ya calculados.

## Uso

```bash
# 1. Une los datos y revisa el reporte.
python -m dermatriage.cli harmonize --pad data/pad_ufes_20 --scin data/scin --out artifacts

# 2. Calcula los embeddings. Usa los de SCIN precalculados y calcula los de PAD-UFES-20.
python -m dermatriage.cli embed --manifest artifacts/manifest.csv --out artifacts/embeddings.npz

# 3. Entrena y evalúa.
python -m dermatriage.cli train --manifest artifacts/manifest.csv \
    --embeddings artifacts/embeddings.npz --out artifacts

# 4. Levanta la API.
DERMA_MODEL_PATH=artifacts/model.joblib uvicorn dermatriage.serve:app --port 8000

# 5. Ejecuta las pruebas.
pytest
```

Salidas de `train` en `artifacts/`: `model.joblib`, `report.json`,
`confusion_matrix.png`, `splits.csv`, `model_card.json`.

## Cómo decide el modelo

1. El clasificador calcula la probabilidad de cada nivel de riesgo.
2. Si la probabilidad de "alto" supera el umbral, el resultado es "alto".
3. Si no, gana "medio" o "bajo", según la probabilidad mayor.
4. El umbral se elige en validación. Es el más alto que logra sensibilidad ≥ 90 % en "alto".
5. Si el resultado no es "alto" y la confianza es menor de 50 %, la respuesta marca `requiere_revision`.

El umbral prioriza no perder melanomas. El costo es más falsos positivos.

## Integración con DermaTriage.jsx

1. Copia `integration/classifyLesion.js` al proyecto.
2. Reemplaza la llamada a la función simulada por `await classifyLesion(photoFile)`.
3. Define `VITE_DERMA_API_URL` con la URL de la API.
4. Si la función lanza un error, pon el caso en la cola de reintento sin conexión.

El resultado trae `risk` (`low`, `medium`, `high`), `probabilities`, `confidence`,
`needsReview` y `referral`. Ajusta los nombres si el prototipo usa otros.

## Decisiones pendientes (de la Dra. Karla)

1. **Condiciones de SCIN.** SCIN tiene sobre todo enfermedades inflamatorias
   (eczema, psoriasis, urticaria...). Pocas coinciden con los 6 tipos de PAD-UFES-20.
   Por defecto, el pipeline descarta lo que no tiene equivalencia clara.
   Revisa `harmonize_report.json` (campo `sin_mapeo_top`).
   Si quieres usar esas condiciones como riesgo bajo, usa `--scin-benign-as-bajo "Eczema,Psoriasis"`.
   Esa es una decisión clínica. El pipeline no la toma por ti.
2. **Casos SCIN con varias etiquetas.** Gana el riesgo más alto del caso.
3. **Textos de recomendación** (`REFERRAL_TEXT` en `config.py`). Son valores iniciales.
4. **Objetivo de sensibilidad** (90 % en "alto") y **umbral de confianza** (50 %).
5. **Árbol de triaje interpretable.** Aprobaste ese protocolo, pero su texto no estaba en esta sesión.
   No lo implementé. Compártelo y lo agrego como capa sobre el clasificador.

## Qué NO está verificado

- **Interfaz de inferencia de Derm Foundation.** El modelo es restringido.
  No pude leer su ficha. La clase `DermFoundationEmbedder` sigue la documentación pública
  (tf.Example con `image/encoded` en PNG de 448 x 448; salida `embedding`).
  Si la primera ejecución falla en ese punto, el cambio es local a esa clase.
- **Nombres de etiquetas de SCIN.** Los confirmé con código público para las 10 condiciones más comunes.
  Los nombres de cáncer (melanoma, BCC, SCC...) deben coincidir con `SCIN_TO_RISK` en `config.py`.
  Revisa la lista de etiquetas sin mapear.
- **Estructura de carpetas de PAD-UFES-20.** El cargador busca los PNG de forma recursiva.
- **Ninguna ejecución con datos reales.** Las 16 pruebas usan datos sintéticos.

## Privacidad

- La API no guarda imágenes y no registra su contenido.
- No subas fotos de pacientes ni datos clínicos reales a este repositorio.
- Activa CORS solo para tu dominio: `DERMA_CORS_ORIGINS=https://tu-dominio`.
