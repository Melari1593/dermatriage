# Plan de implementación: Entrenamiento del modelo de clasificación de riesgo — DermaTriage
Fecha: 2026-09-11

## 1. Objetivo

Construir y evaluar un modelo de clasificación de riesgo (bajo/medio/alto) por fine-tuning de EfficientNet sobre PAD-UFES-20, dejando un modelo exportado y un reporte de métricas clínicamente relevantes que le permitan a Karla decidir si está listo para integrarse a DermaTriage.

## 2. Contexto del problema

DermaTriage tiene su clasificación de riesgo simulada con una función mock; este es el hueco principal para pasar de prototipo a validación real. En vez de esperar a tener datos clínicos propios, se adelanta el entrenamiento usando PAD-UFES-20 (fotos clínicas de celular, lo que reduce el riesgo de domain shift frente a dermatoscopia). Ya están decididos el mapeo de clases a riesgo y la arquitectura base.

## 3. Spec de referencia

`docs/specs/2026-09-11-entrenamiento-modelo-dermatriage.md`

Puntos clave del spec:
- Mapeo de riesgo: Alto = MEL; Medio = BCC, SCC, ACK; Bajo = NEV, SEK.
- Arquitectura: EfficientNet-B0 (o B3) preentrenado en ImageNet, fine-tuning a 3 clases.
- Éxito medido por sensibilidad de la clase "Alto", no solo accuracy general.
- Fuera de alcance: integración a `DermaTriage.jsx`, pre-entrenamiento dermatoscópico, segmentación, multimodalidad — todo eso es v2.

## 4. Lista de tareas a implementar con detalles

### Tarea 1 — Descarga y preparación de datos
- **Qué hacer**: script `prepare_data.py` que descarga PAD-UFES-20 (imágenes + CSV de metadatos), aplica el mapeo de las 6 categorías a los 3 niveles de riesgo, y genera splits train/val/test estratificados por clase de riesgo (ej. 70/15/15).
- **Archivos/componentes**: `data/prepare_data.py`, salida en `data/processed/{train,val,test}/` con subcarpetas por clase de riesgo.
- **Criterio de hecho**: el script corre sin errores, imprime el conteo de imágenes por clase en cada split, y los tres splits no se solapan en pacientes (para evitar leakage si un mismo paciente tiene varias imágenes).

### Tarea 2 — Dataset y augmentation
- **Qué hacer**: definir un `Dataset`/`DataLoader` (PyTorch) que cargue las imágenes procesadas, aplique resize/normalización acorde a EfficientNet, y aplique augmentation (flip, rotación, color jitter) solo en train.
- **Archivos/componentes**: `data/dataset.py`.
- **Criterio de hecho**: iterar un batch del DataLoader de train y de val sin errores, con las dimensiones y rangos de valores esperados por EfficientNet.

### Tarea 3 — Manejo de desbalance de clases
- **Qué hacer**: calcular pesos de clase inversamente proporcionales a la frecuencia (o usar oversampling) a partir del conteo real de la Tarea 1, e incorporarlos a la función de pérdida o al sampler.
- **Archivos/componentes**: `training/class_weights.py`, usado en `training/train.py`.
- **Criterio de hecho**: los pesos/sampler quedan documentados en el log de entrenamiento antes de iniciar (ej. "peso MEL: X, NEV: Y...").

### Tarea 4 — Script de entrenamiento
- **Qué hacer**: `training/train.py` que carga EfficientNet-B0 preentrenado, reemplaza la cabeza por una capa de 3 clases, entrena con la pérdida ponderada, reporta pérdida/accuracy/sensibilidad por clase en validación cada época, y guarda checkpoints periódicos.
- **Archivos/componentes**: `training/train.py`, `checkpoints/`.
- **Criterio de hecho**: el entrenamiento corre por el número de épocas configurado, se puede retomar desde el último checkpoint si se interrumpe, y produce un log legible por época.

### Tarea 5 — Evaluación clínica
- **Qué hacer**: `evaluation/evaluate.py` que corre el modelo final sobre el set de test y genera matriz de confusión, sensibilidad/especificidad por clase, y curvas ROC/PR por clase, guardadas como imágenes/tablas en un reporte.
- **Archivos/componentes**: `evaluation/evaluate.py`, salida en `reports/evaluation_report/`.
- **Criterio de hecho**: el reporte generado muestra explícitamente la sensibilidad de la clase "Alto" de forma destacada (no enterrada entre otras métricas), y las imágenes/tablas se abren correctamente.

### Tarea 6 — Prueba de domain shift
- **Qué hacer**: `evaluation/test_external.py` que corre el modelo sobre un conjunto de fotos externas (provisto por Karla, si existe) y muestra predicción vs. etiqueta esperada (si la hay) imagen por imagen.
- **Archivos/componentes**: `evaluation/test_external.py`, carpeta de entrada `data/external_samples/`.
- **Criterio de hecho**: el script corre incluso si la carpeta de fotos externas está vacía (mostrando un aviso claro, no un error), y produce una tabla/imagen comparando predicción vs. etiqueta cuando sí hay datos.

### Tarea 7 — Exportación del modelo
- **Qué hacer**: `export/export_model.py` que convierte el mejor checkpoint a ONNX (o TorchScript), junto con un archivo `model_card.md` que resume arquitectura, mapeo de clases, y métricas finales del reporte de evaluación.
- **Archivos/componentes**: `export/export_model.py`, salida `models/dermatriage_risk_model.onnx` + `models/model_card.md`.
- **Criterio de hecho**: el modelo exportado se puede cargar con un runtime de inferencia (ej. `onnxruntime`) y producir una predicción de prueba sin errores.

Orden de implementación: Tareas 1→2→3 pueden avanzar en paralelo una vez lista la 1; la 4 depende de 2 y 3; la 5 y 6 dependen de 4; la 7 depende de 5 (se exporta el modelo ya evaluado y aprobado).
