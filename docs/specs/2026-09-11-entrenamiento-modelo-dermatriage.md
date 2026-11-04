# Spec: Entrenamiento del modelo de clasificación de riesgo — DermaTriage
Fecha: 2026-09-11

## 1. Overview

DermaTriage necesita reemplazar su función de clasificación simulada por un modelo real de visión que reciba una foto clínica de una lesión de piel y devuelva un nivel de riesgo (bajo, medio o alto). El modelo se construye haciendo fine-tuning de una arquitectura de visión ya entrenada (EfficientNet) sobre el dataset público PAD-UFES-20, en vez de esperar a recolectar datos propios. Esto acelera la validación técnica del producto sin bloquear el trabajo en la disponibilidad de datos clínicos propios.

## 2. Usuarios objetivo

- **Karla (fundadora/médica)**: usa el pipeline de entrenamiento para producir un modelo evaluable, revisa las métricas clínicas resultantes y decide si el modelo está listo para integrarse al prototipo.
- **El prototipo DermaTriage (consumidor técnico)**: una vez entrenado y exportado, el modelo es consumido por la app como reemplazo directo de la función mock de clasificación — su usuario final indirecto es el mismo GP/enfermería que ya usa DermaTriage.

## 3. Contexto del problema

- El prototipo funcional de DermaTriage ya cubre todo el ciclo de caso (login, captura de foto, clasificación, cierre con diagnóstico real, reporte de concordancia), pero la clasificación está simulada con una función mock — es el hueco más importante para pasar de prototipo a validación real.
- La decisión original era diferir el entrenamiento de un modelo propio hasta tener datos clínicos y evidencia de adopción. Karla decidió adelantar esta etapa usando datos públicos (PAD-UFES-20) para tener un modelo real cuanto antes, en vez de seguir bloqueada por la falta de datos propios.
- Ya se definieron dos decisiones clínicas/técnicas clave en conversación:
  - Mapeo de las 6 categorías del dataset a los 3 niveles de riesgo: **Alto** = MEL; **Medio** = BCC, SCC, ACK; **Bajo** = NEV, SEK.
  - Arquitectura base: EfficientNet-B0/B3 preentrenado en ImageNet, vía fine-tuning.
- Riesgo técnico identificado y a medir explícitamente: PAD-UFES-20 son fotos clínicas de celular (no dermatoscopia), lo cual es favorable porque coincide con el tipo de foto que capturará DermaTriage en consulta — pero aun así puede haber diferencias de dispositivo, iluminación y encuadre frente al uso real, por lo que se necesita una prueba de generalización con fotos fuera del dataset.

## 4. Alcance versión 1

### Sí incluye:
- Descarga y preparación de PAD-UFES-20 con el mapeo de riesgo ya decidido, y split train/val/test estratificado.
- Entrenamiento por fine-tuning de EfficientNet-B0 (o B3 si el rendimiento lo justifica) para clasificación de 3 clases (bajo/medio/alto).
- Manejo del desbalance de clases (MEL tiene pocas muestras frente a NEV/SEK).
- Evaluación con métricas clínicamente relevantes: matriz de confusión, sensibilidad por clase (con foco en que "Alto" no se confunda con "Bajo"), curvas ROC/PR por clase — no solo accuracy general.
- Prueba de generalización con un conjunto pequeño de fotos de celular fuera del dataset (si Karla las provee), para estimar el domain shift frente al uso real en consulta.
- Exportación del modelo final a un formato listo para servir (ONNX o TorchScript).

### Explícitamente fuera de v1 (queda para v2/backlog):
- Integración real del modelo dentro de `DermaTriage.jsx` (reemplazar la función mock) — se hace en una iteración posterior, una vez el modelo esté validado.
- Pre-entrenamiento adicional con datasets dermatoscópicos (ISIC/HAM10000) — solo se considera si la prueba de domain shift muestra que hace falta.
- Segmentación de la lesión o explicabilidad visual tipo ABCDE automatizada.
- Modelo multimodal que incorpore metadatos del paciente (edad, ubicación de la lesión, Fitzpatrick) más allá de la imagen.
- Reentrenamiento incremental con casos reales cerrados en la app (requiere datos de adopción que aún no existen).

## 5. Comportamiento esperado

1. Karla (o quien ejecute el pipeline) corre el script/notebook de preparación de datos, que descarga PAD-UFES-20, aplica el mapeo de riesgo y genera los splits. El resultado es un conjunto de datos listo para entrenar, con la distribución de clases documentada.
2. Se ejecuta el entrenamiento: el pipeline reporta progreso por época (pérdida y métricas en validación) para que Karla pueda ver si el modelo está aprendiendo razonablemente antes de esperar a que termine todo el proceso.
3. Al finalizar el entrenamiento, el pipeline genera un reporte de evaluación sobre el set de test: matriz de confusión, sensibilidad/especificidad por clase, y las curvas correspondientes — en un formato legible (imágenes/tablas), no solo números sueltos en consola.
4. Si Karla provee fotos de celular externas al dataset, el pipeline las corre por el modelo y muestra las predicciones junto a la clasificación esperada (si la tiene), para que ella pueda juzgar visualmente si el modelo generaliza.
5. El pipeline termina exportando el modelo entrenado a un archivo (ONNX/TorchScript) junto con un resumen de las métricas finales, quedando listo como insumo para la siguiente etapa (integración a DermaTriage).

## 6. Posibles errores y mitigaciones

- **El dataset no descarga o cambia de estructura**: el pipeline debe fallar con un mensaje claro indicando qué archivo/URL faltó, no continuar silenciosamente con datos parciales.
- **Desbalance severo de clases (pocas muestras de MEL)**: se mitiga con class weights o oversampling explícito, y se reporta el conteo de muestras por clase antes de entrenar para que sea visible, no un supuesto oculto.
- **El modelo generaliza mal a fotos fuera del dataset (domain shift)**: no se trata como fallo silencioso — el pipeline debe mostrar explícitamente esas predicciones fallidas para que Karla decida si hace falta más datos o un ajuste de estrategia (ver backlog: pre-entrenamiento con dermatoscopia).
- **Entrenamiento se interrumpe (memoria, tiempo, error de hardware)**: guardar checkpoints periódicos para poder retomar sin perder todo el progreso.
- **Métricas engañosas por accuracy alto pero mala sensibilidad en "Alto" riesgo**: el criterio de éxito de este spec nunca es solo accuracy — un modelo con accuracy alto pero que confunde "Alto" con "Bajo" se considera no apto, y el reporte de evaluación debe hacer esa distinción visible de inmediato.

## Éxito

- El pipeline corre de punta a punta sin intervención manual entre pasos.
- El reporte de evaluación permite a Karla decidir, con datos concretos, si el modelo es candidato para integrarse a DermaTriage o si necesita otra iteración (más datos, otra arquitectura, ajuste del mapeo de clases).
- Sensibilidad de la clase "Alto" (MEL) priorizada explícitamente sobre accuracy general, dado el costo clínico de un falso negativo en esa categoría.

## V2

- Integración del modelo en `DermaTriage.jsx`.
- Pre-entrenamiento con datos dermatoscópicos si el domain shift lo justifica.
- Segmentación/explicabilidad ABCDE.
- Modelo multimodal con metadatos clínicos.
- Reentrenamiento incremental con datos reales de adopción.
