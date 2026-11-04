// Reemplaza la función simulada de clasificación en DermaTriage.jsx.
//
// Uso:
//   import { classifyLesion } from "./integration/classifyLesion.js";
//   const result = await classifyLesion(photoFile);
//   // result.risk: "low" | "medium" | "high"
//
// Si la llamada falla, la función lanza un Error.
// El llamador debe poner el caso en la cola de reintento sin conexión.

const API_URL =
  (typeof import.meta !== "undefined" && import.meta.env?.VITE_DERMA_API_URL) ||
  "http://localhost:8000";

const RISK_MAP = { bajo: "low", medio: "medium", alto: "high" };

export async function classifyLesion(photo, { timeoutMs = 30000 } = {}) {
  const body = new FormData();
  body.append("file", photo);

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(`${API_URL}/classify`, {
      method: "POST",
      body,
      signal: controller.signal,
    });
    if (!res.ok) {
      throw new Error(`Clasificación fallida (HTTP ${res.status}).`);
    }
    const data = await res.json();
    return {
      risk: RISK_MAP[data.riesgo],
      riskEs: data.riesgo,
      probabilities: data.probabilidades,
      confidence: data.confianza,
      needsReview: data.requiere_revision,
      referral: data.recomendacion,
      modelVersion: data.version_modelo,
      disclaimer: data.advertencia,
    };
  } finally {
    clearTimeout(timer);
  }
}
