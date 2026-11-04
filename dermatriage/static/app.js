"use strict";

(() => {
  const MAX_SIDE = 1600; // Reduce fotos grandes antes de enviarlas.
  const MAX_BYTES = 10 * 1024 * 1024;
  const LEVELS = ["bajo", "medio", "alto"];
  const LABELS = { bajo: "Riesgo bajo", medio: "Riesgo medio", alto: "Riesgo alto" };
  const BAR_LABELS = { bajo: "Bajo", medio: "Medio", alto: "Alto" };
  const SVG_NS = "http://www.w3.org/2000/svg";
  const ICONS = {
    bajo: "M5 13l4 4L19 7",
    medio: "M12 4l9 16H3L12 4zM12 10v4M12 17.5v.5",
    alto: "M12 3v12M12 19v2",
  };
  const ERRORS = {
    400: "El archivo está vacío. Elige otra foto.",
    413: "La foto pesa más de 10 MB. Elige una más pequeña.",
    415: "Formato no admitido. Usa JPEG, PNG o WebP.",
    422: "No se pudo leer la foto. Toma otra.",
    503: "El servicio de análisis no está disponible ahora. Intenta más tarde.",
  };

  const $ = (id) => document.getElementById(id);
  const el = {
    input: $("foto"), label: $("etiqueta-foto"), preview: $("vista"), img: $("imagen"),
    fileName: $("nombre-archivo"), actions: $("acciones-clasificar"),
    classify: $("clasificar"), change: $("cambiar"), status: $("estado"), error: $("error"),
    result: $("resultado"), badge: $("insignia"), badgeIcon: $("insignia-icono"),
    badgeText: $("insignia-texto"), referral: $("recomendacion"), review: $("revision"),
    bars: $("barras"), meta: $("meta"), disclaimer: $("advertencia"), again: $("nueva"),
  };

  let file = null;
  let objectUrl = null;

  function setBusy(busy) {
    el.classify.disabled = busy;
    el.change.disabled = busy;
    el.input.disabled = busy;
    el.classify.textContent = busy ? "Analizando…" : "Clasificar";
    el.status.textContent = busy ? "Analizando la foto. Esto puede tardar unos segundos." : "";
  }

  function showError(message) {
    el.error.textContent = message;
    el.error.hidden = !message;
  }

  function clearPreview() {
    if (objectUrl) URL.revokeObjectURL(objectUrl);
    objectUrl = null;
    file = null;
    el.img.removeAttribute("src");
    el.preview.hidden = true;
    el.actions.hidden = true;
    el.input.value = "";
  }

  function onFileChosen() {
    const chosen = el.input.files && el.input.files[0];
    showError("");
    el.result.hidden = true;
    if (!chosen) return;
    if (!/^image\/(jpeg|png|webp)$/.test(chosen.type)) {
      clearPreview();
      showError(ERRORS[415]);
      return;
    }
    if (objectUrl) URL.revokeObjectURL(objectUrl);
    file = chosen;
    objectUrl = URL.createObjectURL(chosen);
    el.img.src = objectUrl;
    el.fileName.textContent = chosen.name;
    el.preview.hidden = false;
    el.actions.hidden = false;
    el.classify.focus();
  }

  // Reduce el lado mayor a MAX_SIDE y exporta JPEG. Si falla, envía el original.
  async function prepareImage(original) {
    try {
      const bitmap = await createImageBitmap(original);
      const scale = Math.min(1, MAX_SIDE / Math.max(bitmap.width, bitmap.height));
      const canvas = document.createElement("canvas");
      canvas.width = Math.round(bitmap.width * scale);
      canvas.height = Math.round(bitmap.height * scale);
      canvas.getContext("2d").drawImage(bitmap, 0, 0, canvas.width, canvas.height);
      bitmap.close();
      const blob = await new Promise((ok) => canvas.toBlob(ok, "image/jpeg", 0.92));
      return blob && blob.size < original.size ? blob : original;
    } catch (_) {
      return original;
    }
  }

  function drawIcon(level) {
    el.badgeIcon.replaceChildren();
    const svg = document.createElementNS(SVG_NS, "svg");
    svg.setAttribute("viewBox", "0 0 24 24");
    svg.setAttribute("fill", "none");
    svg.setAttribute("stroke", "currentColor");
    svg.setAttribute("stroke-width", "2.6");
    svg.setAttribute("stroke-linecap", "round");
    svg.setAttribute("stroke-linejoin", "round");
    const path = document.createElementNS(SVG_NS, "path");
    path.setAttribute("d", ICONS[level]);
    svg.appendChild(path);
    el.badgeIcon.appendChild(svg);
  }

  function percent(value) {
    return `${Math.round(value * 100)} %`;
  }

  function renderResult(data) {
    const level = LEVELS.includes(data.riesgo) ? data.riesgo : "medio";
    el.badge.dataset.nivel = level;
    el.badgeText.textContent = LABELS[level];
    drawIcon(level);
    el.referral.textContent = data.recomendacion || "";
    el.review.hidden = !data.requiere_revision;

    el.bars.replaceChildren();
    for (const name of LEVELS) {
      const value = Number((data.probabilidades || {})[name] || 0);
      const row = document.createElement("li");
      row.className = "bar-row";
      const label = document.createElement("span");
      label.className = "bar-label";
      label.textContent = BAR_LABELS[name];
      const track = document.createElement("span");
      track.className = "bar-track";
      const fill = document.createElement("span");
      fill.className = "bar-fill";
      fill.dataset.nivel = name;
      fill.style.width = `${Math.max(0, Math.min(1, value)) * 100}%`;
      track.appendChild(fill);
      const text = document.createElement("span");
      text.className = "bar-value";
      text.textContent = percent(value);
      row.append(label, track, text);
      el.bars.appendChild(row);
    }
    el.meta.textContent = `Modelo ${data.version_modelo || "desconocido"}. Umbral de riesgo alto: ${percent(data.umbral_alto || 0)}.`;
    el.disclaimer.textContent = data.advertencia || "";
    el.result.hidden = false;
    el.result.focus();
    el.result.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  async function classify() {
    if (!file) return;
    showError("");
    el.result.hidden = true;
    setBusy(true);
    try {
      const image = await prepareImage(file);
      if (image.size > MAX_BYTES) {
        showError(ERRORS[413]);
        return;
      }
      const body = new FormData();
      body.append("file", image, "foto.jpg");
      const res = await fetch("/classify", { method: "POST", body });
      if (!res.ok) {
        showError(ERRORS[res.status] || "El servicio no respondió bien. Intenta de nuevo en unos minutos.");
        return;
      }
      renderResult(await res.json());
    } catch (_) {
      showError("No hay conexión con el servicio. Revisa tu red e intenta de nuevo.");
    } finally {
      setBusy(false);
    }
  }

  function reset() {
    clearPreview();
    showError("");
    el.result.hidden = true;
    el.label.scrollIntoView({ behavior: "smooth", block: "center" });
    el.input.focus();
  }

  el.input.addEventListener("change", onFileChosen);
  el.classify.addEventListener("click", classify);
  el.change.addEventListener("click", () => el.input.click());
  el.again.addEventListener("click", reset);
})();
