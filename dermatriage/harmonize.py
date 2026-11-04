"""Fase 0: armonización de datos.

Lee PAD-UFES-20 y SCIN. Produce un manifiesto único con la columna `risk`.
"""
from __future__ import annotations

import ast
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from .config import (
    PAD_TO_RISK,
    RISK_INDEX,
    SCIN_QUALITY_OK,
    SCIN_TO_RISK,
    normalize_label,
)

MANIFEST_COLUMNS = [
    "sample_id",
    "source",
    "image_path",
    "group_key",
    "raw_label",
    "risk",
    "fitzpatrick",
]


def load_pad_ufes(root: Path) -> tuple[pd.DataFrame, dict]:
    """Carga PAD-UFES-20. `root` contiene metadata.csv y las carpetas imgs_part_*."""
    root = Path(root)
    meta = pd.read_csv(root / "metadata.csv")
    images = {p.name: p for p in root.rglob("*.png")}
    rows, dropped = [], Counter()
    for r in meta.itertuples():
        code = str(r.diagnostic).strip().upper()
        if code not in PAD_TO_RISK:
            dropped["etiqueta_sin_mapeo"] += 1
            continue
        path = images.get(str(r.img_id))
        if path is None:
            dropped["imagen_no_encontrada"] += 1
            continue
        fitz = getattr(r, "fitspatrick", None)  # El dataset escribe "fitspatrick".
        rows.append(
            {
                "sample_id": f"pad/{r.img_id}",
                "source": "pad_ufes_20",
                "image_path": str(path),
                "group_key": f"pad/{r.patient_id}",
                "raw_label": code,
                "risk": PAD_TO_RISK[code],
                "fitzpatrick": _to_int(fitz),
            }
        )
    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS), dict(dropped)


def _to_int(value) -> int | None:
    try:
        if pd.isna(value):
            return None
        digits = "".join(ch for ch in str(value) if ch.isdigit())
        return int(digits[0]) if digits else None
    except (TypeError, ValueError):
        return None


def _parse_list(text) -> list:
    if isinstance(text, list):
        return text
    try:
        value = ast.literal_eval(text)
        return value if isinstance(value, list) else []
    except (ValueError, SyntaxError):
        return []


def load_scin(
    root: Path,
    benign_as_bajo: tuple[str, ...] = (),
    min_confidence: float = 0.0,
) -> tuple[pd.DataFrame, dict]:
    """Carga SCIN. `root` contiene scin_cases.csv y scin_labels.csv.

    Cada caso puede tener hasta 3 imágenes. Cada imagen es una muestra.
    Si un caso tiene varias etiquetas con mapeo, gana el riesgo más alto.
    `benign_as_bajo` lista condiciones SCIN que se tratan como riesgo bajo.
    El default es vacío: decide la Dra. Karla (ver README).
    """
    root = Path(root)
    cases_csv = _first_existing(root, ["scin_cases.csv", "dataset/scin_cases.csv"])
    labels_csv = _first_existing(root, ["scin_labels.csv", "dataset/scin_labels.csv"])
    cases = pd.read_csv(cases_csv, dtype={"case_id": str})
    labels = pd.read_csv(labels_csv, dtype={"case_id": str})
    df = cases.merge(labels, on="case_id")

    extra = {normalize_label(x): "bajo" for x in benign_as_bajo}
    mapping = {**SCIN_TO_RISK, **extra}
    rows, dropped, unmapped = [], Counter(), Counter()

    for r in df.to_dict("records"):
        quality = r.get("dermatologist_gradable_for_skin_condition_1")
        if quality != SCIN_QUALITY_OK:
            dropped["calidad_insuficiente"] += 1
            continue
        names = _parse_list(r.get("dermatologist_skin_condition_on_label_name"))
        confs = _parse_list(r.get("dermatologist_skin_condition_confidence"))
        if not names:
            dropped["sin_etiqueta"] += 1
            continue
        confs = confs + [1.0] * (len(names) - len(confs))
        mapped = []
        for name, conf in zip(names, confs):
            risk = mapping.get(normalize_label(name))
            if risk is None:
                unmapped[str(name)] += 1
            elif conf >= min_confidence:
                mapped.append((RISK_INDEX[risk], name))
        if not mapped:
            dropped["etiqueta_sin_mapeo"] += 1
            continue
        risk_idx, raw = max(mapped)
        risk = next(k for k, v in RISK_INDEX.items() if v == risk_idx)
        fitz = _to_int(r.get("dermatologist_fitzpatrick_skin_type_label_1"))
        for col in ("image_1_path", "image_2_path", "image_3_path"):
            path = r.get(col)
            if not isinstance(path, str) or not path:
                continue
            local = root / path
            rows.append(
                {
                    "sample_id": path,  # Igual a la clave del npz precalculado.
                    "source": "scin",
                    "image_path": str(local) if local.exists() else "",
                    "group_key": f"scin/{r['case_id']}",
                    "raw_label": raw,
                    "risk": risk,
                    "fitzpatrick": fitz,
                }
            )
    out = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    return out, {"descartados": dict(dropped), "sin_mapeo_top": unmapped.most_common(25)}


def _first_existing(root: Path, names: list[str]) -> Path:
    for name in names:
        if (root / name).exists():
            return root / name
    raise FileNotFoundError(f"No encontré {names} dentro de {root}")


def harmonize(
    pad_root: Path | None,
    scin_root: Path | None,
    out_dir: Path,
    scin_benign_as_bajo: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Crea manifest.csv y harmonize_report.json en `out_dir`."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames, report = [], {}
    if pad_root:
        df, dropped = load_pad_ufes(Path(pad_root))
        frames.append(df)
        report["pad_ufes_20"] = {"descartados": dropped}
    if scin_root:
        df, info = load_scin(Path(scin_root), benign_as_bajo=scin_benign_as_bajo)
        frames.append(df)
        report["scin"] = info
    if not frames:
        raise ValueError("Indica al menos una fuente: PAD-UFES-20 o SCIN.")
    manifest = pd.concat(frames, ignore_index=True)
    if manifest.empty:
        raise ValueError("El manifiesto quedó vacío. Revisa las rutas y el mapeo.")
    dupes = manifest.sample_id.duplicated().sum()
    if dupes:
        manifest = manifest.drop_duplicates("sample_id")
    report["duplicados_eliminados"] = int(dupes)
    report["conteo_fuente_riesgo"] = {
        f"{s}|{r}": int(n)
        for (s, r), n in manifest.groupby(["source", "risk"]).size().items()
    }
    report["pacientes_o_casos"] = int(manifest.group_key.nunique())
    manifest.to_csv(out_dir / "manifest.csv", index=False)
    (out_dir / "harmonize_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False)
    )
    return manifest
