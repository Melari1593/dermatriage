"""Línea de comandos: harmonize -> embed -> train."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .embeddings import DermFoundationEmbedder, build_embeddings, load_precomputed_scin
from .harmonize import harmonize
from .train import load_xy, run_training


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="dermatriage")
    sub = p.add_subparsers(dest="cmd", required=True)

    h = sub.add_parser("harmonize", help="Fase 0: crea manifest.csv")
    h.add_argument("--pad", type=Path, help="Carpeta de PAD-UFES-20")
    h.add_argument("--scin", type=Path, help="Carpeta de SCIN (CSV e imágenes)")
    h.add_argument("--scin-benign-as-bajo", default="",
                   help="Condiciones SCIN separadas por coma que cuentan como riesgo bajo")
    h.add_argument("--out", type=Path, default=Path("artifacts"))

    e = sub.add_parser("embed", help="Calcula embeddings")
    e.add_argument("--manifest", type=Path, default=Path("artifacts/manifest.csv"))
    e.add_argument("--out", type=Path, default=Path("artifacts/embeddings.npz"))
    e.add_argument("--scin-npz", type=Path, help="npz precalculado de SCIN (si no, se descarga)")
    e.add_argument("--no-precomputed", action="store_true")

    t = sub.add_parser("train", help="Entrena y evalúa")
    t.add_argument("--manifest", type=Path, default=Path("artifacts/manifest.csv"))
    t.add_argument("--embeddings", type=Path, default=Path("artifacts/embeddings.npz"))
    t.add_argument("--out", type=Path, default=Path("artifacts"))
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--target-sensitivity", type=float, default=0.90)

    args = p.parse_args(argv)
    if args.cmd == "harmonize":
        benign = tuple(x.strip() for x in args.scin_benign_as_bajo.split(",") if x.strip())
        m = harmonize(args.pad, args.scin, args.out, scin_benign_as_bajo=benign)
        print(m.groupby(["source", "risk"]).size().to_string())
        print(f"Reporte: {args.out / 'harmonize_report.json'}")
    elif args.cmd == "embed":
        manifest = pd.read_csv(args.manifest)
        precomputed = {}
        if not args.no_precomputed and (manifest.source == "scin").any():
            precomputed = load_precomputed_scin(str(args.scin_npz) if args.scin_npz else None)
        needs_model = ~manifest.sample_id.isin(precomputed)
        embedder = DermFoundationEmbedder() if needs_model.any() else None
        print(json.dumps(build_embeddings(manifest, args.out, embedder, precomputed), indent=2))
    elif args.cmd == "train":
        manifest, X = load_xy(args.manifest, args.embeddings)
        report = run_training(manifest, X, args.out, seed=args.seed,
                              target_sensitivity=args.target_sensitivity)
        print(f"Concordancia (test): {report['concordancia']:.3f}")
        print(f"Sensibilidad 'alto': {report['por_clase']['alto']['sensibilidad']}")
        print(f"Errores críticos (alto->bajo): {report['errores_criticos_alto_como_bajo']}")


if __name__ == "__main__":
    main()
