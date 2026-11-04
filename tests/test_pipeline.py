"""Pruebas del pipeline con datos sintéticos. No usan datos clínicos reales."""
from __future__ import annotations

import io
import json

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from dermatriage import embeddings as emb
from dermatriage.config import PAD_TO_RISK, RISK_LEVELS, normalize_label
from dermatriage.evaluate import evaluate, wilson_interval
from dermatriage.harmonize import harmonize, load_pad_ufes, load_scin
from dermatriage.predict import RiskClassifier
from dermatriage.splits import assert_no_leakage, make_splits
from dermatriage.train import choose_alto_threshold, decide, run_training

DIM = 64


def png_bytes(color=(120, 80, 60), size=(32, 32)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def synthetic(n_groups=360, seed=0):
    rng = np.random.default_rng(seed)
    means = rng.normal(0, 1.0, size=(3, DIM))
    rows, vecs = [], []
    for g in range(n_groups):
        risk = rng.choice(3, p=[0.5, 0.35, 0.15])
        source = "pad_ufes_20" if g % 2 else "scin"
        for k in range(rng.integers(1, 4)):
            sid = f"{source}/{g}_{k}"
            rows.append({"sample_id": sid, "source": source, "image_path": "",
                         "group_key": f"{source}/{g}", "raw_label": "x",
                         "risk": RISK_LEVELS[risk], "fitzpatrick": int(rng.integers(1, 7))})
            vecs.append(means[risk] * 1.5 + rng.normal(0, 1.0, DIM))
    return pd.DataFrame(rows), np.stack(vecs).astype(np.float32)


# ---------- configuración ----------
def test_pad_mapping_is_the_approved_decision():
    assert PAD_TO_RISK == {"MEL": "alto", "BCC": "medio", "SCC": "medio",
                           "ACK": "medio", "NEV": "bajo", "SEK": "bajo"}


def test_normalize_label():
    assert normalize_label("SK/ISK") == "skisk"
    assert normalize_label("SCC/SCCIS") == "sccsccis"
    assert normalize_label("Basal Cell Carcinoma") == "basalcellcarcinoma"


# ---------- armonización ----------
def test_load_pad_ufes(tmp_path):
    (tmp_path / "images" / "imgs_part_1").mkdir(parents=True)
    for name in ("a.png", "b.png"):
        (tmp_path / "images" / "imgs_part_1" / name).write_bytes(png_bytes())
    pd.DataFrame({
        "patient_id": ["P1", "P1", "P2", "P3"],
        "lesion_id": [1, 2, 3, 4],
        "img_id": ["a.png", "b.png", "zz.png", "a.png"],
        "diagnostic": ["MEL", "nev", "BCC", "XXX"],
        "fitspatrick": [2.0, None, 3.0, 1.0],
    }).to_csv(tmp_path / "metadata.csv", index=False)
    df, dropped = load_pad_ufes(tmp_path)
    assert sorted(df.risk) == ["alto", "bajo"]
    assert dropped == {"imagen_no_encontrada": 1, "etiqueta_sin_mapeo": 1}
    assert df.loc[df.raw_label == "MEL", "fitzpatrick"].iloc[0] == 2


def _scin_files(root):
    pd.DataFrame({
        "case_id": ["1", "2", "3"],
        "image_1_path": ["dataset/images/a.png", "dataset/images/c.png", "dataset/images/d.png"],
        "image_2_path": ["dataset/images/b.png", None, None],
    }).to_csv(root / "scin_cases.csv", index=False)
    ok, bad = "DEFAULT_YES_IMAGE_QUALITY_SUFFICIENT", "NO"
    pd.DataFrame({
        "case_id": ["1", "2", "3"],
        "dermatologist_gradable_for_skin_condition_1": [ok, ok, bad],
        "dermatologist_skin_condition_on_label_name": ["['Eczema', 'Melanoma']", "['Eczema']", "['Melanoma']"],
        "dermatologist_skin_condition_confidence": ["[0.9, 0.7]", "[0.8]", "[0.9]"],
        "dermatologist_fitzpatrick_skin_type_label_1": ["FST3", None, None],
    }).to_csv(root / "scin_labels.csv", index=False)


def test_load_scin_default_and_benign_option(tmp_path):
    _scin_files(tmp_path)
    df, info = load_scin(tmp_path)
    assert len(df) == 2 and set(df.risk) == {"alto"}  # Gana el riesgo más alto del caso 1.
    assert df.group_key.nunique() == 1
    assert info["descartados"]["calidad_insuficiente"] == 1
    assert info["descartados"]["etiqueta_sin_mapeo"] == 1
    assert ("Eczema", 2) in info["sin_mapeo_top"]
    df2, _ = load_scin(tmp_path, benign_as_bajo=("Eczema",))
    assert len(df2) == 3 and "bajo" in set(df2.risk)


def test_harmonize_writes_manifest_and_report(tmp_path):
    _scin_files(tmp_path)
    out = tmp_path / "out"
    m = harmonize(None, tmp_path, out)
    assert (out / "manifest.csv").exists()
    report = json.loads((out / "harmonize_report.json").read_text())
    assert report["conteo_fuente_riesgo"]["scin|alto"] == len(m)
    with pytest.raises(ValueError):
        harmonize(None, None, out)


# ---------- particiones ----------
def test_splits_have_no_leakage_and_all_classes():
    df, _ = synthetic()
    out = make_splits(df)
    assert_no_leakage(out)
    assert set(out.split) == {"train", "val", "test"}
    for s in ("train", "val", "test"):
        assert set(out[out.split == s].risk) == set(RISK_LEVELS)
    # Forzar una fuga: un grupo con varias muestras en dos particiones.
    multi = out.groupby("group_key").size()
    group = multi[multi > 1].index[0]
    idx = out.index[out.group_key == group]
    bad = out.copy()
    bad.loc[idx[0], "split"] = "train"
    bad.loc[idx[1], "split"] = "test"
    with pytest.raises(AssertionError):
        assert_no_leakage(bad)


# ---------- decisión y umbral ----------
def test_decide_upgrades_to_alto_by_threshold():
    proba = np.array([[0.60, 0.25, 0.15], [0.10, 0.80, 0.10], [0.50, 0.20, 0.30]])
    assert decide(proba, 0.30).tolist() == [0, 1, 2]
    assert decide(proba, 0.90).tolist() == [0, 1, 0]


def test_choose_threshold_meets_target_and_is_highest():
    p = np.array([0.9, 0.8, 0.6, 0.4, 0.3, 0.2, 0.1])
    is_alto = np.array([1, 1, 1, 1, 0, 0, 0], dtype=bool)
    t = choose_alto_threshold(p, is_alto, target=0.75)
    assert t == 0.6 and (p[is_alto] >= t).mean() >= 0.75
    with pytest.raises(ValueError):
        choose_alto_threshold(p, np.zeros(7, dtype=bool), 0.9)


# ---------- evaluación ----------
def test_evaluate_known_confusion():
    y_true = [2, 2, 2, 2, 1, 1, 0, 0, 0, 0]
    y_pred = [2, 2, 2, 0, 1, 2, 0, 0, 0, 1]
    r = evaluate(y_true, y_pred)
    assert r["por_clase"]["alto"]["sensibilidad"] == 0.75
    assert r["errores_criticos_alto_como_bajo"] == 1
    assert r["concordancia"] == 0.7 and r["cumple_meta_concordancia"] is False
    assert r["por_clase"]["bajo"]["n"] == 4


def test_wilson_interval_bounds():
    lo, hi = wilson_interval(9, 10)
    assert 0.5 < lo < 0.9 < hi <= 1.0
    assert wilson_interval(0, 0) == (None, None)


# ---------- entrenamiento + inferencia ----------
def test_training_end_to_end_and_prediction(tmp_path):
    df, X = synthetic()
    report = run_training(df, X, tmp_path)
    for name in ("model.joblib", "report.json", "confusion_matrix.png", "splits.csv", "model_card.json"):
        assert (tmp_path / name).exists()
    assert report["concordancia"] > 0.70
    assert report["por_clase"]["alto"]["sensibilidad"] >= 0.70
    assert set(report["por_fuente"]) == {"pad_ufes_20", "scin"}
    clf = RiskClassifier.load(tmp_path / "model.joblib")
    out = clf.classify_embedding(X[0])
    assert out["riesgo"] in RISK_LEVELS
    assert abs(sum(out["probabilidades"].values()) - 1) < 1e-6
    assert out["recomendacion"] and out["advertencia"]


def test_classify_bytes_uses_injected_embedder(tmp_path):
    df, X = synthetic()
    run_training(df, X, tmp_path)

    class Fake:
        name = "fake"

        def embed_bytes(self, b):
            return X[0]

    out = RiskClassifier.load(tmp_path / "model.joblib", embedder=Fake()).classify_bytes(png_bytes())
    assert out["riesgo"] in RISK_LEVELS


def test_training_rejects_missing_class(tmp_path):
    df, X = synthetic()
    keep = (df.risk != "alto").to_numpy()
    with pytest.raises(ValueError):
        run_training(df[keep].reset_index(drop=True), X[keep], tmp_path)


# ---------- embeddings ----------
def test_to_png_bytes_resizes_to_448():
    img = Image.open(io.BytesIO(emb.to_png_bytes(png_bytes(size=(50, 20)))))
    assert img.format == "PNG" and img.size == (448, 448)


def test_build_embeddings_prefers_precomputed_and_reports_missing(tmp_path):
    local = tmp_path / "x.png"
    local.write_bytes(png_bytes())
    manifest = pd.DataFrame({
        "sample_id": ["pre", "loc", "gone", "bad"],
        "image_path": ["", str(local), "", str(tmp_path / "nope.png")],
    })

    class Fake:
        name = "fake"

        def embed_bytes(self, b):
            return np.ones(DIM, dtype=np.float32)

    rep = emb.build_embeddings(manifest, tmp_path / "e.npz", Fake(),
                               {"pre": np.zeros(DIM, dtype=np.float32)})
    assert rep == {"total": 4, "con_embedding": 2, "precalculados": 1, "omitidos": 2}
    loaded = emb.load_embeddings(tmp_path / "e.npz")
    assert set(loaded) == {"pre", "loc"} and loaded["loc"].sum() == DIM
    with pytest.raises(RuntimeError):
        emb.build_embeddings(manifest.iloc[[2]], tmp_path / "e2.npz", Fake(), {})


# ---------- API ----------
def test_api_validations(monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    pytest.importorskip("multipart")
    from fastapi.testclient import TestClient

    from dermatriage import serve

    class FakeClf:
        def classify_bytes(self, b):
            if b == b"bad":
                raise ValueError("ilegible")
            return {"riesgo": "bajo"}

    monkeypatch.setattr(serve, "get_classifier", lambda: FakeClf())
    c = TestClient(serve.app)
    assert c.get("/health").json() == {"estado": "ok"}
    ok = c.post("/classify", files={"file": ("a.png", png_bytes(), "image/png")})
    assert ok.status_code == 200 and ok.json()["riesgo"] == "bajo"
    assert c.post("/classify", files={"file": ("a.txt", b"x", "text/plain")}).status_code == 415
    assert c.post("/classify", files={"file": ("a.png", b"", "image/png")}).status_code == 400
    assert c.post("/classify", files={"file": ("a.png", b"bad", "image/png")}).status_code == 422
