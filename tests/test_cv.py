import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src")); sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import cv, metrics as M
from models import Model, MODEL_NAMES
from utils import load_config

CFG = load_config(); DAY = pd.Timedelta(days=1)

def test_folds_respect_the_embargo_and_never_touch_the_test_period(fake_rs):
    f = fake_rs["feats"]; seen = 0
    for name, tr, va in cv.fold_indices(f, CFG):
        fold = next(x for x in CFG["cv"]["folds"] if x["name"] == name); seen += 1
        assert len(tr) > 0 and len(va) > 0 and not set(tr) & set(va)
        assert f.iloc[tr].ts.max() < pd.Timestamp(fold["val_start"]) - CFG["split"]["embargo_days"] * DAY
        assert f.iloc[va].ts.min() >= pd.Timestamp(fold["val_start"]) and f.iloc[va].ts.max() < pd.Timestamp(fold["val_end"]) + DAY
        assert f.iloc[tr].split.eq("dev").all() and f.iloc[va].split.eq("dev").all() and f.iloc[tr].is_final.eq(1).all()
    assert seen == 4

def test_a_constant_score_has_average_precision_equal_to_the_base_rate(fake_rs):
    f = fake_rs["feats"]; name, tr, va = next(cv.fold_indices(f, CFG))
    s = Model("B0_prior").fit(f.iloc[tr], f.iloc[tr].y.to_numpy()).predict(f.iloc[va]); y = f.iloc[va].y.to_numpy()
    assert np.isclose(M.average_precision(y, s), y.mean())

def test_models_beat_chance_when_a_real_signal_exists(fake_rs):
    f = fake_rs["feats"]; oof = M.shuffled(cv.run_cv(f, CFG, ["B0_prior", "B1b_product_rate", "logistic", "lightgbm"], "none", 1, {"lightgbm": {"n_estimators": 80}}, verbose=False))
    base = oof.y.mean(); ap = {m: M.average_precision(oof.y.to_numpy(), oof[m].to_numpy()) for m in ["B0_prior", "B1b_product_rate", "logistic", "lightgbm"]}
    assert abs(ap["B0_prior"] - base) < 0.2 * base and ap["logistic"] > 1.3 * base and ap["lightgbm"] > 1.3 * base and ap["B1b_product_rate"] > 1.1 * base
    assert oof[["logistic", "lightgbm"]].min().min() >= 0 and oof[["logistic", "lightgbm"]].max().max() <= 1

def test_imbalance_strategies_run_and_undersampling_inflates_probabilities(fake_rs):
    f = fake_rs["feats"]; name, tr, va = next(cv.fold_indices(f, CFG)); y = f.iloc[tr].y.to_numpy(); out = {}
    for st in ("none", "weight", "undersample"):
        out[st] = Model("logistic", st, 1).fit(f.iloc[tr], y).predict(f.iloc[va]); assert np.isfinite(out[st]).all()
    assert out["undersample"].mean() > 1.4 * out["none"].mean() and out["weight"].mean() > 1.4 * out["none"].mean()     # probabilities shift up; ranking is what is preserved

def test_undersampling_touches_only_the_training_rows(fake_rs):
    f = fake_rs["feats"]; name, tr, va = next(cv.fold_indices(f, CFG)); y = f.iloc[tr].y.to_numpy()
    m = Model("logistic", "undersample", 1); X, yy = m._resample(f.iloc[tr][m.features], y)
    assert len(yy) < len(y) and yy.sum() == y.sum() and (yy == 0).sum() == 5 * y.sum()

def test_permuted_label_check_passes_on_a_clean_pipeline(fake_rs):
    r = cv.permuted_label_check(fake_rs["feats"], CFG, 1); assert r["passed"] and r["finds_signal"], r
    assert r["shuffled_ap"] < 1.5 * r["base_rate"] and r["real_ap"] > 1.3 * r["base_rate"]

def test_permuted_label_check_fails_if_labels_leak_through_a_side_channel(fake_rs, monkeypatch):
    class Cheater:
        def __init__(self, *a, **k): pass
        def fit(self, df, y): return self
        def predict(self, df): return df["y"].to_numpy(float)          # reads the true label directly
    monkeypatch.setattr(cv, "Model", Cheater, raising=False)
    import models; monkeypatch.setattr(models, "Model", Cheater)
    r = cv.permuted_label_check(fake_rs["feats"], CFG, 1); assert not r["passed"]
