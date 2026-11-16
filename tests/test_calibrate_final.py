import sys, pathlib, json, subprocess
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src")); sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np, pandas as pd, pytest, joblib
import calibrate as C, final as FN, cv
from models import Model
from utils import load_config

CFG = load_config(); SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
PARAMS = {"lightgbm": {"n_estimators": 60}}; SMAP = {"logistic": "undersample", "lightgbm": "undersample"}

@pytest.fixture(scope="module")
def oof(fake_rs):
    return cv.run_cv(fake_rs["feats"], CFG, ["logistic", "lightgbm"], SMAP, 1, PARAMS, verbose=False)

def test_analytic_correction_undoes_a_known_shift():
    rng = np.random.default_rng(0); p = rng.uniform(0.001, 0.3, 5000); r = 0.12
    odds = p / (1 - p) / r; inflated = odds / (1 + odds)                       # what a model trained on r of the negatives would say
    assert np.allclose(C.analytic_correct(inflated, r), p, atol=1e-9)

def test_platt_is_monotone_and_isotonic_is_a_bounded_step_function():
    rng = np.random.default_rng(1); p = rng.uniform(0, 0.5, 4000); y = (rng.random(4000) < p / 3).astype(int)
    pl = C.Calibrator("platt").fit(p, y).predict(np.sort(p)); iso = C.Calibrator("isotonic").fit(p, y).predict(np.sort(p))
    assert (np.diff(pl) >= -1e-12).all() and (np.diff(iso) >= -1e-12).all() and iso.min() >= 0 and iso.max() <= 1
    assert abs(pl.mean() - y.mean()) < 0.01 and len(np.unique(iso)) < len(p)

def test_model_remembers_how_much_it_shifted_the_odds(fake_rs):
    f = fake_rs["feats"]; name, tr, va = next(cv.fold_indices(f, CFG)); y = f.iloc[tr].y.to_numpy()
    m = Model("logistic", "undersample", 1).fit(f.iloc[tr], y); assert np.isclose(m.odds_factor_, 5 * y.sum() / (len(y) - y.sum()))
    assert Model("logistic", "none", 1).fit(f.iloc[tr], y).odds_factor_ == 1.0 and Model("lightgbm", "weight", 1, {"n_estimators": 20}).fit(f.iloc[tr], y).odds_factor_ == pytest.approx(y.sum() / (len(y) - y.sum()))
    p = m.predict(f.iloc[va]); assert abs(m.prior_corrected(p).mean() - y.mean()) < 0.5 * y.mean() and p.mean() > 2 * m.prior_corrected(p).mean()

def test_every_correction_fixes_the_inflated_probabilities_and_the_choice_is_a_calibrating_method(fake_rs, oof):
    fac = C.fold_factors(fake_rs["feats"], CFG, "undersample"); tab = C.forward_chain_eval(oof, "logistic", fac)
    assert tab.loc["none", "ece"] > 0.04 and all(tab.loc[m, "ece"] < 0.5 * tab.loc["none", "ece"] for m in ("analytic", "platt", "isotonic"))
    assert tab.loc["platt", "ap"] == pytest.approx(tab.loc["none", "ap"], rel=0.1) and C.choose_method(tab) != "none"        # per fold Platt keeps the ranking; pooled over folds it can shift slightly

def test_choose_method_prefers_the_simpler_one_within_one_percent():
    t = pd.DataFrame({"log_loss": [0.20, 0.0998, 0.0995, 0.0990]}, index=["none", "analytic", "platt", "isotonic"])
    assert C.choose_method(t) == "analytic"
    t.loc["isotonic", "log_loss"] = 0.08; assert C.choose_method(t) == "isotonic"

def write_decisions(art, method="platt", chosen="logistic"):
    art.mkdir(parents=True, exist_ok=True)
    (art / "best_params.json").write_text(json.dumps(PARAMS)); (art / "strategy_map.json").write_text(json.dumps(SMAP))
    (art / "model_choice.json").write_text(json.dumps({"chosen": chosen})); (art / "calibration_choice.json").write_text(json.dumps({"method": method}))

def test_final_run_refuses_to_start_without_decisions_and_scores_only_once(fake_rs, oof, tmp_path):
    f = fake_rs["feats"]
    with pytest.raises(SystemExit, match="Not ready"): FN.run_final(f, CFG, tmp_path / "a", tmp_path / "p", oof, B=10)
    write_decisions(tmp_path / "a")
    r = FN.run_final(f, CFG, tmp_path / "a", tmp_path / "p", oof, B=15); assert (tmp_path / "a" / "test_lock.json").exists() and len(r["frame"]) > 1000
    with pytest.raises(SystemExit, match="already scored"): FN.run_final(f, CFG, tmp_path / "a", tmp_path / "p", oof, B=10)

def test_test_labels_never_reach_the_models(fake_rs):
    f = fake_rs["feats"]; _, _, _, p1 = FN.fit_and_predict(f, CFG, PARAMS, SMAP, 1)
    g = f.copy(); g.loc[g.split == "test", "y"] = 1 - g.loc[g.split == "test", "y"]; _, _, _, p2 = FN.fit_and_predict(g, CFG, PARAMS, SMAP, 1)
    for k in p1: assert np.allclose(p1[k], p2[k])

def test_final_results_are_sensible_and_the_saved_model_reproduces_them(fake_rs, oof, tmp_path):
    write_decisions(tmp_path / "a", method="platt"); r = FN.run_final(fake_rs["feats"], CFG, tmp_path / "a", tmp_path / "p", oof, B=20)
    t, c = r["table"], r["cal"]; base = r["frame"].y.mean()
    assert t.loc["logistic", "ap"] > 1.3 * base and t.loc["B0_prior", "ap"] < 1.3 * base
    assert c.loc["calibrated", "ece"] < 0.5 * c.loc["uncalibrated", "ece"] and abs(c.loc["calibrated", "mean_prediction"] - base) < 0.5 * base
    assert t.loc["logistic", "roc_auc"] == pytest.approx(t.loc["logistic_calibrated", "roc_auc"], abs=1e-9)
    assert r["bands"].recall.is_monotonic_increasing and r["rel_c"].observed_rate.iloc[-1] > r["rel_c"].observed_rate.iloc[0]
    saved = joblib.load(tmp_path / "p" / "model.pkl"); test = fake_rs["feats"][(fake_rs["feats"].split == "test") & (fake_rs["feats"].is_final == 1)]
    again = saved["calibrator"].predict(saved["model"].predict(test)); assert np.allclose(again, r["frame"]["logistic_calibrated"].to_numpy())
    assert saved["features"] == C.__dict__.get("FEATURES", saved["features"]) and saved["trained_through"] == CFG["split"]["dev_end"]

def test_calibration_and_final_run_from_the_command_line(fake_rs, oof, tmp_path):
    f = fake_rs["feats"]; f.to_pickle(tmp_path / "f.pkl"); oof.to_pickle(tmp_path / "oof.pkl"); art = tmp_path / "a"; write_decisions(art)
    r = subprocess.run([sys.executable, str(SRC / "calibrate.py"), "--features", str(tmp_path / "f.pkl"), "--oof", str(tmp_path / "oof.pkl"), "--strategy-map", str(art / "strategy_map.json"),
                        "--choice", str(art / "model_choice.json"), "--out-dir", str(art)], capture_output=True, text=True, cwd=str(SRC))
    assert r.returncode == 0 and "CHOSEN CALIBRATION" in r.stdout, r.stderr[-1200:]
    r = subprocess.run([sys.executable, str(SRC / "final.py"), "--features", str(tmp_path / "f.pkl"), "--oof", str(tmp_path / "oof.pkl"), "--artifacts", str(art), "--processed", str(tmp_path / "p"), "--boot", "15"],
                       capture_output=True, text=True, cwd=str(SRC))
    assert r.returncode == 0 and "TEST RESULTS (scored once)" in r.stdout and "What flagging the riskiest lines would catch" in r.stdout, r.stderr[-1500:]
    r = subprocess.run([sys.executable, str(SRC / "final.py"), "--features", str(tmp_path / "f.pkl"), "--oof", str(tmp_path / "oof.pkl"), "--artifacts", str(art), "--processed", str(tmp_path / "p"), "--boot", "15"],
                       capture_output=True, text=True, cwd=str(SRC))
    assert r.returncode != 0 and "already scored" in (r.stderr + r.stdout)
