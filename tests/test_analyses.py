import sys, pathlib, json, subprocess
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src")); sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np, pandas as pd, pytest
import analyses as AN, final as FN, cv
from utils import load_config

CFG = load_config(); PARAMS = {"logistic": {"C": 0.05}, "lightgbm": {"n_estimators": 40}}; SMAP = {"logistic": "undersample", "lightgbm": "undersample"}

def test_ablation_shows_that_history_carries_the_planted_signal(fake_rs):
    t = AN.ablation(fake_rs["feats"], CFG, PARAMS["logistic"], "undersample", B=30).set_index("variant")
    assert len(t) == 9 and t.loc["full", "n_features"] == 26 and t.loc["full", "ap_change_vs_full"] != t.loc["full", "ap_change_vs_full"]       # no change reported for the reference itself (NaN)
    assert t.loc["without all history", "ap_change_vs_full"] < 0 and t.loc["history only", "ap"] > 0.6 * t.loc["full", "ap"]
    assert (t["ap_lo"] <= t["ap"]).all() and (t["ap"] <= t["ap_hi"]).all()

def test_label_sensitivity_changes_the_label_but_leaves_the_warehouse_untouched(fake_rs):
    conn = fake_rs["conn"]; before = conn.execute("SELECT COUNT(*), SUM(matched_qty) FROM bridge_return_match").fetchone(); db = fake_rs["dir"] / "w.db"; conn.commit()
    t = AN.label_sensitivity(db, CFG, PARAMS["logistic"], "undersample", windows=(60, 90, 120), B=20)
    assert list(t.label_definition)[0].startswith("window 60") and len(t) == 4 and t.positive_lines_dev.is_monotonic_increasing is False or t.positive_lines_dev.iloc[0] <= t.positive_lines_dev.iloc[2]
    assert t.loc[0, "positive_lines_dev"] <= t.loc[1, "positive_lines_dev"] <= t.loc[2, "positive_lines_dev"] and (t.lift > 1.0).all()
    assert conn.execute("SELECT COUNT(*), SUM(matched_qty) FROM bridge_return_match").fetchone() == before

@pytest.fixture(scope="module")
def scored(fake_rs, tmp_path_factory):
    d = tmp_path_factory.mktemp("scored"); art = d / "a"; art.mkdir()
    for k, v in {"best_params.json": PARAMS, "strategy_map.json": SMAP, "model_choice.json": {"chosen": "logistic"}, "calibration_choice.json": {"method": "analytic"}}.items(): (art / k).write_text(json.dumps(v))
    oof = cv.run_cv(fake_rs["feats"], CFG, ["logistic", "lightgbm"], SMAP, 1, PARAMS, verbose=False); r = FN.run_final(fake_rs["feats"], CFG, art, d / "p", oof, B=10)
    return {"frame": r["frame"], "dir": d}

def test_audit_slices_partition_the_test_lines_and_report_calibration(fake_rs, scored):
    t, errs = AN.audit(scored["frame"], fake_rs["feats"], "logistic_calibrated", fake_rs["dir"] / "w.db", top_n=5); s = t.set_index("slice")
    n = s.loc["all test lines", "lines"]; assert n == len(scored["frame"])
    assert s.loc["customer: first invoice in the data", "lines"] + s.loc["customer: earlier invoices, none old enough to know", "lines"] + s.loc["customer: some history old enough to know", "lines"] == n
    assert s.loc["country: United Kingdom", "lines"] + s.loc["country: other", "lines"] == n and sum(v for k, v in s.lines.items() if k.startswith("month ")) == n
    assert s.loc["all test lines", "observed_rate"] == pytest.approx(scored["frame"].y.mean()) and len(errs) == 10 and errs.description_clean.notna().all() and set(errs.kind) == {"false alarm: highest score, not returned", "miss: lowest score, returned"}

def test_drift_table_has_dev_and_test_months_and_predictions_only_for_test(fake_rs, scored):
    t = AN.drift(fake_rs["feats"], scored["frame"], "logistic_calibrated")
    assert set(t.split) == {"dev", "test"} and t[t.split == "dev"].mean_predicted.isna().all() and t[t.split == "test"].mean_predicted.notna().all()
    assert t.base_rate.between(0, 1).all()

def test_analyses_run_from_the_command_line(fake_rs, scored, tmp_path):
    f = fake_rs["feats"]; f.to_pickle(tmp_path / "f.pkl"); scored["frame"].to_pickle(tmp_path / "t.pkl"); art = tmp_path / "a"; art.mkdir()
    for k, v in {"best_params.json": PARAMS, "strategy_map.json": SMAP, "model_choice.json": {"chosen": "logistic"}}.items(): (art / k).write_text(json.dumps(v))
    base = [sys.executable, str(pathlib.Path(__file__).resolve().parents[1] / "src" / "analyses.py")]
    common = ["--features", str(tmp_path / "f.pkl"), "--artifacts", str(art), "--db", str(fake_rs["dir"] / "w.db"), "--test-preds", str(tmp_path / "t.pkl"), "--boot", "20"]
    for what, marker in (("ablation", "Feature-group ablation"), ("audit", "Largest errors"), ("drift", "Return rate by month")):
        r = subprocess.run(base + [what] + common, capture_output=True, text=True, cwd=str(pathlib.Path(__file__).resolve().parents[1] / "src")); assert r.returncode == 0 and marker in r.stdout, (what, r.stderr[-1200:])
