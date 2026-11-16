import sys, pathlib, json, subprocess
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src")); sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np, pandas as pd, pytest
import choose_strategy as CS, select_model as SM, tune, cv
from utils import load_config

CFG = load_config(); SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
F = lambda **c: pd.DataFrame(c, index=["V1", "V2", "V3", "V4"])

def test_strategy_rule_keeps_none_unless_the_gain_beats_one_standard_error():
    tabs = {"none": F(m=[.05, .09, .08, .04]), "weight": F(m=[.0505, .0905, .0795, .0402]), "undersample": F(m=[.05, .09, .08, .04])}
    assert CS.choose(tabs, "m")[0] == "none"                                                   # tiny, inconsistent gain
    tabs["undersample"] = F(m=[.06, .10, .09, .05])
    c, d = CS.choose(tabs, "m"); assert c == "undersample" and d["folds_where_better"] == 4     # consistent gain in every fold

def test_strategy_rule_picks_none_when_it_is_best():
    tabs = {"none": F(m=[.06, .1, .09, .05]), "weight": F(m=[.05, .09, .08, .04]), "undersample": F(m=[.05, .09, .08, .04])}
    assert CS.choose(tabs, "m")[0] == "none"

def test_model_rule_prefers_the_simpler_model_inside_one_standard_error():
    tab = F(logistic=[.045, .098, .083, .036], lightgbm=[.050, .100, .085, .040])
    c, d = SM.choose_model(tab); assert d["best_model"] == "lightgbm" and c == "logistic"
    tab = F(logistic=[.04, .06, .05, .03], lightgbm=[.06, .09, .08, .05]); c, d = SM.choose_model(tab); assert c == "lightgbm" and d["folds_won_by_best"]["logistic"] == 4

def test_fold_table_averages_lightgbm_seeds():
    oof = pd.DataFrame({"fold": ["V1"] * 200 + ["V2"] * 200, "y": ([1] * 20 + [0] * 180) * 2, "logistic": np.tile(np.r_[np.ones(20), np.zeros(180)], 2),
                        "lightgbm": np.tile(np.r_[np.ones(20), np.zeros(180)], 2), "lightgbm__seed2": np.tile(np.r_[np.zeros(20), np.ones(180)], 2)})
    t = SM.fold_ap_table(oof, ["logistic", "lightgbm"]); assert np.allclose(t["logistic"], 1.0) and (t["lightgbm"] < 1.0).all()

def test_tuning_stays_in_range_is_reproducible_and_ignores_the_test_period(fake_rs):
    f = fake_rs["feats"]; a = tune.tune_model(f, CFG, "logistic", "none", 1, 3); b = tune.tune_model(f, CFG, "logistic", "none", 1, 3)
    assert 1e-4 <= a["best_params"]["C"] <= 10 and a["trial_values"] == b["trial_values"] and np.isclose(a["best_value"], max(a["trial_values"]))
    t = f.copy(); t.loc[t.split == "test", "y"] = 1 - t.loc[t.split == "test", "y"]; assert tune.tune_model(t, CFG, "logistic", "none", 1, 3)["trial_values"] == a["trial_values"]

def test_lightgbm_search_runs_and_customer_sample_keeps_whole_customers(fake_rs):
    f = fake_rs["feats"]; r = tune.tune_model(f, CFG, "lightgbm", "undersample", 1, 2, customer_fraction=0.6)
    assert 4 <= r["best_params"]["num_leaves"] <= 64 and 50 <= r["best_params"]["min_child_samples"] <= 1000
    s = tune.sample_customers(f, 0.5, 42); full = f.groupby("customer_key").size()
    sizes = s.groupby("customer_key").size(); assert (sizes == full.loc[sizes.index]).all() and 0.4 < s.customer_key.nunique() / f.customer_key.nunique() < 0.6

def test_strategy_map_and_seed_table_work_from_the_command_line(fake_rs, tmp_path):
    f = fake_rs["feats"]; f.to_pickle(tmp_path / "f.pkl"); (tmp_path / "map.json").write_text(json.dumps({"logistic": "weight", "lightgbm": "undersample"}))
    (tmp_path / "p.json").write_text(json.dumps({"lightgbm": {"n_estimators": 40}, "logistic": {"C": 0.5}}))
    r = subprocess.run([sys.executable, str(SRC / "run_cv.py"), "--features", str(tmp_path / "f.pkl"), "--models", "logistic,lightgbm", "--strategy-map", str(tmp_path / "map.json"),
                        "--params", str(tmp_path / "p.json"), "--extra-seeds", "2,3", "--boot", "20", "--out-dir", str(tmp_path / "a"), "--oof", str(tmp_path / "oof.pkl")], capture_output=True, text=True, cwd=str(SRC))
    assert r.returncode == 0, r.stderr[-1500:]; assert "Seed stability" in r.stdout
    oof = pd.read_pickle(tmp_path / "oof.pkl"); assert {"lightgbm__seed2", "lightgbm__seed3"} <= set(oof.columns) and "logistic__seed2" not in oof.columns
    r = subprocess.run([sys.executable, str(SRC / "select_model.py"), "--oof", str(tmp_path / "oof.pkl"), "--out", str(tmp_path / "choice.json")], capture_output=True, text=True, cwd=str(SRC))
    assert r.returncode == 0 and "CHOSEN MODEL" in r.stdout
