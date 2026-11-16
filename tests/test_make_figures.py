import sys, pathlib, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src")); sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import pandas as pd
import analyses as AN, final as FN, cv, make_figures as MF
from utils import load_config
CFG = load_config(); PARAMS = {"logistic": {"C": 0.05}, "lightgbm": {"n_estimators": 40}}; SMAP = {"logistic": "undersample", "lightgbm": "undersample"}

def test_all_three_figures_are_drawn_from_real_outputs_and_missing_inputs_do_not_crash(fake_rs, tmp_path):
    art = tmp_path / "a"; art.mkdir()
    for k, v in {"best_params.json": PARAMS, "strategy_map.json": SMAP, "model_choice.json": {"chosen": "logistic"}, "calibration_choice.json": {"method": "analytic"}}.items(): (art / k).write_text(json.dumps(v))
    f = fake_rs["feats"]; oof = cv.run_cv(f, CFG, ["logistic", "lightgbm"], SMAP, 1, PARAMS, verbose=False); r = FN.run_final(f, CFG, art, tmp_path / "p", oof, B=10)
    AN.ablation(f, CFG, PARAMS["logistic"], "undersample", B=15).to_csv(art / "ablation.csv", index=False)
    AN.label_sensitivity(fake_rs["dir"] / "w.db", CFG, PARAMS["logistic"], "undersample", windows=(60, 90), B=10).to_csv(art / "label_sensitivity.csv", index=False)
    s, _ = AN.audit(r["frame"], f, "logistic_calibrated"); s.to_csv(art / "audit_slices.csv", index=False); AN.drift(f, r["frame"], "logistic_calibrated").to_csv(art / "drift_monthly.csv", index=False)
    pd.DataFrame({"window_days": [60, 90, 120, 180], "matched_credit_line_share": [0.78, 0.82, 0.85, 0.88]}).to_csv(art / "label_audit_windows.csv", index=False)
    paths = MF.make_all(art, fake_rs["dir"] / "w.db", tmp_path / "p" / "test_preds.pkl", tmp_path / "figs")
    assert len(paths) == 3 and all(p.exists() and p.stat().st_size > 5000 for p in paths)
    empty = MF.make_all(tmp_path / "nothing", None, None, tmp_path / "figs2"); assert all(p.exists() for p in empty)
    import shutil; shutil.copytree(tmp_path / "figs", "/tmp/figs_check", dirs_exist_ok=True)
