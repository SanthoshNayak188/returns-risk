"""THE SINGLE SCORED RUN on the test period (purchases 1 April to 10 September 2011). Run it once, after every choice is fixed and committed.
Everything is fitted on the development period only; the test labels are used only to score. A lock file blocks reruns.
Usage: python src/final.py [--boot 1000]"""
import argparse, hashlib, json, pathlib, datetime
import numpy as np, pandas as pd, joblib
from utils import load_config, ROOT
from models import Model
from calibrate import Calibrator
from metrics import metric_row, shuffled, cluster_bootstrap, cluster_bootstrap_scalar, expected_calibration_error, log_loss, recall_precision_at_k
from features import FEATURES

BASELINES = ["B0_prior", "B1a_customer_rate", "B1b_product_rate", "B2_logistic_small"]; MODELS = ["logistic", "lightgbm"]
RULES = ("B1a_customer_rate", "B1b_product_rate"); REQUIRED = ["model_choice.json", "strategy_map.json", "calibration_choice.json", "best_params.json"]

def reliability(y, p, bins=10):
    order = np.argsort(p, kind="stable"); rows = []
    for i, ix in enumerate(np.array_split(order, bins), 1):
        rows.append({"bin": i, "lines": len(ix), "mean_predicted": float(p[ix].mean()), "observed_rate": float(y[ix].mean()), "positives": int(y[ix].sum())})
    return pd.DataFrame(rows)

def fit_and_predict(df, cfg, params, smap, seed):
    """Fit on development lines only; predict the test lines. The test labels are never passed to any model."""
    dev = df[(df.split == "dev") & (df.is_final == 1)]; test = df[(df.split == "test") & (df.is_final == 1)]
    fitted, preds = {}, {}
    for name in BASELINES + MODELS:
        strat = smap.get(name, "none") if name in MODELS else "none"
        m = Model(name, strat, seed, params.get(name), neg_per_pos=cfg["imbalance"]["undersample_negatives_per_positive"]).fit(dev, dev["y"].to_numpy())
        fitted[name] = m; preds[name] = m.predict(test.drop(columns=["y"]).assign(y=0))
    return dev, test, fitted, preds

def calibrators(fitted, oof, method):
    out = {}
    for m in MODELS:
        out[m] = Calibrator(method, factor=fitted[m].odds_factor_)
        if method in ("platt", "isotonic"): out[m].fit(oof[m].to_numpy(float), oof["y"].to_numpy())
    return out

def run_final(df, cfg, art_dir, proc_dir, oof, B=1000, seed=1, force=False):
    art = pathlib.Path(art_dir); proc = pathlib.Path(proc_dir); art.mkdir(parents=True, exist_ok=True); proc.mkdir(parents=True, exist_ok=True); lock = art / "test_lock.json"
    missing = [f for f in REQUIRED if not (art / f).exists()]
    if missing: raise SystemExit("Not ready: these decisions must exist before the test is scored: %s" % missing)
    if lock.exists() and not force: raise SystemExit("The test period was already scored (%s). Rerunning would turn the test into a second validation set. If a genuine bug forces a rerun, delete the lock file AND record why in decision_log.md." % lock)
    params = json.load(open(art / "best_params.json")); smap = json.load(open(art / "strategy_map.json")); chosen = json.load(open(art / "model_choice.json"))["chosen"]
    method = json.load(open(art / "calibration_choice.json"))["method"]; ks = list(cfg["eval"]["top_k"]) + [0.10]
    dev, test, fitted, preds = fit_and_predict(df, cfg, params, smap, seed); cals = calibrators(fitted, oof, method)
    for m in MODELS: preds[m + "_calibrated"] = cals[m].predict(preds[m])
    frame = pd.DataFrame({"line_key": test.line_key.to_numpy(), "customer_key": test.customer_key.to_numpy(), "product_key": test.product_key.to_numpy(), "ts": test.ts.to_numpy(), "y": test.y.to_numpy(), **preds})
    ev = shuffled(frame, 0); y = ev["y"].to_numpy(); names = list(preds)
    table = pd.DataFrame({n: metric_row(y, ev[n].to_numpy(float), ks, probability=n not in RULES) for n in names}).T
    boot = cluster_bootstrap(ev, BASELINES + MODELS, ["B1b_product_rate", "B2_logistic_small"], ks, B=B, seed=cfg["seed"])
    cal = {}
    for lab, col in (("uncalibrated", chosen), ("calibrated", chosen + "_calibrated")):
        p = ev[col].to_numpy(float); row = {"ece": expected_calibration_error(y, p), "brier": float(np.mean((p - y) ** 2)), "log_loss": log_loss(y, p), "mean_prediction": float(p.mean()), "observed_rate": float(y.mean())}
        for k, fn in (("ece", lambda f, c=col: expected_calibration_error(f.y.to_numpy(), f[c].to_numpy(float))), ("log_loss", lambda f, c=col: log_loss(f.y.to_numpy(), f[c].to_numpy(float)))):
            row[k + "_lo"], row[k + "_hi"] = cluster_bootstrap_scalar(ev, fn, B=min(B, 300), seed=cfg["seed"])
        cal[lab] = row
    cal = pd.DataFrame(cal).T
    rel_u, rel_c = reliability(y, ev[chosen].to_numpy(float)), reliability(y, ev[chosen + "_calibrated"].to_numpy(float))
    bands = []
    for k, (rec, prec) in recall_precision_at_k(y, ev[chosen + "_calibrated"].to_numpy(float), [0.01, 0.02, 0.05, 0.10, 0.20]).items():
        bands.append({"flag_top_share": k, "lines_flagged": int(np.ceil(k * len(y))), "precision": float(prec), "recall": float(rec), "lift": float(prec / y.mean())})
    bands = pd.DataFrame(bands)
    table.to_csv(art / "test_results.csv"); boot.to_csv(art / "test_bootstrap.csv", index=False); cal.to_csv(art / "test_calibration.csv"); bands.to_csv(art / "test_risk_bands.csv", index=False)
    rel_u.to_csv(art / "test_reliability_uncalibrated.csv", index=False); rel_c.to_csv(art / "test_reliability_calibrated.csv", index=False); frame.to_pickle(proc / "test_preds.pkl")
    joblib.dump({"model": fitted[chosen], "calibrator": cals[chosen], "calibration_method": method, "chosen": chosen, "features": list(FEATURES), "strategy": smap.get(chosen, "none"),
                 "params": params.get(chosen, {}), "trained_through": cfg["split"]["dev_end"], "training_lines": int(len(dev))}, proc / "model.pkl")
    lock.write_text(json.dumps({"scored_at": datetime.datetime.now().isoformat(timespec="seconds"), "chosen": chosen, "calibration": method, "test_lines": int(len(test)), "test_positives": int(test.y.sum()),
                                "config_sha256": hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()}, indent=2))
    return {"table": table, "boot": boot, "cal": cal, "bands": bands, "rel_u": rel_u, "rel_c": rel_c, "chosen": chosen, "method": method, "frame": frame}

def main():
    cfg = load_config(); P = ROOT / "data" / "processed"; ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=str(P / "features.pkl")); ap.add_argument("--oof", default=str(P / "cv_oof.pkl")); ap.add_argument("--artifacts", default=str(ROOT / "artifacts"))
    ap.add_argument("--processed", default=str(P)); ap.add_argument("--boot", type=int, default=1000); ap.add_argument("--seed", type=int, default=1); a = ap.parse_args()
    r = run_final(pd.read_pickle(a.features), cfg, a.artifacts, a.processed, pd.read_pickle(a.oof), a.boot, a.seed)
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
    t = r["table"]; ks = ["ap", "ap_lift", "roc_auc", "recall_at_2%", "recall_at_5%", "recall_at_10%", "precision_at_5%"]
    print("TEST RESULTS (scored once). Chosen model: %s, calibration: %s. Test lines %d, positives %d, base rate %.4f\n" % (r["chosen"], r["method"], len(r["frame"]), r["frame"].y.sum(), r["frame"].y.mean()))
    print(t[ks].round(4).to_string()); b = r["boot"]
    print("\nCustomer-cluster bootstrap, average precision with 95% intervals and paired differences:")
    print(b[["model", "ap", "ap_lo", "ap_hi"] + [c for c in b.columns if c.startswith("ap_diff")]].round(4).to_string(index=False))
    print("\nCalibration of %s on the test period:" % r["chosen"]); print(r["cal"].round(5).to_string()); print("\nReliability (ten equal-count bins), calibrated:"); print(r["rel_c"].round(4).to_string(index=False))
    print("\nWhat flagging the riskiest lines would catch:"); print(r["bands"].round(4).to_string(index=False))
    print("\nSaved results to artifacts/, predictions to data/processed/test_preds.pkl, model to data/processed/model.pkl. The lock file now blocks reruns.")

if __name__ == "__main__":
    main()
