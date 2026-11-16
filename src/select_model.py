"""Step 2 of the model choice, RULE FIXED IN ADVANCE (in the proposal): take the model with the highest mean fold average precision; every model
whose mean is within one standard error (across folds) of the best counts as equally good; among those choose the simplest (logistic regression
before LightGBM). The five-feature logistic regression is a baseline, not a candidate. Usage: python src/select_model.py [--oof ...] [--candidates logistic,lightgbm]"""
import argparse, json, pathlib
import numpy as np, pandas as pd
from utils import ROOT
from metrics import average_precision, shuffled

SIMPLICITY = ["logistic", "lightgbm"]

def fold_ap_table(oof, models):
    ev = shuffled(oof); cols = {}
    for m in models:
        runs = [m] + [c for c in oof.columns if c.startswith(m + "__seed")]
        cols[m] = pd.concat([pd.Series({f: average_precision(g["y"].to_numpy(), g[c].to_numpy(float)) for f, g in ev.groupby("fold")}) for c in runs], axis=1).mean(axis=1)
    return pd.DataFrame(cols)

def choose_model(fold_ap):
    mean = fold_ap.mean(); best = mean.idxmax(); se = fold_ap[best].std(ddof=1) / np.sqrt(len(fold_ap)); thr = mean[best] - se
    within = [m for m in fold_ap.columns if mean[m] >= thr]; order = {m: i for i, m in enumerate(SIMPLICITY)}
    chosen = sorted(within, key=lambda m: (order.get(m, 99), -mean[m]))[0]
    return chosen, {"best_model": best, "best_mean_ap": float(mean[best]), "standard_error": float(se), "threshold": float(thr), "mean_fold_ap": {m: float(v) for m, v in mean.items()},
                    "within_one_se": within, "folds_won_by_best": {m: int((fold_ap[best] > fold_ap[m]).sum()) for m in fold_ap.columns if m != best}}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--oof", default=str(ROOT / "data" / "processed" / "cv_oof.pkl")); ap.add_argument("--candidates", default="logistic,lightgbm")
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "model_choice.json")); a = ap.parse_args()
    tab = fold_ap_table(pd.read_pickle(a.oof), [m for m in a.candidates.split(",") if m]); chosen, d = choose_model(tab)
    print("Average precision per fold (seed-averaged for LightGBM):"); print(tab.round(4).to_string())
    print("\nBest model: %s, mean fold AP %.4f, standard error across folds %.4f; threshold %.4f" % (d["best_model"], d["best_mean_ap"], d["standard_error"], d["threshold"]))
    print("Models within one standard error:", d["within_one_se"]); print("Folds in which the best model beat each other model:", d["folds_won_by_best"], "of", len(tab))
    print("\nCHOSEN MODEL (simplest within one standard error): %s" % chosen)
    out = pathlib.Path(a.out); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps({"chosen": chosen, **d}, indent=2)); print("Saved", out)

if __name__ == "__main__":
    main()
