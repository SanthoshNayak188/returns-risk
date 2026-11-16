"""Step 1 of the model choice: which imbalance strategy to use for each model. RULE (fixed on <date of this run>, AFTER seeing the untuned results,
and recorded as such in the decision log): take the strategy with the highest mean fold average precision; use it only if its advantage over
'none' is larger than one standard error of the paired fold differences, otherwise use 'none'. Reads the files written by
`run_cv.py --compare-strategies`. Usage: python src/choose_strategy.py [--models logistic,lightgbm]"""
import argparse, json, pathlib
import numpy as np, pandas as pd
from utils import ROOT
from metrics import average_precision, shuffled

STRATEGIES = ["none", "weight", "undersample"]

def fold_ap(oof, models):
    return pd.DataFrame({m: {f: average_precision(g["y"].to_numpy(), g[m].to_numpy(float)) for f, g in shuffled(oof).groupby("fold")} for m in models})

def choose(ap_by_strategy, model):
    """ap_by_strategy: {strategy: DataFrame(folds x models)}. Returns (chosen strategy, details)."""
    means = {s: float(ap_by_strategy[s][model].mean()) for s in ap_by_strategy}; best = max(means, key=means.get)
    d = {"mean_fold_ap": means, "best": best}
    if best == "none": d["chosen"] = "none"; return "none", d
    diff = ap_by_strategy[best][model] - ap_by_strategy["none"][model]; se = float(diff.std(ddof=1) / np.sqrt(len(diff)))
    d.update({"advantage_over_none": float(diff.mean()), "standard_error": se, "folds_where_better": int((diff > 0).sum()), "folds": int(len(diff))})
    d["chosen"] = best if diff.mean() > se else "none"; return d["chosen"], d

def main():
    ap = argparse.ArgumentParser(); P = ROOT / "data" / "processed"
    ap.add_argument("--models", default="logistic,lightgbm"); ap.add_argument("--dir", default=str(P)); ap.add_argument("--out", default=str(ROOT / "artifacts" / "strategy_map.json"))
    a = ap.parse_args(); models = [m for m in a.models.split(",") if m]
    tabs = {s: fold_ap(pd.read_pickle(pathlib.Path(a.dir) / ("cv_oof_%s.pkl" % s)), models) for s in STRATEGIES}
    chosen = {}
    for m in models:
        c, d = choose(tabs, m); chosen[m] = c
        print("%s: mean fold AP %s  ->  best %s" % (m, {k: round(v, 4) for k, v in d["mean_fold_ap"].items()}, d["best"]))
        if "advantage_over_none" in d: print("   advantage over none %.4f, standard error %.4f, better in %d of %d folds" % (d["advantage_over_none"], d["standard_error"], d["folds_where_better"], d["folds"]))
        print("   CHOSEN:", c)
    out = pathlib.Path(a.out); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(chosen, indent=2)); print("Saved", out)

if __name__ == "__main__":
    main()
