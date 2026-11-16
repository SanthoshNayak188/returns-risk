"""Compare the baselines and models on the time-ordered folds (development period only) and print one honest table.
Usage: python src/run_cv.py [--models ...] [--strategy none|weight|undersample] [--compare-strategies] [--sanity]"""
import argparse, json, pathlib
import numpy as np, pandas as pd
from utils import load_config, ROOT
from cv import run_cv, permuted_label_check
from metrics import metric_row, shuffled, cluster_bootstrap
from models import MODEL_NAMES

def main():
    cfg = load_config(); ks = cfg["eval"]["top_k"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=str(ROOT / "data" / "processed" / "features.pkl")); ap.add_argument("--models", default=",".join(MODEL_NAMES))
    ap.add_argument("--strategy", default="none"); ap.add_argument("--compare-strategies", action="store_true"); ap.add_argument("--sanity", action="store_true")
    ap.add_argument("--params", default=None); ap.add_argument("--strategy-map", default=None, help="JSON {model: strategy}"); ap.add_argument("--extra-seeds", default=""); ap.add_argument("--seed", type=int, default=1); ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--out-dir", default=str(ROOT / "artifacts")); ap.add_argument("--oof", default=str(ROOT / "data" / "processed" / "cv_oof.pkl"))
    a = ap.parse_args(); df = pd.read_pickle(a.features); models = [m for m in a.models.split(",") if m]
    params = json.load(open(a.params)) if a.params else {}; out = pathlib.Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
    if a.sanity:
        r = permuted_label_check(df, cfg, a.seed)
        print("Permuted-label check: base rate %.4f, shuffled-label AP %.4f -> %s" % (r["base_rate"], r["shuffled_ap"], "PASS (no label leaks into the model)" if r["passed"] else "FAIL, investigate before trusting any result"))
        print("Signal check: real model AP %.4f is %.2f times the base rate -> %s\n" % (r["real_ap"], r["real_lift"], "the model finds signal" if r["finds_signal"] else "little or no signal beyond chance"))
    strategies = ["none", "weight", "undersample"] if a.compare_strategies else [a.strategy]
    smap = json.load(open(a.strategy_map)) if a.strategy_map else None; extra = [int(s) for s in a.extra_seeds.split(",") if s]
    for st in strategies:
        print("=== imbalance strategy: %s (development period, 4 time-ordered folds with a 90 day embargo) ===" % (", ".join("%s=%s" % kv for kv in smap.items()) if smap else st), flush=True)
        oof = run_cv(df, cfg, models, smap if smap else st, a.seed, params, extra_seeds=extra)
        ev = shuffled(oof, 0); prob = {m: m not in ("B1a_customer_rate", "B1b_product_rate") for m in models}
        tab = pd.DataFrame({m: metric_row(ev["y"], ev[m], ks, probability=prob[m]) for m in models}).T
        print("\nPooled over all folds (%d validation lines, base rate %.4f):" % (len(ev), ev.y.mean()))
        cols = ["ap", "ap_lift", "roc_auc"] + ["recall_at_%g%%" % (100 * k) for k in ks] + ["precision_at_%g%%" % (100 * ks[-1]), "brier", "log_loss", "ece"]
        print(tab[cols].round(4).to_string())
        seed_cols = {m: [m] + [c for c in oof.columns if c.startswith(m + "__seed")] for m in models}
        if any(len(v) > 1 for v in seed_cols.values()):
            print("\nSeed stability (pooled AP per seed):")
            print(pd.DataFrame({m: (lambda v: {"seeds": len(cs), "mean_ap": v.mean(), "sd": v.std(ddof=1), "min": v.min(), "max": v.max()})(np.array([metric_row(ev["y"], ev[c], ks, probability=False)["ap"] for c in cs])) for m, cs in seed_cols.items() if len(cs) > 1}).T.round(4).to_string())
        print("\nAverage precision by fold:"); print(pd.DataFrame({f: {m: metric_row(g["y"], g[m], ks, probability=False)["ap"] for m in models} for f, g in ev.groupby("fold")}).round(4).to_string())
        base = "B1b_product_rate" if "B1b_product_rate" in models else models[0]
        ci = cluster_bootstrap(ev, models, base, ks, B=a.boot, seed=cfg["seed"])
        print("\nCustomer-cluster bootstrap (%d resamples), 95%% intervals:" % a.boot)
        print(ci[["model", "ap", "ap_lo", "ap_hi"] + [c for c in ci.columns if c.startswith("ap_diff")]].round(4).to_string(index=False))
        tab.to_csv(out / ("cv_results_%s.csv" % st)); ci.to_csv(out / ("cv_bootstrap_%s.csv" % st), index=False)
        oof.to_pickle(pathlib.Path(a.oof).with_name("cv_oof_%s.pkl" % st) if a.compare_strategies else a.oof); print()

if __name__ == "__main__":
    main()
