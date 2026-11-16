"""Graduate analyses for the return-risk model.
  ablation   which feature groups matter (development folds only)
  labels     how sensitive the conclusions are to the label definition: window 60, 90, 120 days, and oldest-first versus most-recent-first matching (development folds only)
  audit      AFTER the test run: where the scored predictions fail (slices, largest errors); descriptive only, never used to change a choice
  drift      base rate by month and calibration by month
Usage: python src/analyses.py ablation|labels|audit|drift [--boot N]"""
import argparse, copy, json, pathlib, shutil, sqlite3, tempfile
import numpy as np, pandas as pd
from utils import load_config, ROOT
from cv import fold_indices
from features import FEATURE_GROUPS, FEATURES, build_features
from models import Model
from metrics import average_precision, roc_auc, recall_precision_at_k, shuffled, cluster_bootstrap, cluster_bootstrap_scalar
import match as M

HISTORY = FEATURE_GROUPS["customer_history"] + FEATURE_GROUPS["product_history"] + FEATURE_GROUPS["pair_history"]

def logistic_oof(df, cfg, features, params, strategy, seed=1):
    """Out-of-fold scores of the chosen kind of model (logistic regression) for a given feature list, on the time-ordered folds."""
    parts = []
    for name, tr, va in fold_indices(df, cfg):
        m = Model("logistic", strategy, seed, params, features=features, neg_per_pos=cfg["imbalance"]["undersample_negatives_per_positive"]).fit(df.iloc[tr], df.iloc[tr]["y"].to_numpy())
        o = df.iloc[va][["line_key", "customer_key", "y"]].copy(); o["score"] = m.predict(df.iloc[va]); o["fold"] = name; parts.append(o)
    return pd.concat(parts, ignore_index=True)

def ablation(df, cfg, params, strategy, B=200):
    variants = {"full": list(FEATURES)}
    for g, cols in FEATURE_GROUPS.items(): variants["without " + g] = [c for c in FEATURES if c not in cols]
    variants["without all history"] = [c for c in FEATURES if c not in HISTORY]; variants["history only"] = list(HISTORY)
    frame = None
    for v, cols in variants.items():
        o = logistic_oof(df, cfg, cols, params, strategy)
        if frame is None: frame = o[["line_key", "customer_key", "y", "fold"]].copy()
        frame[v] = o["score"].to_numpy()
    ev = shuffled(frame); ks = [0.05]; boot = cluster_bootstrap(ev, list(variants), "full", ks, B=B, seed=cfg["seed"])
    boot.insert(1, "n_features", [len(variants[m]) for m in boot.model]); boot = boot.rename(columns={"model": "variant", "ap_diff_vs_full": "ap_change_vs_full", "ap_diff_lo": "change_lo", "ap_diff_hi": "change_hi"})
    boot["ap_change_percent"] = 100 * boot["ap_change_vs_full"] / boot.loc[boot.variant == "full", "ap"].iloc[0]
    return boot[["variant", "n_features", "ap", "ap_lo", "ap_hi", "ap_change_vs_full", "change_lo", "change_hi", "ap_change_percent", "recall_at_5%"]]

def label_sensitivity(db_path, cfg, params, strategy, windows=(60, 90, 120), B=100):
    """Rebuild the label and the features for each definition on a COPY of the warehouse and rerun the development folds. Average precision depends on the base rate,
    so definitions are compared by lift (average precision divided by the base rate) and recall at 5 percent."""
    tmp = pathlib.Path(tempfile.mkdtemp()); src = sqlite3.connect(str(db_path)); work = sqlite3.connect(str(tmp / "w.db")); src.backup(work); src.close(); rows = []
    variants = [("window %d days, oldest purchase first" % w, w, "fifo") for w in windows] + [("window 90 days, most recent purchase first", 90, "lifo")]
    for label, w, rule in variants:
        M.run_matching(work, w, rule); c2 = copy.deepcopy(cfg); c2["label"]["window_days"] = w; c2["label"]["maturity_lag_days"] = w; c2["split"]["embargo_days"] = w
        df = build_features(work, c2, maturity_lag_days=w); o = shuffled(logistic_oof(df, c2, FEATURES, params, strategy)); y, s = o["y"].to_numpy(), o["score"].to_numpy()
        base = float(y.mean()); ap = average_precision(y, s); lo, hi = cluster_bootstrap_scalar(o, lambda f: average_precision(f.y.to_numpy(), f.score.to_numpy()) / f.y.mean(), B=B, seed=cfg["seed"])
        dev = df[(df.split == "dev") & (df.is_final == 1)]
        rows.append({"label_definition": label, "positive_lines_dev": int(dev.y.sum()), "base_rate": base, "ap": ap, "lift": ap / base, "lift_lo": lo, "lift_hi": hi,
                     "roc_auc": roc_auc(y, s), "recall_at_5%": float(recall_precision_at_k(y, s, [0.05])[0.05][0])})
    shutil.rmtree(tmp, ignore_errors=True); return pd.DataFrame(rows)

def slices(t):
    price_cut = t["log_price"].quantile([1 / 3, 2 / 3]).to_numpy(); m = t["ts"].dt.to_period("M").astype(str)
    s = {"all test lines": np.ones(len(t), bool),
         "customer: first invoice in the data": (t.cust_log_prior_lines == 0).to_numpy(),
         "customer: earlier invoices, none old enough to know": ((t.cust_log_prior_lines > 0) & (t.cust_log_mature_lines == 0)).to_numpy(),
         "customer: some history old enough to know": (t.cust_log_mature_lines > 0).to_numpy(),
         "product: no history old enough to know": (t.prod_log_mature_lines == 0).to_numpy(), "product: some history old enough to know": (t.prod_log_mature_lines > 0).to_numpy(),
         "country: United Kingdom": (t.is_uk == 1).to_numpy(), "country: other": (t.is_uk == 0).to_numpy(),
         "invoice with 1 to 5 lines": (t.n_lines_in_invoice <= 5).to_numpy(), "invoice with 6 to 20 lines": t.n_lines_in_invoice.between(6, 20).to_numpy(), "invoice with more than 20 lines": (t.n_lines_in_invoice > 20).to_numpy(),
         "unit price: lowest third": (t.log_price <= price_cut[0]).to_numpy(), "unit price: middle third": ((t.log_price > price_cut[0]) & (t.log_price <= price_cut[1])).to_numpy(), "unit price: highest third": (t.log_price > price_cut[1]).to_numpy()}
    for mo in sorted(m.unique()): s["month " + mo] = (m == mo).to_numpy()
    return s

def audit(preds, feats, col, db_path=None, top_n=15):
    cols = ["line_key", "log_price", "n_lines_in_invoice", "is_uk", "cust_log_prior_lines", "cust_log_mature_lines", "prod_log_mature_lines"]
    t = preds.merge(feats[cols], on="line_key", how="left"); t["ts"] = pd.to_datetime(t["ts"]); rows = []
    for name, mask in slices(t).items():
        g = t[mask]; y = g["y"].to_numpy(); n_pos = int(y.sum()); p = g[col].to_numpy(float)
        row = {"slice": name, "lines": len(g), "positives": n_pos, "observed_rate": float(y.mean()) if len(g) else np.nan, "mean_predicted": float(p.mean()) if len(g) else np.nan}
        row["predicted_over_observed"] = row["mean_predicted"] / row["observed_rate"] if n_pos else np.nan
        if n_pos >= 30 and n_pos < len(g): row["ap_lift"] = average_precision(y, p) / y.mean(); row["roc_auc"] = roc_auc(y, p)
        else: row["ap_lift"] = row["roc_auc"] = np.nan
        rows.append(row)
    errs = None
    if db_path is not None:
        conn = sqlite3.connect(str(db_path)); info = pd.read_sql("""SELECT s.line_key, s.invoice_no, s.invoice_ts, s.quantity, s.unit_price, c.customer_id, p.stock_code, p.description_clean
                                   FROM fact_sales_line s JOIN dim_customer c ON c.customer_key = s.customer_key JOIN dim_product p ON p.product_key = s.product_key""", conn)
        neg = t[t.y == 0].nlargest(top_n, col)[["line_key", "y", col]].assign(kind="false alarm: highest score, not returned")
        pos = t[t.y == 1].nsmallest(top_n, col)[["line_key", "y", col]].assign(kind="miss: lowest score, returned")
        errs = pd.concat([neg, pos]).merge(info, on="line_key", how="left").rename(columns={col: "score"})
    return pd.DataFrame(rows), errs

def drift(feats, preds, col):
    use = feats[(feats.is_final == 1) & feats.split.isin(["dev", "test"])].copy(); use["month"] = use.ts.dt.to_period("M").astype(str)
    out = use.groupby(["month", "split"]).agg(lines=("y", "size"), positives=("y", "sum"), base_rate=("y", "mean")).reset_index()
    p = preds.copy(); p["month"] = pd.to_datetime(p.ts).dt.to_period("M").astype(str)
    pm = p.groupby("month").agg(mean_predicted=(col, "mean")).reset_index(); return out.merge(pm, on="month", how="left")

def main():
    cfg = load_config(); P = ROOT / "data" / "processed"; A = ROOT / "artifacts"; ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["ablation", "labels", "audit", "drift"]); ap.add_argument("--features", default=str(P / "features.pkl")); ap.add_argument("--db", default=str(ROOT / cfg["data"]["db_path"]))
    ap.add_argument("--artifacts", default=str(A)); ap.add_argument("--test-preds", default=str(P / "test_preds.pkl")); ap.add_argument("--boot", type=int, default=200); a = ap.parse_args()
    art = pathlib.Path(a.artifacts); art.mkdir(parents=True, exist_ok=True); pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30); pd.set_option("display.max_colwidth", 60)
    params = json.load(open(art / "best_params.json")).get("logistic", {}); smap = json.load(open(art / "strategy_map.json")); strategy = smap.get("logistic", "none")
    if a.what == "ablation":
        t = ablation(pd.read_pickle(a.features), cfg, params, strategy, a.boot); t.to_csv(art / "ablation.csv", index=False)
        print("Feature-group ablation, logistic regression, development folds (average precision, customers resampled %d times):" % a.boot); print(t.round(4).to_string(index=False))
    elif a.what == "labels":
        t = label_sensitivity(a.db, cfg, params, strategy, B=max(50, a.boot // 2)); t.to_csv(art / "label_sensitivity.csv", index=False)
        print("Label sensitivity (development folds; lift = average precision divided by the base rate):"); print(t.round(4).to_string(index=False))
    elif a.what == "audit":
        col = json.load(open(art / "model_choice.json"))["chosen"] + "_calibrated"; t, errs = audit(pd.read_pickle(a.test_preds), pd.read_pickle(a.features), col, a.db if pathlib.Path(a.db).exists() else None)
        t.to_csv(art / "audit_slices.csv", index=False); print("Test predictions by slice (descriptive; %s):" % col); print(t.round(4).to_string(index=False))
        if errs is not None: errs.to_csv(art / "audit_errors.csv", index=False); print("\nLargest errors:"); print(errs.round(4).to_string(index=False))
    else:
        col = json.load(open(art / "model_choice.json"))["chosen"] + "_calibrated"; t = drift(pd.read_pickle(a.features), pd.read_pickle(a.test_preds), col); t.to_csv(art / "drift_monthly.csv", index=False)
        print("Return rate by month of purchase, and mean calibrated prediction on the test months:"); print(t.round(4).to_string(index=False))

if __name__ == "__main__":
    main()
