"""Gates 1 and 2: can the return label be built well enough to continue? Run this FIRST on the real data.
For several label windows it reports how many credit lines can be matched and how many positive lines each window gives in the
development and test periods. It then evaluates the gates for the configured window and prints a recommendation.
It also saves 20 matched and 20 unmatched credit lines for you to read by hand.
Usage: python src/label_audit.py [--db warehouse/retail.db] [--out-dir artifacts]"""
import argparse, json, pathlib, sqlite3
import numpy as np, pandas as pd
from utils import load_config, ROOT
from match import load_for_matching, fifo_match, conservation_violations

def window_table(sales, credits, cfg):
    s = cfg["split"]; ts = pd.to_datetime(sales["ts"])
    dev = ((ts >= pd.Timestamp(s["dev_start"])) & (ts < pd.Timestamp(s["dev_end"]) + pd.Timedelta(days=1))).to_numpy()
    test = ((ts >= pd.Timestamp(s["test_start"])) & (ts < pd.Timestamp(s["test_end"]) + pd.Timedelta(days=1))).to_numpy()
    rows, matches = [], {}
    for w in cfg["label"]["windows_to_compare"]:
        m = fifo_match(sales, credits, w); matches[w] = m
        pos = set(m.sales_line_key)
        is_pos = sales["line_key"].isin(pos).to_numpy()
        rows.append({"window_days": w, "credit_lines": len(credits), "matched_credit_line_share": m.credit_line_key.nunique() / max(1, len(credits)),
                     "matched_quantity_share": m.matched_qty.sum() / max(1, credits.qty.sum()),
                     "positives_dev": int((is_pos & dev).sum()), "eligible_dev": int(dev.sum()), "positive_rate_dev": float((is_pos & dev).sum() / max(1, dev.sum())),
                     "positives_test": int((is_pos & test).sum()), "eligible_test": int(test.sum()), "positive_rate_test": float((is_pos & test).sum() / max(1, test.sum()))})
    return pd.DataFrame(rows), matches

def evaluate_gates(table, matches, cfg):
    L = cfg["label"]; w = L["window_days"]; row = table[table.window_days == w].iloc[0]
    longest = max(matches); m = matches[longest]
    within = float(m.matched_qty[m.days_to_return <= w].sum() / max(1, m.matched_qty.sum()))
    g = {"window_days": w, "matched_credit_line_share": float(row.matched_credit_line_share), "share_of_matches_within_window": within,
         "positives_dev": int(row.positives_dev), "positives_test": int(row.positives_test)}
    g["gate1_passed"] = bool(row.matched_credit_line_share >= L["min_matched_credit_line_share"] and within >= L["min_share_within_window"])
    g["gate2_passed"] = bool(row.positives_dev >= L["min_dev_positives"] and row.positives_test >= L["min_test_positives"])
    msgs = []
    if not g["gate1_passed"]:
        if row.matched_credit_line_share < L["min_matched_credit_line_share"]:
            msgs.append("Gate 1 fails: only %.0f%% of credit lines can be matched to a purchase. Read the unmatched samples, then consider an invoice-level label." % (100 * row.matched_credit_line_share))
        else:
            msgs.append("Gate 1 fails: only %.0f%% of matches fall inside %d days, so many returns arrive later. Consider a longer window (see the table) and widen the embargo to match." % (100 * within, w))
    if not g["gate2_passed"]:
        ok = table[(table.positives_dev >= L["min_dev_positives"]) & (table.positives_test >= L["min_test_positives"])]
        if len(ok):
            msgs.append("Gate 2 fails at %d days (too few positive lines); the shortest tested window that passes is %d days." % (w, int(ok.window_days.min())))
        else:
            msgs.append("Gate 2 fails: no tested window gives enough positive lines. Use an invoice-level label, or stop and submit the store sales project.")
    g["recommendation"] = " ".join(msgs) if msgs else "Both gates pass with a %d day window. Continue with the plan." % w
    return g

def credit_samples(conn, sales, credits, matches_at_w, seed=42, n=20):
    rng = np.random.default_rng(seed)
    matched = set(matches_at_w.credit_line_key); keys = credits.credit_key.to_numpy()
    is_m = np.isin(keys, list(matched))
    pick = lambda mask: rng.choice(keys[mask], min(n, int(mask.sum())), replace=False) if mask.any() else np.array([], dtype=int)
    chosen = {"matched": pick(is_m), "unmatched": pick(~is_m)}
    info = pd.read_sql("""SELECT r.credit_line_key AS credit_key, c.customer_id, p.stock_code, p.description_clean, r.credit_ts, r.quantity_returned
                          FROM fact_return_line r JOIN dim_customer c ON c.customer_key = r.customer_key JOIN dim_product p ON p.product_key = r.product_key""", conn)
    rows = []
    for kind, ks in chosen.items():
        for k in ks:
            r = info[info.credit_key == k].iloc[0].to_dict(); r["kind"] = kind
            cr = credits[credits.credit_key == k].iloc[0]
            same = sales[(sales.customer_key == cr.customer_key) & (sales.product_key == cr.product_key) & (sales.t < cr.t)]
            r["earlier_purchases_any_time"] = int(len(same))
            r["days_since_latest_earlier_purchase"] = (int((cr.t - same.t.max()) // 86400) if len(same) else None)
            mm = matches_at_w[matches_at_w.credit_line_key == k]
            r["matched_days_to_return"] = (int(mm.days_to_return.iloc[0]) if len(mm) else None)
            rows.append(r)
    return pd.DataFrame(rows)

def main():
    cfg = load_config()
    ap = argparse.ArgumentParser(); ap.add_argument("--db", default=str(ROOT / cfg["data"]["db_path"])); ap.add_argument("--out-dir", default=str(ROOT / "artifacts"))
    a = ap.parse_args()
    conn = sqlite3.connect(a.db); sales, credits = load_for_matching(conn)
    n_guest = conn.execute("SELECT (SELECT COUNT(*) FROM fact_sales_line WHERE customer_key = 0), (SELECT COUNT(*) FROM fact_return_line WHERE customer_key = 0)").fetchone()
    print("Registered-customer lines: %d sales, %d credits. Guest lines left out: %d sales, %d credits." % (len(sales), len(credits), n_guest[0], n_guest[1]), flush=True)
    table, matches = window_table(sales, credits, cfg)
    pd.set_option("display.width", 200); pd.set_option("display.max_columns", 20)
    print("\nBy label window (shares are of credit lines and quantity; rates are of purchase lines):")
    print(table.round(4).to_string(index=False))
    m = matches[max(matches)]
    print("\nDays from purchase to credit, matches found with the longest window (%d days): median %.0f, 75th percentile %.0f, 90th percentile %.0f" % (
        max(matches), m.days_to_return.median(), m.days_to_return.quantile(.75), m.days_to_return.quantile(.9)))
    g = evaluate_gates(table, matches, cfg)
    print("\nGate 1 (matching quality): %s   matched credit lines %.1f%% (need %.0f%%), matches inside the window %.1f%% (need %.0f%%)" % (
        "PASS" if g["gate1_passed"] else "FAIL", 100 * g["matched_credit_line_share"], 100 * cfg["label"]["min_matched_credit_line_share"],
        100 * g["share_of_matches_within_window"], 100 * cfg["label"]["min_share_within_window"]))
    print("Gate 2 (enough positives): %s   dev %d (need %d), test %d (need %d)" % ("PASS" if g["gate2_passed"] else "FAIL", g["positives_dev"],
          cfg["label"]["min_dev_positives"], g["positives_test"], cfg["label"]["min_test_positives"]))
    print("Recommendation:", g["recommendation"])
    out = pathlib.Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "label_audit_windows.csv", index=False)
    samples = credit_samples(conn, sales, credits, matches[cfg["label"]["window_days"]], cfg["seed"])
    samples.to_csv(out / "label_audit_samples.csv", index=False)
    (out / "label_audit.json").write_text(json.dumps(g, indent=2))
    print("\nSaved label_audit.json, label_audit_windows.csv and label_audit_samples.csv. Read the samples file: do matched pairs look like real returns? Why are credits unmatched?")

if __name__ == "__main__":
    main()
