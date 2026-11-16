"""Point-in-time features for the return-risk model. One row per purchase line of a registered customer.
Every feature for a line uses only information available when its invoice is placed:
  * facts about the line and invoice itself (price, quantity, invoice size, hour, weekday, country);
  * label-free history from strictly earlier lines (counts, tenure, spend, recent units sold);
  * return-rate history built ONLY from lines bought at least `maturity_lag_days` earlier, because only then is a line's outcome known.
Nothing is fitted on data here, so no later rows can leak in. Imputation and scaling happen inside the models' pipelines, per fold.
Usage: python src/features.py [--db warehouse/retail.db] [--out data/processed/features.pkl]"""
import argparse, pathlib, sqlite3
import numpy as np, pandas as pd
from utils import load_config, ROOT

DAY = 86400
FEATURE_GROUPS = {
    "line": ["log_qty", "log_price", "log_line_amount", "price_ratio", "qty_ratio", "line_share"],
    "invoice_time": ["n_lines_in_invoice", "log_invoice_total", "hour", "dow", "month_sin", "month_cos"],
    "country": ["is_uk", "country_group"],
    "customer_history": ["cust_log_prior_lines", "cust_tenure_days", "cust_days_since_last", "cust_log_spend", "cust_log_mature_lines", "cust_rate_mature"],
    "product_history": ["prod_log_prior_lines", "prod_days_since_first", "prod_log_units_recent", "prod_log_mature_lines", "prod_rate_mature"],
    "pair_history": ["pair_prior_lines"],
}
FEATURES = [c for cols in FEATURE_GROUPS.values() for c in cols]
ID_COLS = ["line_key", "invoice_no", "ts", "customer_key", "product_key", "y", "is_final", "split"]

def load_lines(conn):
    df = pd.read_sql("""SELECT s.line_key, s.invoice_no, s.invoice_ts AS ts, s.customer_key, s.product_key, s.quantity, s.unit_price, s.line_amount,
                               s.n_lines_in_invoice, s.invoice_total, c.country, l.y, l.is_final
                        FROM fact_sales_line s JOIN dim_customer c ON c.customer_key = s.customer_key JOIN v_label l ON l.line_key = s.line_key
                        WHERE s.customer_key != 0 ORDER BY s.invoice_ts, s.line_key""", conn)
    df["ts"] = pd.to_datetime(df["ts"])
    df["t"] = ((df["ts"] - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1)).to_numpy("int64")
    return df

def group_stats(df, key, L, R):
    """For each line: history of its customer (or product) using strictly earlier lines, and label history using lines at least L seconds older."""
    n = len(df); names = ["n_prior", "tenure", "since_last", "spend", "n_mature", "k_mature", "units_recent", "price_mean", "qty_mean"]
    out = {k: np.full(n, np.nan) for k in names}
    t_all, y_all = df["t"].to_numpy(), df["y"].to_numpy(float)
    amt, qty, prc = df["line_amount"].to_numpy(float), df["quantity"].to_numpy(float), df["unit_price"].to_numpy(float)
    for _, idx in df.groupby(key, sort=False).indices.items():          # idx are positions in ascending time order
        t = t_all[idx]; z = np.zeros(1)
        cy, ca, cq, cp = (np.concatenate([z, np.cumsum(v[idx])]) for v in (y_all, amt, qty, prc))
        before = np.searchsorted(t, t, side="left")                      # lines strictly earlier
        mature = np.searchsorted(t, t - L, side="right")                 # lines bought at least L earlier (outcome known)
        recent = np.minimum(np.searchsorted(t, t - R, side="right"), before)
        has = before > 0
        out["n_prior"][idx] = before; out["spend"][idx] = ca[before]
        out["tenure"][idx] = np.where(has, (t - t[0]) / DAY, np.nan)
        out["since_last"][idx] = np.where(has, (t - t[np.maximum(before - 1, 0)]) / DAY, np.nan)
        out["n_mature"][idx] = mature; out["k_mature"][idx] = cy[mature]
        out["units_recent"][idx] = cq[before] - cq[recent]
        out["price_mean"][idx] = np.where(has, cp[before] / np.maximum(before, 1), np.nan)
        out["qty_mean"][idx] = np.where(has, cq[before] / np.maximum(before, 1), np.nan)
    return out

def build_features(conn, cfg, maturity_lag_days=None):
    F = cfg["features"]; S = cfg["split"]
    L = (cfg["label"]["maturity_lag_days"] if maturity_lag_days is None else maturity_lag_days) * DAY; R = F["recent_days"] * DAY; m = float(F["shrinkage_m"])
    df = load_lines(conn); t = df["t"].to_numpy()
    cu, pr = group_stats(df, "customer_key", L, R), group_stats(df, "product_key", L, R)
    cyg = np.concatenate([[0.0], np.cumsum(df["y"].to_numpy(float))]); gm = np.searchsorted(t, t - L, side="right")
    prior = (cyg[gm] + 2.0) / (gm + 100.0)                               # global mature return rate as of each line
    shrunk = lambda k, n: (k + m * prior) / (n + m)
    out = df[["line_key", "invoice_no", "ts", "customer_key", "product_key"]].copy()
    out["y"] = df["y"].astype(int); out["is_final"] = df["is_final"].astype(int)
    out["log_qty"] = np.log1p(df.quantity); out["log_price"] = np.log(df.unit_price); out["log_line_amount"] = np.log1p(df.line_amount)
    out["price_ratio"] = df.unit_price / pr["price_mean"]; out["qty_ratio"] = df.quantity / pr["qty_mean"]; out["line_share"] = df.line_amount / df.invoice_total
    out["n_lines_in_invoice"] = df.n_lines_in_invoice; out["log_invoice_total"] = np.log1p(df.invoice_total)
    out["hour"] = df.ts.dt.hour; out["dow"] = df.ts.dt.day_of_week
    out["month_sin"] = np.sin(2 * np.pi * (df.ts.dt.month - 1) / 12); out["month_cos"] = np.cos(2 * np.pi * (df.ts.dt.month - 1) / 12)
    countries = list(F["countries"]); out["is_uk"] = (df.country == countries[0]).astype(int)
    out["country_group"] = df.country.map({c: i for i, c in enumerate(countries)}).fillna(len(countries)).astype(int)
    out["cust_log_prior_lines"] = np.log1p(cu["n_prior"]); out["cust_tenure_days"] = cu["tenure"]; out["cust_days_since_last"] = cu["since_last"]
    out["cust_log_spend"] = np.log1p(cu["spend"]); out["cust_log_mature_lines"] = np.log1p(cu["n_mature"]); out["cust_rate_mature"] = shrunk(cu["k_mature"], cu["n_mature"])
    out["prod_log_prior_lines"] = np.log1p(pr["n_prior"]); out["prod_days_since_first"] = pr["tenure"]; out["prod_log_units_recent"] = np.log1p(pr["units_recent"])
    out["prod_log_mature_lines"] = np.log1p(pr["n_mature"]); out["prod_rate_mature"] = shrunk(pr["k_mature"], pr["n_mature"])
    out["pair_prior_lines"] = df.groupby(["customer_key", "product_key"]).cumcount().to_numpy()
    d = df.ts.dt.normalize()
    out["split"] = np.select([(d >= pd.Timestamp(S["dev_start"])) & (d <= pd.Timestamp(S["dev_end"])),
                              (d > pd.Timestamp(S["dev_end"])) & (d < pd.Timestamp(S["test_start"])),
                              (d >= pd.Timestamp(S["test_start"])) & (d <= pd.Timestamp(S["test_end"]))], ["dev", "embargo", "test"], default="other")
    fl = out.select_dtypes("float64").columns; out[fl] = out[fl].astype("float32")
    return out.reset_index(drop=True)

def feature_columns(df=None):
    return list(FEATURES)

def main():
    cfg = load_config()
    ap = argparse.ArgumentParser(); ap.add_argument("--db", default=str(ROOT / cfg["data"]["db_path"]))
    ap.add_argument("--out", default=str(ROOT / "data" / "processed" / "features.pkl")); a = ap.parse_args()
    df = build_features(sqlite3.connect(a.db), cfg)
    out = pathlib.Path(a.out); out.parent.mkdir(parents=True, exist_ok=True); df.to_pickle(out)
    print("rows %d, features %d, saved to %s" % (len(df), len(FEATURES), out))
    use = df[df.split.isin(["dev", "test"]) & (df.is_final == 1)]
    print(use.groupby("split").agg(lines=("y", "size"), positives=("y", "sum"), rate=("y", "mean")).round(4).to_string())
    print("\nShare of empty values per feature (empty means 'no history yet'):")
    print(df[FEATURES].isna().mean().round(3).sort_values(ascending=False).head(10).to_string())

if __name__ == "__main__":
    main()
