"""Builds the return label. A credit note does not say which invoice it refers to, so each credit is assigned to the same
customer's earlier purchases of the same product, OLDEST FIRST, within the label window. A partial return counts.
The result is stored in bridge_return_match and exposed through the view v_label.
Usage: python src/match.py [--db warehouse/retail.db]"""
import argparse, sqlite3
import numpy as np, pandas as pd
from utils import load_config, ROOT

DAY = 86400

def _secs(ts):
    return ((pd.to_datetime(ts) - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1)).to_numpy("int64")

def load_for_matching(conn):
    """Registered customers only: guests cannot be linked to their returns."""
    sales = pd.read_sql("SELECT line_key, customer_key, product_key, invoice_ts AS ts, quantity AS qty FROM fact_sales_line WHERE customer_key != 0", conn)
    credits = pd.read_sql("SELECT credit_line_key AS credit_key, customer_key, product_key, credit_ts AS ts, quantity_returned AS qty FROM fact_return_line WHERE customer_key != 0", conn)
    for d in (sales, credits): d["t"] = _secs(d["ts"])
    return sales, credits

def fifo_match(sales, credits, window_days=90):
    """sales: line_key, customer_key, product_key, t (seconds), qty.  credits: credit_key, customer_key, product_key, t, qty (positive).
    Returns one row per (sale line, credit line) pair: matched quantity and days between purchase and credit."""
    win = window_days * DAY
    pairs = pd.MultiIndex.from_frame(credits[["customer_key", "product_key"]].drop_duplicates())          # only these pairs can ever match
    sales = sales[pd.MultiIndex.from_frame(sales[["customer_key", "product_key"]]).isin(pairs)]
    groups = {}
    for (c, p), g in sales.sort_values("t").groupby(["customer_key", "product_key"], sort=False):
        groups[(c, p)] = (g["t"].to_numpy(), g["qty"].to_numpy().astype(np.int64).copy(), g["line_key"].to_numpy())
    out = []
    for r in credits.sort_values("t").itertuples(index=False):
        g = groups.get((r.customer_key, r.product_key))
        if g is None: continue
        t_arr, left, keys = g
        lo = int(np.searchsorted(t_arr, r.t - win, side="left")); hi = int(np.searchsorted(t_arr, r.t, side="left"))
        need = int(r.qty)
        for i in range(lo, hi):
            if left[i] <= 0: continue
            take = min(int(left[i]), need); left[i] -= take; need -= take
            out.append((int(keys[i]), int(r.credit_key), take, int((r.t - t_arr[i]) // DAY)))
            if need <= 0: break
    return pd.DataFrame(out, columns=["sales_line_key", "credit_line_key", "matched_qty", "days_to_return"])

def write_bridge(conn, matches):
    conn.execute("DELETE FROM bridge_return_match")
    rows = matches.astype("int64").to_numpy().tolist()          # plain Python ints: sqlite would store numpy integers as blobs
    conn.executemany("INSERT INTO bridge_return_match VALUES (?,?,?,?)", rows)
    conn.execute("CREATE INDEX IF NOT EXISTS ix_bridge_sale ON bridge_return_match(sales_line_key)")      # the label view looks up every purchase here
    conn.execute("CREATE INDEX IF NOT EXISTS ix_bridge_credit ON bridge_return_match(credit_line_key)")
    conn.commit()

def create_label_view(conn, window_days):
    w = int(window_days)
    conn.execute("DROP VIEW IF EXISTS v_label")
    conn.execute("""CREATE VIEW v_label AS
        SELECT s.line_key, s.invoice_ts, s.customer_key, s.product_key,
               CASE WHEN EXISTS (SELECT 1 FROM bridge_return_match m WHERE m.sales_line_key = s.line_key AND m.days_to_return <= %d) THEN 1 ELSE 0 END AS y,
               CASE WHEN julianday(s.invoice_ts) + %d <= (SELECT julianday(MAX(invoice_ts)) FROM fact_sales_line) THEN 1 ELSE 0 END AS is_final
        FROM fact_sales_line s""" % (w, w))
    conn.commit()

def conservation_violations(conn):
    """A credit cannot return more than it holds, and a purchase cannot be returned more than once over."""
    a = conn.execute("""SELECT COUNT(*) FROM (SELECT credit_line_key, SUM(matched_qty) AS m FROM bridge_return_match GROUP BY credit_line_key) b
                        JOIN fact_return_line r ON r.credit_line_key = b.credit_line_key WHERE b.m > r.quantity_returned""").fetchone()[0]
    b = conn.execute("""SELECT COUNT(*) FROM (SELECT sales_line_key, SUM(matched_qty) AS m FROM bridge_return_match GROUP BY sales_line_key) b
                        JOIN fact_sales_line s ON s.line_key = b.sales_line_key WHERE b.m > s.quantity""").fetchone()[0]
    return int(a + b)

def run_matching(conn, window_days):
    sales, credits = load_for_matching(conn)
    matches = fifo_match(sales, credits, window_days)
    write_bridge(conn, matches); create_label_view(conn, window_days)
    return matches, len(credits)

def main():
    cfg = load_config()
    ap = argparse.ArgumentParser(); ap.add_argument("--db", default=str(ROOT / cfg["data"]["db_path"])); a = ap.parse_args()
    conn = sqlite3.connect(a.db)
    m, n_credits = run_matching(conn, cfg["label"]["window_days"])
    print("Window %d days: %d matches for %d credit lines of registered customers (%.1f percent of credit lines matched)" % (
        cfg["label"]["window_days"], len(m), n_credits, 100 * m.credit_line_key.nunique() / max(1, n_credits)))
    print("Conservation violations (must be 0):", conservation_violations(conn))
    print("Positive lines:", conn.execute("SELECT SUM(y), COUNT(*) FROM v_label WHERE customer_key != 0").fetchone())

if __name__ == "__main__":
    main()
