"""Builds the star schema in SQLite from the Online Retail II rows and loads it day by day, reconciling every batch.
Usage: python src/etl.py [--raw ...] [--db warehouse/retail.db] [--replay-days 30] [--fresh]
Steps: stage raw rows -> classify each row (sale line, return line, or rejected with a reason) -> dimensions ->
first N trading days one at a time -> the rest in one bulk batch. A failed reconciliation check stops the run."""
import argparse, hashlib, pathlib, sqlite3
import numpy as np, pandas as pd
from utils import load_config, ROOT
from raw import load_raw
from audit import non_product_mask

UNKNOWN = 0     # customer_key for guests (no customer id)

def connect(path):
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path)); conn.execute("PRAGMA foreign_keys = ON"); return conn

def init_schema(conn):
    conn.executescript((ROOT / "sql" / "schema.sql").read_text())

def classify(df, cfg):
    """Why a row is not a sale or return line, or '' if it is one."""
    credit = df["invoice"].str.startswith("C")
    reason = np.select([non_product_mask(df, cfg), df["price"] <= 0, credit & (df["quantity"] >= 0), ~credit & (df["quantity"] <= 0)],
                       ["non_product", "bad_price", "bad_credit_sign", "bad_sale_sign"], default="")
    return pd.Series(reason, index=df.index)

def stage(conn, df, cfg, raw_label="input"):
    d = df.copy()
    d.insert(0, "source_row", np.arange(1, len(d) + 1))
    d["day"] = d["invoice_ts"].dt.strftime("%Y-%m-%d"); d["ts"] = d["invoice_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
    d["is_credit"] = d["invoice"].str.startswith("C").astype(int)
    d["reason"] = classify(d, cfg)
    d["customer_id"] = d["customer_id"].astype(object).where(d["customer_id"].notna(), None)
    d = d.drop(columns="invoice_ts")
    d["stock_code"] = d["stock_code"].astype(object); d["description"] = d["description"].astype(object); d["country"] = d["country"].astype(object)
    d.to_sql("stg_transactions", conn, if_exists="replace", index=False)
    conn.execute("CREATE INDEX IF NOT EXISTS ix_stg_day ON stg_transactions(day)")
    conn.execute("DELETE FROM stg_files"); conn.execute("INSERT INTO stg_files VALUES (?,?,?)", (raw_label, "", len(d))); conn.commit()

def load_dimensions(conn):
    lo, hi = conn.execute("SELECT MIN(day), MAX(day) FROM stg_transactions").fetchone()
    rows = []
    for d in pd.date_range(lo, hi):
        rows.append((int(d.strftime("%Y%m%d")), d.strftime("%Y-%m-%d"), d.isoweekday(), d.month, d.year, int(d.isoweekday() >= 6)))
    conn.executemany("INSERT OR IGNORE INTO dim_date VALUES (?,?,?,?,?,?)", rows)
    conn.execute("INSERT OR IGNORE INTO dim_customer VALUES (?,?,?,?)", (UNKNOWN, "UNKNOWN", None, 1))
    cust = pd.read_sql("SELECT customer_id, MIN(country) AS country FROM stg_transactions WHERE customer_id IS NOT NULL GROUP BY customer_id ORDER BY customer_id", conn)
    conn.executemany("INSERT OR IGNORE INTO dim_customer VALUES (?,?,?,0)", [(i + 1, c, k) for i, (c, k) in enumerate(zip(cust.customer_id, cust.country))])
    prod = pd.read_sql("""SELECT stock_code, description FROM (SELECT stock_code, description, COUNT(*) n FROM stg_transactions WHERE reason = '' AND stock_code IS NOT NULL
                          GROUP BY stock_code, description) ORDER BY stock_code, n DESC""", conn).drop_duplicates("stock_code")
    conn.executemany("INSERT OR IGNORE INTO dim_product VALUES (?,?,?,1)", [(i + 1, s, d) for i, (s, d) in enumerate(zip(prod.stock_code, prod.description))])
    conn.commit()

def reconcile_range(conn, batch_id, start, end):
    a, b = int(start.replace("-", "")), int(end.replace("-", ""))
    q = lambda sql, *p: conn.execute(sql, p).fetchone()[0] or 0
    src = q("SELECT COUNT(*) FROM stg_transactions WHERE day BETWEEN ? AND ?", start, end)
    sales = q("SELECT COUNT(*) FROM fact_sales_line WHERE date_key BETWEEN ? AND ?", a, b)
    rets = q("SELECT COUNT(*) FROM fact_return_line WHERE date_key BETWEEN ? AND ?", a, b)
    rej = q("SELECT COUNT(*) FROM reject_rows WHERE day BETWEEN ? AND ?", start, end)
    amt_src = q("SELECT SUM(s.quantity * s.price) FROM stg_transactions s JOIN fact_sales_line f ON f.source_row = s.source_row WHERE s.day BETWEEN ? AND ?", start, end)
    amt_wh = q("SELECT SUM(line_amount) FROM fact_sales_line WHERE date_key BETWEEN ? AND ?", a, b)
    qty_src = q("SELECT SUM(-s.quantity) FROM stg_transactions s JOIN fact_return_line f ON f.source_row = s.source_row WHERE s.day BETWEEN ? AND ?", start, end)
    qty_wh = q("SELECT SUM(quantity_returned) FROM fact_return_line WHERE date_key BETWEEN ? AND ?", a, b)
    orphans = q("""SELECT COUNT(*) FROM fact_sales_line f LEFT JOIN dim_customer c ON f.customer_key = c.customer_key LEFT JOIN dim_product p ON f.product_key = p.product_key
                   LEFT JOIN dim_date d ON f.date_key = d.date_key WHERE f.date_key BETWEEN ? AND ? AND (c.customer_key IS NULL OR p.product_key IS NULL OR d.date_key IS NULL)""", a, b) \
        + q("""SELECT COUNT(*) FROM fact_return_line f LEFT JOIN dim_customer c ON f.customer_key = c.customer_key LEFT JOIN dim_product p ON f.product_key = p.product_key
               LEFT JOIN dim_date d ON f.date_key = d.date_key WHERE f.date_key BETWEEN ? AND ? AND (c.customer_key IS NULL OR p.product_key IS NULL OR d.date_key IS NULL)""", a, b)
    dup = q("SELECT COUNT(*) - COUNT(DISTINCT source_row) FROM fact_sales_line WHERE date_key BETWEEN ? AND ?", a, b) + \
          q("SELECT COUNT(*) - COUNT(DISTINCT source_row) FROM fact_return_line WHERE date_key BETWEEN ? AND ?", a, b)
    inv_bad = q("""SELECT COUNT(*) FROM (SELECT invoice_no, ABS(SUM(line_amount) - MAX(invoice_total)) AS d FROM fact_sales_line
                   WHERE date_key BETWEEN ? AND ? GROUP BY invoice_no HAVING d > 0.000001)""", a, b)
    ok = lambda x, y: abs(float(x) - float(y)) < 1e-6
    res = [("rows_in_equals_sales_plus_returns_plus_rejected", src, sales + rets + rej, ok(src, sales + rets + rej)),
           ("sale_amount_matches_source", amt_src, amt_wh, ok(amt_src, amt_wh)),
           ("returned_quantity_matches_source", qty_src, qty_wh, ok(qty_src, qty_wh)),
           ("duplicate_source_rows_in_facts", 0, dup, dup == 0),
           ("foreign_key_orphans", 0, orphans, orphans == 0),
           ("invoice_total_equals_sum_of_lines", 0, inv_bad, inv_bad == 0)]
    conn.execute("DELETE FROM recon_results WHERE batch_id = ?", (batch_id,))
    conn.executemany("INSERT INTO recon_results VALUES (?,?,?,?,?)", [(batch_id, n, s, w, int(p)) for n, s, w, p in res]); conn.commit()
    return res

def load_range(conn, start, end):
    cur = conn.execute("INSERT INTO etl_batch_log (load_date, rows_in, status) VALUES (?,?,?)", ("%s..%s" % (start, end), 0, "running"))
    batch_id = cur.lastrowid
    d = pd.read_sql("SELECT * FROM stg_transactions WHERE day BETWEEN ? AND ?", conn, params=(start, end))
    conn.execute("UPDATE etl_batch_log SET rows_in = ? WHERE batch_id = ?", (len(d), batch_id))
    cust = dict(conn.execute("SELECT customer_id, customer_key FROM dim_customer").fetchall())
    prod = dict(conn.execute("SELECT stock_code, product_key FROM dim_product").fetchall())
    conn.execute("DELETE FROM reject_rows WHERE day BETWEEN ? AND ?", (start, end))
    rej = d[d.reason != ""]
    conn.executemany("INSERT OR REPLACE INTO reject_rows VALUES (?,?,?)", list(zip(rej.source_row.astype(int), rej.day, rej.reason)))
    ok = d[d.reason == ""].copy()
    ok["date_key"] = ok.day.str.replace("-", "").astype(int)
    ok["customer_key"] = ok.customer_id.map(lambda c: cust.get(c, UNKNOWN) if c is not None else UNKNOWN)
    ok["product_key"] = ok.stock_code.map(prod)
    sales = ok[ok.is_credit == 0].copy()
    sales["line_amount"] = sales.quantity * sales.price
    g = sales.groupby("invoice")["line_amount"]
    sales["n_lines_in_invoice"] = g.transform("size"); sales["invoice_total"] = g.transform("sum")
    conn.executemany("INSERT OR IGNORE INTO fact_sales_line VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        [(int(r.source_row), int(r.source_row), r.invoice, r.ts, int(r.date_key), int(r.customer_key), int(r.product_key), int(r.quantity), float(r.price),
          float(r.line_amount), int(r.n_lines_in_invoice), float(r.invoice_total)) for r in sales.itertuples()])
    rets = ok[ok.is_credit == 1]
    conn.executemany("INSERT OR IGNORE INTO fact_return_line VALUES (?,?,?,?,?,?,?,?)",
        [(int(r.source_row), int(r.source_row), r.invoice, r.ts, int(r.date_key), int(r.customer_key), int(r.product_key), int(-r.quantity)) for r in rets.itertuples()])
    conn.commit()
    res = reconcile_range(conn, batch_id, start, end)
    failed = [n for n, _, _, p in res if not p]
    a, b = int(start.replace("-", "")), int(end.replace("-", ""))
    loaded = conn.execute("SELECT (SELECT COUNT(*) FROM fact_sales_line WHERE date_key BETWEEN ? AND ?) + (SELECT COUNT(*) FROM fact_return_line WHERE date_key BETWEEN ? AND ?)", (a, b, a, b)).fetchone()[0]
    conn.execute("UPDATE etl_batch_log SET rows_loaded = ?, rows_rejected = ?, status = ? WHERE batch_id = ?", (loaded, len(rej), "FAILED" if failed else "ok", batch_id)); conn.commit()
    if failed: raise RuntimeError("Reconciliation failed for %s..%s: %s" % (start, end, failed))
    return batch_id

def build(conn, df, cfg, replay_days=30, verbose=False):
    init_schema(conn); stage(conn, df, cfg); load_dimensions(conn)
    days = [r[0] for r in conn.execute("SELECT DISTINCT day FROM stg_transactions ORDER BY day")]
    n = min(replay_days, len(days))
    for d in days[:n]:
        load_range(conn, d, d)
        if verbose: print("loaded day", d, flush=True)
    if n < len(days):
        load_range(conn, days[n], days[-1])
        if verbose: print("bulk loaded", days[n], "to", days[-1])

def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(ROOT / cfg["data"]["raw_path"])); ap.add_argument("--interim", default=str(ROOT / cfg["data"]["interim_csv"]))
    ap.add_argument("--db", default=str(ROOT / cfg["data"]["db_path"])); ap.add_argument("--replay-days", type=int, default=30)
    ap.add_argument("--fresh", action="store_true")
    a = ap.parse_args()
    if a.fresh and pathlib.Path(a.db).exists(): pathlib.Path(a.db).unlink()
    conn = connect(a.db)
    build(conn, load_raw(a.raw, a.interim), cfg, a.replay_days, verbose=True)
    print("\nWarehouse summary")
    for t in ["dim_date", "dim_customer", "dim_product", "fact_sales_line", "fact_return_line", "reject_rows", "etl_batch_log", "recon_results"]:
        print("  %-18s %d" % (t, conn.execute("SELECT COUNT(*) FROM " + t).fetchone()[0]))
    print("  rejected rows by reason:", dict(conn.execute("SELECT reason, COUNT(*) FROM stg_transactions WHERE reason != '' GROUP BY reason").fetchall()))
    print("  failed reconciliation checks:", conn.execute("SELECT COUNT(*) FROM recon_results WHERE passed = 0").fetchone()[0])

if __name__ == "__main__":
    main()
