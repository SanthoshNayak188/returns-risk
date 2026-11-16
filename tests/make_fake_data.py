"""Generates fake Online Retail II style data with PLANTED returns, so the pipeline can be tested without the real file.
Returns the truth table of planted returns (invoice, stock code, days until the credit note)."""
import pathlib
import numpy as np, pandas as pd

def make(path, n_customers=300, n_products=80, seed=0, return_rate=0.06, as_xlsx=False, invoices_per_day=5, risk_signal=False):
    rng = np.random.default_rng(seed)
    days = pd.date_range("2009-12-01", "2011-12-09"); days = days[days.day_of_week != 5]          # no trading on Saturdays, as in the real data
    prods = ["%d%s" % (20000 + i, "A" if i % 7 == 0 else "") for i in range(n_products)]
    price = dict(zip(prods, np.round(rng.uniform(0.5, 15, n_products), 2)))
    custs = rng.choice(np.arange(12000, 18000), n_customers, replace=False)
    rows, sale_idx, inv = [], [], 500000
    for d in days:
        for _ in range(rng.poisson(invoices_per_day)):
            inv += 1; guest = rng.random() < 0.10; cid = np.nan if guest else float(rng.choice(custs))
            ts = d + pd.Timedelta(hours=int(rng.integers(8, 17)), minutes=int(rng.integers(0, 60)))
            for it in rng.choice(prods, int(rng.integers(1, 6)), replace=False):
                rows.append([str(inv), it, "ITEM " + it, int(rng.integers(1, 25)), ts, price[it], cid, "United Kingdom"])
                if not guest: sale_idx.append(len(rows) - 1)
            if rng.random() < 0.10: rows.append([str(inv), "POST", "POSTAGE", 1, ts, 18.0, cid, "United Kingdom"])
            if rng.random() < 0.01: rows.append([str(inv), prods[0], "FREE ITEM", 1, ts, 0.0, cid, "United Kingdom"])
    end = pd.Timestamp("2011-12-09 23:59"); truth = []; cinv = 900000
    risky_p = set(prods[::7][:max(1, n_products // 7)]) if risk_signal else set(); risky_c = set(custs[: max(1, n_customers // 8)]) if risk_signal else set()
    for i in sale_idx:
        rate = return_rate * (6 if rows[i][1] in risky_p else 1) * (3 if rows[i][6] in risky_c else 1) if risk_signal else return_rate
        if rng.random() >= min(rate, 0.9): continue
        s = rows[i]; cts = s[4] + pd.Timedelta(days=int(rng.integers(1, 150)))
        if cts.day_of_week == 5: cts += pd.Timedelta(days=1)
        if cts > end: continue
        cinv += 1; rows.append(["C" + str(cinv), s[1], s[2], -int(rng.integers(1, s[3] + 1)), cts, s[5], s[6], s[7]])
        truth.append((s[0], s[1], (cts - s[4]).days))
    for _ in range(int(0.03 * len(truth))):                                                         # credit notes with no matching purchase
        cinv += 1; it = rng.choice(prods); ts = rng.choice(days) + pd.Timedelta(hours=10)
        rows.append(["C" + str(cinv), it, "ITEM " + it, -int(rng.integers(1, 10)), ts, price[it], float(rng.choice(custs)), "United Kingdom"])
    df = pd.DataFrame(rows, columns=["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price", "Customer ID", "Country"])
    df = df.sort_values("InvoiceDate", kind="stable").reset_index(drop=True)
    path = pathlib.Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if as_xlsx:
        cut = pd.Timestamp("2010-12-09")
        with pd.ExcelWriter(path) as w:
            df[df.InvoiceDate < cut].to_excel(w, sheet_name="Year 2009-2010", index=False); df[df.InvoiceDate >= cut].to_excel(w, sheet_name="Year 2010-2011", index=False)
    else:
        out = df.copy(); out["Customer ID"] = out["Customer ID"].map(lambda v: "" if pd.isna(v) else "%.1f" % v); out.to_csv(path, index=False)
    return pd.DataFrame(truth, columns=["invoice", "stock_code", "days_to_credit"])
