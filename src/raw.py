"""Reads the Online Retail II data (Excel with two sheets, or CSV) into one clean table with fixed column names."""
import pathlib
import numpy as np, pandas as pd

# fixed name -> accepted names in the file
COLS = {"invoice": ["Invoice", "InvoiceNo"], "stock_code": ["StockCode"], "description": ["Description"], "quantity": ["Quantity"],
        "invoice_ts": ["InvoiceDate"], "price": ["Price", "UnitPrice"], "customer_id": ["Customer ID", "CustomerID"], "country": ["Country"]}

def normalise(df):
    rename, missing = {}, []
    for new, options in COLS.items():
        hit = next((o for o in options if o in df.columns), None)
        if hit is None: missing.append(new)
        else: rename[hit] = new
    if missing:
        raise ValueError("Columns not found: %s. The file has: %s" % (missing, list(df.columns)))
    df = df.rename(columns=rename)[list(COLS)].copy()
    df["invoice"] = df["invoice"].astype(str).str.strip()
    df["stock_code"] = df["stock_code"].astype("string").str.strip().str.upper()
    df["description"] = df["description"].astype("string").str.strip().str.upper()
    df["country"] = df["country"].astype("string").str.strip()
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["invoice_ts"] = pd.to_datetime(df["invoice_ts"], errors="coerce")
    cid = pd.to_numeric(df["customer_id"], errors="coerce")
    df["customer_id"] = cid.astype("Int64").astype("string")          # missing stays missing (guest)
    return df.dropna(subset=["invoice_ts", "quantity", "price"]).reset_index(drop=True)

def load_raw(path, interim_csv=None):
    """Excel is read once and cached as CSV, because reading about a million Excel rows takes minutes."""
    path = pathlib.Path(path); cache = pathlib.Path(interim_csv) if interim_csv else None
    if path.suffix.lower() in (".xlsx", ".xls"):
        if cache is not None and cache.exists() and cache.stat().st_mtime >= path.stat().st_mtime:
            return normalise(pd.read_csv(cache, dtype=str))
        sheets = pd.read_excel(path, sheet_name=None, dtype=str)
        df = pd.concat(sheets.values(), ignore_index=True)
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True); df.to_csv(cache, index=False)
        return normalise(df)
    return normalise(pd.read_csv(path, dtype=str))
