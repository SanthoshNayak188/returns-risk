"""Gate 0 and data audit for Online Retail II. Every number printed here feeds decision_log.md.
Usage: python src/audit.py [--raw data/raw/online_retail_II.xlsx] [--out-dir artifacts]"""
import argparse, json, pathlib, re
import numpy as np, pandas as pd
from utils import load_config, ROOT
from raw import load_raw

def non_product_mask(df, cfg):
    codes = {c.upper() for c in cfg["non_product_codes"]}
    low = df["stock_code"].fillna("").str.lower()
    return df["stock_code"].isin(codes) | low.str.startswith(tuple(p.lower() for p in cfg["non_product_prefixes"])) | df["stock_code"].isna() | (df["stock_code"] == "")

def audit(df, cfg):
    r = {}
    r["rows"] = int(len(df)); r["date_min"] = str(df.invoice_ts.min().date()); r["date_max"] = str(df.invoice_ts.max().date())
    r["gate0_rows_match"] = bool(abs(len(df) - cfg["data"]["expected_rows"]) <= 0.01 * cfg["data"]["expected_rows"])
    r["invoices"] = int(df.invoice.nunique()); r["customers"] = int(df.customer_id.nunique())
    r["guest_rows_share"] = round(float(df.customer_id.isna().mean()), 4)
    credit = df.invoice.str.startswith("C")
    r["credit_lines"] = int(credit.sum()); r["credit_line_share"] = round(float(credit.mean()), 4)
    r["credit_invoices"] = int(df.loc[credit, "invoice"].nunique()); r["credit_invoice_share"] = round(r["credit_invoices"] / r["invoices"], 4)
    r["credit_lines_with_nonnegative_quantity"] = int((credit & (df.quantity >= 0)).sum())
    r["sale_lines_with_nonpositive_quantity"] = int((~credit & (df.quantity <= 0)).sum())
    npm = non_product_mask(df, cfg)
    r["non_product_rows"] = int(npm.sum())
    r["zero_or_negative_price_rows"] = int((df.price <= 0).sum())
    r["missing_description_rows"] = int(df.description.isna().sum())
    r["exact_duplicate_rows"] = int(df.duplicated(["invoice", "stock_code", "quantity", "invoice_ts", "price", "customer_id"]).sum())
    other = df.loc[~df.stock_code.fillna("").str.match(r"^\d{5}") & ~npm, "stock_code"].value_counts().head(20)
    r["top_nonnumeric_codes_not_in_config"] = {str(k): int(v) for k, v in other.items()}
    big = df.loc[~credit & (df.quantity > 5000)]
    r["sale_lines_over_5000_units"] = int(len(big))
    r["rows_per_month"] = {str(k): int(v) for k, v in df.invoice_ts.dt.to_period("M").value_counts().sort_index().items()}
    r["rows_by_weekday"] = {int(k): int(v) for k, v in df.invoice_ts.dt.day_of_week.value_counts().sort_index().items()}
    r["country_top5"] = {str(k): int(v) for k, v in df.country.value_counts().head(5).items()}
    return r

def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(ROOT / cfg["data"]["raw_path"]))
    ap.add_argument("--interim", default=str(ROOT / cfg["data"]["interim_csv"]))
    ap.add_argument("--out-dir", default=str(ROOT / "artifacts"))
    a = ap.parse_args()
    df = load_raw(a.raw, a.interim)
    rep = audit(df, cfg)
    out = pathlib.Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "audit_report.json").write_text(json.dumps(rep, indent=2))
    for k, v in rep.items(): print("%s: %s" % (k, v))
    print("\nSaved audit_report.json. Write one decision_log.md line per issue, and extend non_product_codes in config.yaml if needed.")

if __name__ == "__main__":
    main()
