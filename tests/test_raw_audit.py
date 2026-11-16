import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src")); sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from make_fake_data import make
import raw, audit
from utils import load_config

def test_csv_is_normalised_and_guests_stay_missing(tmp_path):
    make(tmp_path / "a.csv", n_customers=60); df = raw.load_raw(tmp_path / "a.csv")
    assert list(df.columns) == list(raw.COLS)
    assert df.customer_id.dropna().str.fullmatch(r"\d{5}").all()          # '12345.0' became '12345'
    assert df.customer_id.isna().any() and df.invoice_ts.is_monotonic_increasing and (df.stock_code == df.stock_code.str.upper()).all()

def test_excel_with_two_sheets_gives_the_same_rows_and_is_cached(tmp_path):
    make(tmp_path / "a.csv", n_customers=60, seed=3); make(tmp_path / "a.xlsx", n_customers=60, seed=3, as_xlsx=True)
    cache = tmp_path / "interim.csv"
    x = raw.load_raw(tmp_path / "a.xlsx", cache); c = raw.load_raw(tmp_path / "a.csv")
    assert len(x) == len(c) and cache.exists()
    assert np.isclose(x.price.sum(), c.price.sum()) and x.customer_id.isna().sum() == c.customer_id.isna().sum()
    y = raw.load_raw(tmp_path / "a.xlsx", cache)                          # second read comes from the cache
    assert len(y) == len(x)

def test_alternative_column_names_are_accepted():
    df = pd.DataFrame({"InvoiceNo": ["1"], "StockCode": ["20000"], "Description": ["x"], "Quantity": [1], "InvoiceDate": ["2010-01-01 10:00:00"],
                       "UnitPrice": [1.0], "CustomerID": [12345.0], "Country": ["UK"]})
    assert raw.normalise(df).customer_id.iloc[0] == "12345"

def test_missing_columns_give_a_clear_error():
    import pytest
    with pytest.raises(ValueError, match="Columns not found"):
        raw.normalise(pd.DataFrame({"Invoice": ["1"]}))

def test_audit_finds_the_planted_features(fake):
    r = audit.audit(fake["df"], load_config())
    assert r["rows"] == len(fake["df"]) and r["credit_lines"] > 0 and r["non_product_rows"] > 0 and r["zero_or_negative_price_rows"] > 0
    assert r["guest_rows_share"] > 0.05 and r["credit_lines_with_nonnegative_quantity"] == 0 and r["gate0_rows_match"] is False
    assert 5 not in r["rows_by_weekday"]                                  # no Saturdays (pandas day_of_week 5)
