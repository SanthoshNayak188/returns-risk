import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src")); sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import pytest
import etl
from utils import load_config

def n(conn, sql): return conn.execute(sql).fetchone()[0]

def test_every_reconciliation_check_passes(fake):
    c = fake["conn"]
    assert n(c, "SELECT COUNT(*) FROM recon_results") > 0 and n(c, "SELECT COUNT(*) FROM recon_results WHERE passed = 0") == 0
    assert n(c, "SELECT COUNT(*) FROM etl_batch_log WHERE status != 'ok'") == 0

def test_five_daily_batches_then_one_bulk(fake):
    assert n(fake["conn"], "SELECT COUNT(*) FROM etl_batch_log") == 5 + 1

def test_every_row_is_a_sale_a_return_or_a_rejection(fake):
    c = fake["conn"]
    assert n(c, "SELECT COUNT(*) FROM fact_sales_line") + n(c, "SELECT COUNT(*) FROM fact_return_line") + n(c, "SELECT COUNT(*) FROM reject_rows") == n(c, "SELECT COUNT(*) FROM stg_transactions") == len(fake["df"])
    reasons = {r[0] for r in c.execute("SELECT DISTINCT reason FROM reject_rows")}
    assert {"non_product", "bad_price"} <= reasons

def test_guests_go_to_the_unknown_member_and_returns_are_positive_quantities(fake):
    c = fake["conn"]
    assert n(c, "SELECT COUNT(*) FROM fact_sales_line WHERE customer_key = 0") > 0 and n(c, "SELECT is_unknown_member FROM dim_customer WHERE customer_key = 0") == 1
    assert n(c, "SELECT COUNT(*) FROM fact_return_line WHERE quantity_returned <= 0") == 0 and n(c, "SELECT COUNT(*) FROM fact_sales_line WHERE quantity <= 0 OR unit_price <= 0") == 0

def test_rerun_changes_nothing(fake):
    c = fake["conn"]; before = n(c, "SELECT COUNT(*) FROM fact_sales_line")
    days = [r[0] for r in c.execute("SELECT DISTINCT day FROM stg_transactions ORDER BY day LIMIT 3")]
    etl.load_range(c, days[0], days[-1])
    assert n(c, "SELECT COUNT(*) FROM fact_sales_line") == before and n(c, "SELECT COUNT(*) FROM recon_results WHERE passed = 0") == 0

def test_reconciliation_detects_a_deleted_row(fake):
    import sqlite3
    c = sqlite3.connect(":memory:"); fake["conn"].backup(c)                 # corrupt a COPY, never the shared warehouse
    c.execute("DELETE FROM fact_sales_line WHERE line_key = (SELECT MIN(line_key) FROM fact_sales_line)"); c.commit()
    res = etl.reconcile_range(c, 1, "2009-12-01", "2011-12-09")
    assert any(not p for _, _, _, p in res)
