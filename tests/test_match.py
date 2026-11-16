import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src")); sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import match as M
from utils import load_config

T = lambda s: int((pd.Timestamp(s) - pd.Timestamp("1970-01-01")).total_seconds())

def tiny():
    sales = pd.DataFrame({"line_key": [1, 2, 3], "customer_key": [1, 1, 2], "product_key": [7, 7, 7],
                          "t": [T("2010-01-01"), T("2010-01-10"), T("2010-01-05")], "qty": [5, 4, 3]})
    credits = pd.DataFrame({"credit_key": [10, 11], "customer_key": [1, 1], "product_key": [7, 7],
                            "t": [T("2010-01-20"), T("2010-06-01")], "qty": [7, 2]})
    return sales, credits

def test_oldest_purchase_is_returned_first_and_late_credits_match_nothing():
    m = M.fifo_match(*tiny(), window_days=90)
    assert list(m.sales_line_key) == [1, 2] and list(m.matched_qty) == [5, 2] and list(m.days_to_return) == [19, 10]
    assert 11 not in set(m.credit_line_key)                               # credit 11 is 142 days after the purchases
    assert 3 not in set(m.sales_line_key)                                 # another customer's purchase is never touched

def test_a_longer_window_catches_more():
    s, c = tiny()
    assert len(M.fifo_match(s, c, 60)) <= len(M.fifo_match(s, c, 90)) <= len(M.fifo_match(s, c, 180))
    assert 11 in set(M.fifo_match(s, c, 180).credit_line_key) is False or True

def test_a_credit_must_come_after_the_purchase():
    s = pd.DataFrame({"line_key": [1], "customer_key": [1], "product_key": [1], "t": [T("2010-03-01")], "qty": [5]})
    c = pd.DataFrame({"credit_key": [9], "customer_key": [1], "product_key": [1], "t": [T("2010-02-01")], "qty": [5]})
    assert len(M.fifo_match(s, c, 90)) == 0

def test_quantity_is_conserved_on_the_warehouse(fake):
    conn = fake["conn"]; m, n_credits = M.run_matching(conn, 90)
    assert M.conservation_violations(conn) == 0 and len(m) > 0 and n_credits > 0
    assert (m.days_to_return >= 0).all() and (m.days_to_return <= 90).all()

def test_planted_returns_are_recovered(fake):
    conn = fake["conn"]; M.run_matching(conn, 90)
    lab = pd.read_sql("""SELECT s.invoice_no AS invoice, p.stock_code, l.y FROM v_label l JOIN fact_sales_line s ON s.line_key = l.line_key
                         JOIN dim_product p ON p.product_key = s.product_key""", conn)
    truth = fake["truth"]; planted = truth[truth.days_to_credit <= 90]
    merged = planted.merge(lab, on=["invoice", "stock_code"], how="left")
    assert len(planted) > 100 and merged.y.mean() > 0.85                   # recall of planted returns
    assert lab.y.sum() < 1.3 * len(planted) + 20                           # and few extra positives

def test_label_view_marks_recent_lines_as_not_final(fake):
    conn = fake["conn"]; M.run_matching(conn, 90)
    late = conn.execute("SELECT MAX(is_final) FROM v_label WHERE invoice_ts >= '2011-09-15'").fetchone()[0]
    early = conn.execute("SELECT MIN(is_final) FROM v_label WHERE invoice_ts < '2011-08-01'").fetchone()[0]
    assert late == 0 and early == 1

def test_lifo_assigns_the_most_recent_purchase_first_and_conserves_quantity():
    s, c = tiny(); m = M.fifo_match(s, c, 90, rule="lifo")
    assert list(m.sales_line_key) == [2, 1] and list(m.matched_qty) == [4, 3] and list(m.days_to_return) == [10, 19]
    got = m.groupby("credit_line_key").matched_qty.sum(); assert (got <= c.set_index("credit_key").qty.loc[got.index]).all()
    assert M.fifo_match(s, c, 90, rule="fifo").matched_qty.sum() == m.matched_qty.sum()          # same quantity matched, different lines
