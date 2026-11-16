import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src")); sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import label_audit as LA
from match import load_for_matching
from utils import load_config

CFG = load_config()

def test_window_table_grows_with_the_window_and_gates_fail_with_the_right_reason(fake):
    sales, credits = load_for_matching(fake["conn"]); t, m = LA.window_table(sales, credits, CFG)
    assert list(t.window_days) == CFG["label"]["windows_to_compare"]
    assert t.positives_dev.is_monotonic_increasing and t.matched_credit_line_share.is_monotonic_increasing
    g = LA.evaluate_gates(t, m, CFG)
    assert 0.5 < g["share_of_matches_within_window"] < 0.7                  # planted delays are spread over 150 days, so 90 days catches about 60 percent
    assert g["gate1_passed"] is False and g["gate2_passed"] is False        # and the fake data are too small for Gate 2
    assert "longer window" in g["recommendation"] and "Gate 2 fails" in g["recommendation"]

def test_gates_pass_when_the_thresholds_are_met(fake):
    sales, credits = load_for_matching(fake["conn"]); t, m = LA.window_table(sales, credits, CFG)
    easy = {**CFG, "label": {**CFG["label"], "min_dev_positives": 10, "min_test_positives": 10, "min_matched_credit_line_share": 0.4, "min_share_within_window": 0.5}}
    g = LA.evaluate_gates(t, m, easy)
    assert g["gate1_passed"] and g["gate2_passed"] and "Continue" in g["recommendation"]

def test_low_matching_gives_a_matching_message(fake):
    sales, credits = load_for_matching(fake["conn"]); t, m = LA.window_table(sales, credits, CFG)
    strict = {**CFG, "label": {**CFG["label"], "min_matched_credit_line_share": 0.95}}
    assert "can be matched" in LA.evaluate_gates(t, m, strict)["recommendation"]

def test_samples_contain_matched_and_unmatched_credits(fake):
    sales, credits = load_for_matching(fake["conn"]); t, m = LA.window_table(sales, credits, CFG)
    s = LA.credit_samples(fake["conn"], sales, credits, m[90])
    assert set(s.kind) == {"matched", "unmatched"} and s[s.kind == "matched"].matched_days_to_return.notna().all()
    assert s[s.kind == "unmatched"].matched_days_to_return.isna().all()
