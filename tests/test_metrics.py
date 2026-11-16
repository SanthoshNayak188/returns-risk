import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
import metrics as M

def toy(n=4000, seed=0, sig=1.0):
    rng = np.random.default_rng(seed); y = (rng.random(n) < 0.05).astype(int); s = rng.normal(size=n) + sig * y
    return y, s

def test_average_precision_and_auc_match_scikit_learn_including_ties():
    y, s = toy(); s_ties = np.round(s, 1)
    for sc in (s, s_ties):
        assert np.isclose(M.average_precision(y, sc), average_precision_score(y, sc)); assert np.isclose(M.roc_auc(y, sc), roc_auc_score(y, sc))

def test_recall_and_precision_at_k_by_hand():
    y = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 1]); s = np.array([.9, .8, .7, .6, .5, .4, .3, .2, .1, 0.05])
    r = M.recall_precision_at_k(y, s, [0.2, 0.5])
    assert r[0.2] == (1 / 3, 1 / 2) and r[0.5] == (2 / 3, 2 / 5)

def test_calibration_error_is_small_for_calibrated_and_large_for_overconfident_scores():
    rng = np.random.default_rng(1); p = rng.uniform(0, 0.3, 20000); y = (rng.random(20000) < p).astype(int)
    assert M.expected_calibration_error(y, p) < 0.01 and M.expected_calibration_error(y, np.clip(p * 3, 0, 1)) > 0.1

def test_bootstrap_intervals_contain_the_estimate_and_pair_the_models():
    y, good = toy(sig=2.0); _, bad = toy(sig=0.3, seed=0)
    df = M.shuffled(pd.DataFrame({"y": y, "good": good, "bad": bad, "customer_key": np.random.default_rng(2).integers(0, 400, len(y))}))
    a = M.cluster_bootstrap(df, ["good", "bad"], "bad", [0.05], B=120, seed=3); b = M.cluster_bootstrap(df, ["good", "bad"], "bad", [0.05], B=120, seed=3)
    pd.testing.assert_frame_equal(a, b)
    for r in a.itertuples(): assert r.ap_lo <= r.ap <= r.ap_hi
    g = a[a.model == "good"].iloc[0]; assert g.ap_diff_lo > 0

def test_shuffle_is_deterministic_and_keeps_every_row():
    df = pd.DataFrame({"a": range(50)}); s = M.shuffled(df, 7)
    assert sorted(s.a) == list(range(50)) and M.shuffled(df, 7).a.tolist() == s.a.tolist() and s.a.tolist() != list(range(50))
