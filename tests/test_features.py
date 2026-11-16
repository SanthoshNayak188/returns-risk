import sys, pathlib, sqlite3
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src")); sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import features as F, match
from utils import load_config

CFG = load_config(); DAY = pd.Timedelta(days=1)

def test_features_match_a_brute_force_calculation(fake_rs):
    feats, conn = fake_rs["feats"], fake_rs["conn"]; df = F.load_lines(conn); rng = np.random.default_rng(0)
    L, m = 90 * DAY, CFG["features"]["shrinkage_m"]
    for i in rng.choice(len(df), 60, replace=False):
        r = df.iloc[i]; o = feats.iloc[i]; assert o.line_key == r.line_key
        for key, pre in (("customer_key", "cust"), ("product_key", "prod")):
            g = df[df[key] == r[key]]; earlier = g[g.ts < r.ts]; mature = g[g.ts <= r.ts - L]
            gm = df[df.ts <= r.ts - L]; prior = (gm.y.sum() + 2) / (len(gm) + 100)
            assert np.isclose(o[pre + "_log_prior_lines"], np.log1p(len(earlier)), atol=1e-5) and np.isclose(o[pre + "_log_mature_lines"], np.log1p(len(mature)), atol=1e-5)
            assert np.isclose(o[pre + "_rate_mature"], (mature.y.sum() + m * prior) / (len(mature) + m), atol=1e-5)
        pair = df[(df.customer_key == r.customer_key) & (df.product_key == r.product_key)]
        assert o.pair_prior_lines == (pair.index < i).sum()

def rebuild_without_credits_after(fake_rs, cutoff, lag=None):
    src = fake_rs["conn"]; c = sqlite3.connect(":memory:"); src.backup(c)
    c.execute("DELETE FROM fact_return_line WHERE credit_ts > ?", (cutoff,)); c.commit()
    match.run_matching(c, CFG["label"]["window_days"]); return F.build_features(c, CFG, lag)

def test_deleting_credit_notes_after_a_cutoff_changes_no_earlier_feature(fake_rs):
    cutoff = "2011-02-01"; alt = rebuild_without_credits_after(fake_rs, cutoff); base = fake_rs["feats"]
    early = (base.ts <= pd.Timestamp(cutoff)).to_numpy(); assert early.sum() > 1000
    pd.testing.assert_frame_equal(base.loc[early, F.FEATURES].reset_index(drop=True), alt.loc[early, F.FEATURES].reset_index(drop=True))
    late = (base.ts > pd.Timestamp(cutoff) + 120 * DAY).to_numpy()
    assert not base.loc[late, F.FEATURES].reset_index(drop=True).equals(alt.loc[late, F.FEATURES].reset_index(drop=True))     # the change does matter later

def test_the_leak_detector_catches_a_planted_leak(fake_rs):
    """Self-check: with maturity lag 0 the history would use outcomes that are not yet known, and the same test must fail."""
    cutoff = "2011-02-01"; base0 = F.build_features(fake_rs["conn"], CFG, 0); alt0 = rebuild_without_credits_after(fake_rs, cutoff, 0)
    early = (base0.ts <= pd.Timestamp(cutoff)).to_numpy()
    assert not base0.loc[early, F.FEATURES].reset_index(drop=True).equals(alt0.loc[early, F.FEATURES].reset_index(drop=True))

def test_no_label_or_id_columns_are_features():
    assert not set(F.FEATURES) & set(F.ID_COLS) and len(F.FEATURES) == len(set(F.FEATURES)) == 26
    assert not any("y" == c or c.startswith("is_final") for c in F.FEATURES)

def test_split_labels_and_ranges(fake_rs):
    f = fake_rs["feats"]; S = CFG["split"]
    assert f[f.split == "dev"].ts.max() < pd.Timestamp(S["dev_end"]) + DAY and f[f.split == "test"].ts.min() >= pd.Timestamp(S["test_start"])
    assert f[f.split == "embargo"].ts.min() >= pd.Timestamp(S["dev_end"]) and (f[f.split == "other"].ts >= pd.Timestamp(S["test_end"])).all()
    assert set(f.split) == {"dev", "embargo", "test", "other"} and f[f.split == "dev"].is_final.eq(1).all()

def test_values_are_finite_and_in_range(fake_rs):
    f = fake_rs["feats"]; x = f[F.FEATURES]
    assert not np.isinf(x.to_numpy(float)).any() and f.hour.between(0, 23).all() and f.cust_rate_mature.between(0, 1).all() and f.prod_rate_mature.between(0, 1).all()
    assert f.y.isin([0, 1]).all() and f.groupby("line_key").size().max() == 1
