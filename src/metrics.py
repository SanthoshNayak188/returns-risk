"""Metrics for rare-event risk scores, and a bootstrap that resamples whole customers (lines of one customer are correlated).
The evaluation frame is shuffled once so that tied scores (the rule baselines have many) are broken at random, not by time."""
import numpy as np, pandas as pd
from scipy.stats import rankdata

def average_precision(y, s):
    """Tie-aware average precision (same value as scikit-learn's average_precision_score)."""
    order = np.argsort(-s, kind="stable"); s, y = s[order], y[order].astype(float)
    tp = np.cumsum(y); last = np.r_[s[1:] != s[:-1], True]
    tp_g, n_g = tp[last], np.arange(1, len(y) + 1)[last]
    return float(np.sum(np.diff(np.r_[0.0, tp_g / max(tp[-1], 1.0)]) * (tp_g / n_g)))

def roc_auc(y, s):
    r = rankdata(s); npos = y.sum(); nneg = len(y) - npos
    return float((r[y == 1].sum() - npos * (npos + 1) / 2) / max(npos * nneg, 1))

def recall_precision_at_k(y, s, ks):
    order = np.argsort(-s, kind="stable"); ys = y[order]; cs = np.cumsum(ys); out = {}
    for k in ks:
        n = max(1, int(np.ceil(k * len(y)))); out[k] = (cs[n - 1] / max(y.sum(), 1), cs[n - 1] / n)
    return out

def expected_calibration_error(y, p, bins=10):
    """Equal-frequency bins, so the rare high-risk bins are not empty."""
    order = np.argsort(p, kind="stable"); parts = np.array_split(order, bins)
    return float(sum(len(ix) / len(y) * abs(p[ix].mean() - y[ix].mean()) for ix in parts if len(ix)))

def log_loss(y, p, eps=1e-12):
    p = np.clip(p, eps, 1 - eps); return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

def metric_row(y, s, ks, probability=True):
    y = np.asarray(y); s = np.asarray(s, float); r = {"ap": average_precision(y, s), "roc_auc": roc_auc(y, s), "base_rate": float(y.mean())}
    r["ap_lift"] = r["ap"] / max(r["base_rate"], 1e-12)
    for k, (rec, prec) in recall_precision_at_k(y, s, ks).items():
        r["recall_at_%g%%" % (100 * k)] = float(rec); r["precision_at_%g%%" % (100 * k)] = float(prec)
    if probability:
        r["brier"] = float(np.mean((s - y) ** 2)); r["log_loss"] = log_loss(y, s); r["ece"] = expected_calibration_error(y, s)
    return r

def shuffled(df, seed=0):
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

def cluster_bootstrap(df, score_cols, base, ks, B=1000, seed=42, cluster="customer_key", level=0.95):
    """Resample whole customers B times; every model sees the same draws, so differences are paired.
    Returns per-model point estimates with intervals for AP and recall at k, and the paired AP difference to `base`."""
    rng = np.random.default_rng(seed); y = df["y"].to_numpy()
    groups = list(df.groupby(cluster, sort=False).indices.values()); G = len(groups)
    cols = {c: df[c].to_numpy(float) for c in score_cols}
    names = ["ap"] + ["recall_at_%g%%" % (100 * k) for k in ks]
    draws = {c: {n: [] for n in names} for c in score_cols}
    for _ in range(B):
        idx = np.concatenate([groups[i] for i in rng.integers(0, G, G)]); yb = y[idx]
        for c in score_cols:
            sb = cols[c][idx]; draws[c]["ap"].append(average_precision(yb, sb))
            for k, (rec, _) in recall_precision_at_k(yb, sb, ks).items(): draws[c]["recall_at_%g%%" % (100 * k)].append(rec)
    lo, hi = (1 - level) / 2 * 100, (1 + level) / 2 * 100; rows = []
    for c in score_cols:
        point = metric_row(y, cols[c], ks, probability=False); row = {"model": c}
        for n in names:
            d = np.array(draws[c][n]); row[n] = point[n]; row[n + "_lo"], row[n + "_hi"] = np.percentile(d, [lo, hi])
        for b in ([base] if isinstance(base, str) else list(base)):
            if c == b: continue
            diff = np.array(draws[c]["ap"]) - np.array(draws[b]["ap"]); d0 = point["ap"] - metric_row(y, cols[b], ks, probability=False)["ap"]
            if isinstance(base, str): row["ap_diff_vs_" + b] = d0; row["ap_diff_lo"], row["ap_diff_hi"] = np.percentile(diff, [lo, hi])
            else: row["ap_diff_vs_" + b] = d0; row["ap_diff_vs_%s_lo" % b], row["ap_diff_vs_%s_hi" % b] = np.percentile(diff, [lo, hi])
        rows.append(row)
    return pd.DataFrame(rows)


def cluster_bootstrap_scalar(df, fn, B=300, seed=42, cluster="customer_key", level=0.95):
    """Interval for any single number fn(frame), resampling whole customers."""
    rng = np.random.default_rng(seed); groups = list(df.groupby(cluster, sort=False).indices.values()); G = len(groups); vals = []
    for _ in range(B):
        vals.append(fn(df.iloc[np.concatenate([groups[i] for i in rng.integers(0, G, G)])]))
    return tuple(np.percentile(vals, [(1 - level) / 2 * 100, (1 + level) / 2 * 100]))
