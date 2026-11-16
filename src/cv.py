"""Time-ordered cross-validation with an embargo. Training lines end `embargo_days` before the validation block starts, because a line
bought just before the block only gets its label 90 days later. The test period is never touched here, and only lines whose
outcome window is complete (is_final) are used."""
import numpy as np, pandas as pd
from models import Model

def usable(df):
    return df[(df.is_final == 1) & df.split.isin(["dev", "test"])]

def fold_indices(df, cfg):
    """Positions in df of the training and validation lines of each fold."""
    ts = df["ts"]; dev = (df["split"] == "dev") & (df["is_final"] == 1); emb = pd.Timedelta(days=cfg["split"]["embargo_days"])
    for f in cfg["cv"]["folds"]:
        vs = pd.Timestamp(f["val_start"]); ve = pd.Timestamp(f["val_end"]) + pd.Timedelta(days=1)
        yield f["name"], np.flatnonzero((dev & (ts < vs - emb)).to_numpy()), np.flatnonzero((dev & (ts >= vs) & (ts < ve)).to_numpy())

def run_cv(df, cfg, model_names, strategy="none", seed=1, params=None, verbose=True, extra_seeds=()):
    """One row per validation line with the fold, the label and one score column per model (further seeds as <model>__seed<k>)."""
    params = params or {}; parts = []
    for name, tr, va in fold_indices(df, cfg):
        assert df.iloc[tr]["split"].eq("dev").all() and df.iloc[va]["split"].eq("dev").all()
        assert df.iloc[tr]["ts"].max() < df.iloc[va]["ts"].min() - pd.Timedelta(days=cfg["split"]["embargo_days"]) + pd.Timedelta(seconds=1)
        out = df.iloc[va][["line_key", "customer_key", "ts", "y"]].copy(); out["fold"] = name
        for m in model_names:
            seeds = [seed] + ([s for s in extra_seeds if s != seed] if m in ("lightgbm",) else [])
            for k, s in enumerate(seeds):
                st = strategy.get(m, "none") if isinstance(strategy, dict) else strategy
                mdl = Model(m, st, s, params.get(m), neg_per_pos=cfg["imbalance"]["undersample_negatives_per_positive"]).fit(df.iloc[tr], df.iloc[tr]["y"].to_numpy())
                out[m if k == 0 else "%s__seed%d" % (m, s)] = mdl.predict(df.iloc[va])
            if verbose: print("  fold %s  %-18s trained on %d lines (%d positive), scored %d" % (name, m, len(tr), int(df.iloc[tr]["y"].sum()), len(va)), flush=True)
        parts.append(out)
    return pd.concat(parts, ignore_index=True)

def permuted_label_check(df, cfg, seed=1):
    """Sanity check: train on fold V1 with the training labels shuffled. The model must then be no better than chance (average precision
    close to the validation base rate). If it is much better, label information reaches the model by some route other than the features."""
    name, tr, va = next(fold_indices(df, cfg)); rng = np.random.default_rng(seed)
    ytr = df.iloc[tr]["y"].to_numpy(); yva = df.iloc[va]["y"].to_numpy()
    from metrics import average_precision
    shuffled = Model("lightgbm", seed=seed, params={"n_estimators": 100}).fit(df.iloc[tr], rng.permutation(ytr)).predict(df.iloc[va])
    real = Model("lightgbm", seed=seed, params={"n_estimators": 100}).fit(df.iloc[tr], ytr).predict(df.iloc[va])
    r = {"base_rate": float(yva.mean()), "shuffled_ap": average_precision(yva, shuffled), "real_ap": average_precision(yva, real)}
    r["passed"] = bool(r["shuffled_ap"] < 1.5 * r["base_rate"])               # the leak check: shuffled labels must give chance-level results
    r["real_lift"] = r["real_ap"] / max(r["base_rate"], 1e-12); r["finds_signal"] = bool(r["real_lift"] > 1.2); return r
