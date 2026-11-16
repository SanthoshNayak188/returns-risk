"""Calibration of the risk scores. Undersampling and reweighting raise the predicted probabilities, so they must be corrected before they can be read as probabilities.
Three corrections are compared on the development folds with FORWARD CHAINING (a correction for fold k is fitted only on the out-of-fold predictions of earlier folds):
  analytic   undo the known shift of the odds (needs no fitting);
  platt      a logistic regression on the logit of the score;
  isotonic   a monotone step function fitted to the scores.
RULE (stated in the proposal): lowest log-loss; among methods within 1 percent of the best, the simplest (analytic, then platt, then isotonic).
Usage: python src/calibrate.py"""
import argparse, json, pathlib
import numpy as np, pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from utils import load_config, ROOT
from cv import fold_indices
from metrics import expected_calibration_error, log_loss, average_precision

METHODS = ["none", "analytic", "platt", "isotonic"]; SIMPLEST_FIRST = ["analytic", "platt", "isotonic", "none"]

def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6); return np.log(p / (1 - p))

def analytic_correct(p, factor):
    p = np.clip(np.asarray(p, float), 1e-9, 1 - 1e-9); return factor * p / (1 - p + factor * p)

class Calibrator:
    def __init__(self, method, factor=1.0): self.method, self.factor = method, factor
    def fit(self, p, y):
        if self.method == "platt": self.lr_ = LogisticRegression(C=1e6, max_iter=2000).fit(_logit(p).reshape(-1, 1), y)
        if self.method == "isotonic": self.iso_ = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(p, y)
        return self
    def predict(self, p):
        p = np.asarray(p, float)
        if self.method == "none": return p
        if self.method == "analytic": return analytic_correct(p, self.factor)
        if self.method == "platt": return self.lr_.predict_proba(_logit(p).reshape(-1, 1))[:, 1]
        return self.iso_.predict(p)

def fold_factors(df, cfg, strategy):
    """Per fold: how much the training step shifted the odds (1 when nothing was changed)."""
    out = {}
    for name, tr, va in fold_indices(df, cfg):
        y = df.iloc[tr]["y"].to_numpy(); npos, nneg = int(y.sum()), int(len(y) - y.sum())
        out[name] = (min(1.0, cfg["imbalance"]["undersample_negatives_per_positive"] * npos / nneg) if strategy == "undersample" else npos / nneg if strategy == "weight" else 1.0)
    return out

def forward_chain_eval(oof, col, factors, methods=METHODS):
    """Evaluate each correction on folds V2, V3, ... using only earlier folds to fit it."""
    folds = sorted(oof["fold"].unique()); rows = {m: [] for m in methods}; ys = []
    for k in range(1, len(folds)):
        fit = oof[oof["fold"].isin(folds[:k])]; ev = oof[oof["fold"] == folds[k]]; ys.append(ev["y"].to_numpy())
        for m in methods:
            rows[m].append(Calibrator(m, factors[folds[k]]).fit(fit[col].to_numpy(float), fit["y"].to_numpy()).predict(ev[col].to_numpy(float)))
    y = np.concatenate(ys); tab = {}
    for m in methods:
        p = np.concatenate(rows[m]); tab[m] = {"log_loss": log_loss(y, p), "brier": float(np.mean((p - y) ** 2)), "ece": expected_calibration_error(y, p), "mean_prediction": float(p.mean()),
                                              "observed_rate": float(y.mean()), "ap": average_precision(y, p)}
    return pd.DataFrame(tab).T

def choose_method(tab):
    best = tab["log_loss"].min(); close = [m for m in tab.index if tab.loc[m, "log_loss"] <= best * 1.01]
    return sorted(close, key=lambda m: SIMPLEST_FIRST.index(m))[0]

def main():
    cfg = load_config(); P = ROOT / "data" / "processed"; ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=str(P / "features.pkl")); ap.add_argument("--oof", default=str(P / "cv_oof.pkl"))
    ap.add_argument("--strategy-map", default=str(ROOT / "artifacts" / "strategy_map.json")); ap.add_argument("--choice", default=str(ROOT / "artifacts" / "model_choice.json"))
    ap.add_argument("--out-dir", default=str(ROOT / "artifacts")); a = ap.parse_args()
    chosen = json.load(open(a.choice))["chosen"]; strategy = json.load(open(a.strategy_map)).get(chosen, "none")
    oof = pd.read_pickle(a.oof); df = pd.read_pickle(a.features); fac = fold_factors(df, cfg, strategy)
    tab = forward_chain_eval(oof, chosen, fac); method = choose_method(tab)
    pd.set_option("display.width", 200)
    print("Model %s, imbalance strategy %s. Odds factors by fold: %s" % (chosen, strategy, {k: round(v, 3) for k, v in fac.items()}))
    print("\nCalibration on folds V2 to V4, each corrected using only earlier folds:"); print(tab.round(5).to_string())
    print("\nCHOSEN CALIBRATION (lowest log-loss, simplest within 1 percent): %s" % method)
    out = pathlib.Path(a.out_dir); out.mkdir(parents=True, exist_ok=True); tab.to_csv(out / "calibration_dev_eval.csv")
    (out / "calibration_choice.json").write_text(json.dumps({"method": method, "model": chosen, "strategy": strategy, "odds_factors_by_fold": fac}, indent=2)); print("Saved calibration_choice.json")

if __name__ == "__main__":
    main()
