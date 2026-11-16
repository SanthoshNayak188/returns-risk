"""Models for return risk. Everything learned (imputation, scaling, the model itself) is fitted on the training rows only.
Rule baselines need no fitting beyond the training return rate. Imbalance strategies, all applied to the TRAINING part only:
  none         fit on the data as it is, rank by the predicted probability;
  weight       give positive lines a larger weight in the loss;
  undersample  keep all positives and a random sample of negatives (validation and test data are never resampled)."""
import numpy as np, pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from features import FEATURES

B2_FEATURES = ["log_qty", "log_price", "n_lines_in_invoice", "cust_rate_mature", "prod_rate_mature"]
RULES = {"B1a_customer_rate": "cust_rate_mature", "B1b_product_rate": "prod_rate_mature"}
MODEL_NAMES = ["B0_prior", "B1a_customer_rate", "B1b_product_rate", "B2_logistic_small", "logistic", "lightgbm"]
STRATEGIES = ["none", "weight", "undersample"]

class Winsorizer(BaseEstimator, TransformerMixin):
    """Clips every feature to the 1st to 99th percentile range of the TRAINING rows. Growth features such as customer tenure keep rising with calendar
    time, and a linear model would otherwise extrapolate far outside what it was trained on."""
    def __init__(self, q=0.01): self.q = q
    def fit(self, X, y=None):
        X = pd.DataFrame(X); self.lo_, self.hi_ = X.quantile(self.q), X.quantile(1 - self.q); return self
    def transform(self, X):
        return pd.DataFrame(X).clip(lower=self.lo_, upper=self.hi_, axis=1)

class Model:
    def __init__(self, name, strategy="none", seed=42, params=None, features=None, neg_per_pos=5):
        assert name in MODEL_NAMES and strategy in STRATEGIES
        self.name, self.strategy, self.seed, self.params, self.neg_per_pos = name, strategy, seed, dict(params or {}), neg_per_pos
        self.features = list(features) if features is not None else (B2_FEATURES if name == "B2_logistic_small" else list(FEATURES))

    def _resample(self, X, y):
        if self.strategy != "undersample": return X, y
        rng = np.random.default_rng(self.seed); pos = np.flatnonzero(y == 1); neg = np.flatnonzero(y == 0)
        keep = np.concatenate([pos, rng.choice(neg, min(len(neg), self.neg_per_pos * max(len(pos), 1)), replace=False)])
        keep.sort(); return X.iloc[keep], y[keep]

    def fit(self, df, y):
        y = np.asarray(y).astype(int); self.prior_ = float(y.mean()); self.odds_factor_ = 1.0
        npos, nneg = int((y == 1).sum()), int((y == 0).sum())
        if self.name in ("logistic", "lightgbm") and self.strategy == "weight": self.odds_factor_ = npos / max(nneg, 1)       # positives upweighted by nneg/npos
        if self.name in ("logistic", "lightgbm") and self.strategy == "undersample": self.odds_factor_ = min(1.0, self.neg_per_pos * max(npos, 1) / max(nneg, 1))   # share of negatives kept
        if self.name in ("B0_prior", *RULES): return self
        X, yy = self._resample(df[self.features], y)
        if self.name in ("logistic", "B2_logistic_small"):
            cw = "balanced" if self.strategy == "weight" else None
            self.est_ = Pipeline([("win", Winsorizer()), ("imp", SimpleImputer(strategy="median", add_indicator=True)), ("sc", StandardScaler()),
                                  ("clf", LogisticRegression(C=self.params.get("C", 1.0), class_weight=cw, max_iter=3000))])
            self.est_.fit(X, yy)
        else:
            g = dict(n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=100, subsample=0.8, subsample_freq=1,
                     colsample_bytree=0.8, random_state=self.seed, verbose=-1, n_jobs=-1)
            g.update(self.params)
            if self.strategy == "weight": g["scale_pos_weight"] = float((yy == 0).sum() / max((yy == 1).sum(), 1))
            self.est_ = lgb.LGBMClassifier(**g).fit(X, yy)
        return self

    def prior_corrected(self, p):
        """Undo the training-time shift of the odds: true odds = model odds x factor. Needs no data, only how the training set was altered."""
        p = np.clip(np.asarray(p, float), 1e-9, 1 - 1e-9); return self.odds_factor_ * p / (1 - p + self.odds_factor_ * p)

    def predict(self, df):
        if self.name == "B0_prior": return np.full(len(df), self.prior_)
        if self.name in RULES: return df[RULES[self.name]].fillna(self.prior_).to_numpy(float)
        return self.est_.predict_proba(df[self.features])[:, 1]
