"""Hyperparameter tuning with Optuna on the time-ordered folds (development period only). Objective: mean average precision over the four folds,
under the imbalance strategy chosen for each model. To keep it affordable, tuning uses a fixed seeded sample of CUSTOMERS (all their lines);
the final comparison uses everyone. Same trial budget for every model. Saved after every finished study; use --resume to continue.
Usage: python src/tune.py [--models logistic,lightgbm] [--seeds 1,2,3] [--trials 25] [--customer-fraction 0.5] [--resume]"""
import argparse, json, pathlib, time
import numpy as np, pandas as pd, optuna
from utils import load_config, ROOT
from cv import fold_indices
from models import Model
from metrics import average_precision, shuffled

optuna.logging.set_verbosity(optuna.logging.WARNING)
SPACES = {
    "logistic": lambda t: {"C": t.suggest_float("C", 1e-4, 10.0, log=True)},
    "lightgbm": lambda t: {"n_estimators": t.suggest_int("n_estimators", 100, 600), "learning_rate": t.suggest_float("learning_rate", 0.02, 0.15, log=True),
                           "num_leaves": t.suggest_int("num_leaves", 4, 64, log=True), "min_child_samples": t.suggest_int("min_child_samples", 50, 1000, log=True),
                           "subsample": t.suggest_float("subsample", 0.5, 1.0), "colsample_bytree": t.suggest_float("colsample_bytree", 0.4, 1.0),
                           "reg_lambda": t.suggest_float("reg_lambda", 1e-3, 30.0, log=True)}}

def sample_customers(df, fraction, seed):
    if fraction >= 1.0: return df
    c = np.sort(df["customer_key"].unique()); keep = np.random.default_rng(seed).choice(c, max(1, int(round(fraction * len(c)))), replace=False)
    return df[df["customer_key"].isin(keep)].reset_index(drop=True)

def cv_mean_ap(df, cfg, name, strategy, params, seed):
    aps = []
    for _, tr, va in fold_indices(df, cfg):
        m = Model(name, strategy, seed, params, neg_per_pos=cfg["imbalance"]["undersample_negatives_per_positive"]).fit(df.iloc[tr], df.iloc[tr]["y"].to_numpy())
        v = shuffled(df.iloc[va][["y"]].assign(s=m.predict(df.iloc[va])), 0); aps.append(average_precision(v["y"].to_numpy(), v["s"].to_numpy(float)))
    return float(np.mean(aps))

def tune_model(df, cfg, name, strategy, seed, n_trials, customer_fraction=1.0, verbose=False):
    sub = sample_customers(df, customer_fraction, cfg["seed"]); t0 = time.time()
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    def objective(trial):
        v = cv_mean_ap(sub, cfg, name, strategy, SPACES[name](trial), seed)
        if verbose: print("  %s seed %d trial %2d/%d  mean fold AP %.4f  (%.0fs)" % (name, seed, trial.number + 1, n_trials, v, time.time() - t0), flush=True)
        return v
    study.optimize(objective, n_trials=n_trials)
    return {"best_value": float(study.best_value), "best_params": study.best_params, "strategy": strategy, "n_trials": n_trials,
            "trial_values": [float(t.value) for t in study.trials], "customers_used": int(sub.customer_key.nunique()), "seconds": round(time.time() - t0)}

def summarize(results):
    best, summ = {}, {}
    for m, seeds in results.items():
        vals = {s: r["best_value"] for s, r in seeds.items()}; sb = max(vals, key=vals.get)
        best[m] = seeds[sb]["best_params"]; summ[m] = {"best_seed": sb, "best_value_per_seed": vals, "spread": max(vals.values()) - min(vals.values())}
    return best, summ

def main():
    cfg = load_config(); ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=str(ROOT / "data" / "processed" / "features.pkl")); ap.add_argument("--models", default="logistic,lightgbm")
    ap.add_argument("--seeds", default="1,2,3"); ap.add_argument("--trials", type=int, default=25); ap.add_argument("--customer-fraction", type=float, default=0.5)
    ap.add_argument("--strategy-map", default=str(ROOT / "artifacts" / "strategy_map.json")); ap.add_argument("--out-dir", default=str(ROOT / "artifacts")); ap.add_argument("--resume", action="store_true")
    a = ap.parse_args(); out = pathlib.Path(a.out_dir); out.mkdir(parents=True, exist_ok=True); path = out / "tuning_results.json"
    smap = json.load(open(a.strategy_map)); results = json.loads(path.read_text()) if a.resume and path.exists() else {}
    df = pd.read_pickle(a.features)
    for m in [x for x in a.models.split(",") if x]:
        for s in [int(x) for x in a.seeds.split(",") if x]:
            if str(s) in results.get(m, {}): print("skipping %s seed %d (done)" % (m, s)); continue
            print("tuning %s (strategy %s), seed %d, %d trials, customer fraction %.2f" % (m, smap.get(m, "none"), s, a.trials, a.customer_fraction), flush=True)
            results.setdefault(m, {})[str(s)] = tune_model(df, cfg, m, smap.get(m, "none"), s, a.trials, a.customer_fraction, verbose=True); path.write_text(json.dumps(results, indent=2))
    best, summ = summarize(results); (out / "best_params.json").write_text(json.dumps(best, indent=2))
    print("\nBest settings per model:")
    for m in best: print(" ", m, best[m]); print("     best mean fold AP per tuning seed:", {k: round(v, 4) for k, v in summ[m]["best_value_per_seed"].items()}, " spread %.4f" % summ[m]["spread"])

if __name__ == "__main__":
    main()
