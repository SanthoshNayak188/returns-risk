"""Draws the three paper figures from the saved results. Panels whose input file is missing are left empty with a note.
Usage: python src/make_figures.py [--artifacts artifacts] [--db warehouse/retail.db] [--test-preds data/processed/test_preds.pkl] [--out paper/figures]"""
import argparse, json, pathlib, sqlite3
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.ticker
from utils import ROOT

plt.rcParams.update({"font.size": 6.5, "axes.titlesize": 7, "axes.labelsize": 6.5, "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 5.5,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 200})
TEAL, NAVY, GREY, RED = "#2a7f8e", "#1d3557", "#9aa5b1", "#c0392b"
NAMES = {"B0_prior": "constant", "B1a_customer_rate": "customer rate", "B1b_product_rate": "product rate", "B2_logistic_small": "5-feature logistic", "logistic": "logistic", "lightgbm": "LightGBM"}

def _missing(ax, title): ax.set_title(title); ax.text(0.5, 0.5, "not available", ha="center", va="center", transform=ax.transAxes); ax.set_xticks([]); ax.set_yticks([])
def _read(p): return pd.read_csv(p) if pathlib.Path(p).exists() else None

def fig_data(art, db, out):
    fig, ax = plt.subplots(1, 3, figsize=(5.5, 1.6)); d = _read(art / "drift_monthly.csv")
    if d is None: _missing(ax[0], "(a) Return rate by month")
    else:
        d = d.sort_values("month"); x = np.arange(len(d)); c = np.where(d.split == "test", NAVY, TEAL)
        ax[0].bar(x, 100 * d.base_rate, color=c, width=0.8); ax[0].set_xticks(x[::6]); ax[0].set_xticklabels([m[2:] for m in d.month[::6]], rotation=0)
        ax[0].set_ylabel("lines returned (%)"); ax[0].set_title("(a) Return rate by month")
    if db is not None and pathlib.Path(db).exists():
        days = pd.read_sql("SELECT days_to_return FROM bridge_return_match", sqlite3.connect(str(db)))["days_to_return"]
        ax[1].hist(days, bins=np.arange(0, 95, 5), color=TEAL, edgecolor="white", linewidth=0.3); ax[1].set_xlabel("days from purchase to credit"); ax[1].set_ylabel("matches"); ax[1].set_title("(b) Time to return")
    else: _missing(ax[1], "(b) Time to return")
    w = _read(art / "label_audit_windows.csv")
    if w is None: _missing(ax[2], "(c) Credit lines matched")
    else:
        ax[2].bar(w.window_days.astype(str), 100 * w.matched_credit_line_share, color=[NAVY if x == 90 else TEAL for x in w.window_days], width=0.7)
        ax[2].set_ylim(60, 95); ax[2].set_xlabel("window (days)"); ax[2].set_ylabel("credit lines matched (%)"); ax[2].set_title("(c) Label window")
    fig.tight_layout(w_pad=1.4); p = out / "fig_data.png"; fig.savefig(p, bbox_inches="tight"); plt.close(fig); return p

def fig_results(art, preds_path, out):
    fig, ax = plt.subplots(1, 3, figsize=(5.5, 1.6)); b = _read(art / "test_bootstrap.csv"); ch = json.load(open(art / "model_choice.json"))["chosen"] if (art / "model_choice.json").exists() else "logistic"
    if b is None: _missing(ax[0], "(a) Average precision")
    else:
        b = b.set_index("model").loc[list(NAMES)]; y = np.arange(len(b))[::-1]; lab = [NAMES[m] + (" (chosen)" if m == ch else "") for m in b.index]
        ax[0].barh(y, b.ap, xerr=[b.ap - b.ap_lo, b.ap_hi - b.ap], color=[NAVY if m == ch else TEAL if m not in ("B0_prior", "B1a_customer_rate", "B1b_product_rate") else GREY for m in b.index], error_kw={"lw": 0.6, "capsize": 1.5})
        ax[0].axvline(b.loc["B0_prior", "ap"], color=RED, lw=0.7, ls="--"); ax[0].set_yticks(y); ax[0].set_yticklabels(lab); ax[0].set_xlabel("average precision (test)"); ax[0].set_title("(a) Ranking quality")
    u, c = _read(art / "test_reliability_uncalibrated.csv"), _read(art / "test_reliability_calibrated.csv")
    if u is None or c is None: _missing(ax[1], "(b) Calibration by decile")
    else:
        lo, hi = min(u.mean_predicted.min(), c.mean_predicted.min(), c.observed_rate.min()) * 0.8, max(u.mean_predicted.max(), c.observed_rate.max()) * 1.2
        ax[1].plot([lo, hi], [lo, hi], color="black", lw=0.5); ax[1].plot(u.mean_predicted, u.observed_rate, "o-", color=GREY, ms=2, lw=0.7, label="as trained"); ax[1].plot(c.mean_predicted, c.observed_rate, "o-", color=TEAL, ms=2, lw=0.9, label="corrected")
        ax[1].set_xscale("log"); ax[1].set_yscale("log"); ticks = [x for x in (0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5) if lo / 1.3 <= x <= hi * 1.3]
        for axis in (ax[1].xaxis, ax[1].yaxis): axis.set_major_locator(matplotlib.ticker.FixedLocator(ticks)); axis.set_major_formatter(matplotlib.ticker.FixedFormatter(["%g%%" % (100 * x) for x in ticks])); axis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax[1].set_xlabel("mean predicted"); ax[1].set_ylabel("observed rate"); ax[1].legend(frameon=False, loc="upper left"); ax[1].set_title("(b) Calibration by decile")
    if preds_path is not None and pathlib.Path(preds_path).exists():
        t = pd.read_pickle(preds_path).sort_values(ch + "_calibrated", ascending=False); n = len(t); k = int(0.3 * n); cum = np.cumsum(t.y.to_numpy())[:k] / t.y.sum()
        ax[2].plot(np.arange(1, k + 1) / n * 100, cum * 100, color=TEAL, lw=1.0, label="model"); ax[2].plot([0, 30], [0, 30], color="black", lw=0.5, ls="--", label="random")
        ax[2].set_xlabel("lines flagged (%)"); ax[2].set_ylabel("returns found (%)"); ax[2].legend(frameon=False, loc="lower right"); ax[2].set_title("(c) Returns caught")
    else: _missing(ax[2], "(c) Returns caught")
    fig.tight_layout(w_pad=1.2); p = out / "fig_results.png"; fig.savefig(p, bbox_inches="tight"); plt.close(fig); return p

def fig_graduate(art, out):
    fig, ax = plt.subplots(1, 3, figsize=(5.5, 1.6)); a = _read(art / "ablation.csv")
    if a is None: _missing(ax[0], "(a) Feature groups removed")
    else:
        full_ap = float(a.loc[a.variant == "full", "ap"].iloc[0]); a = a[~a.variant.isin(["full", "history only"])].copy()
        a["label"] = a.variant.str.replace("without ", "", regex=False).str.replace("_", " ", regex=False); a["pct"] = a.ap_change_percent; a["lo"] = 100 * a.change_lo / full_ap; a["hi"] = 100 * a.change_hi / full_ap
        y = np.arange(len(a))[::-1]; ax[0].barh(y, a.pct, color=TEAL, xerr=[(a.pct - a.lo).clip(lower=0), (a.hi - a.pct).clip(lower=0)], error_kw={"lw": 0.6, "capsize": 1.5})
        ax[0].axvline(0, color="black", lw=0.5); ax[0].set_yticks(y); ax[0].set_yticklabels(a.label); ax[0].set_xlabel("change in average precision (%)"); ax[0].set_title("(a) Group removed")
    l = _read(art / "label_sensitivity.csv")
    if l is None: _missing(ax[1], "(b) Label definition")
    else:
        short = [s.replace("window ", "w").replace(" days, oldest purchase first", " oldest").replace(" days, most recent purchase first", " recent") for s in l.label_definition]; x = np.arange(len(l))
        ax[1].bar(x, l.lift, color=[NAVY if "90 oldest" in s else TEAL for s in short], yerr=[l.lift - l.lift_lo, l.lift_hi - l.lift], error_kw={"lw": 0.6, "capsize": 1.5}, width=0.7)
        ax[1].set_xticks(x); ax[1].set_xticklabels(short, rotation=35, ha="right"); ax[1].set_ylabel("lift over base rate"); ax[1].set_title("(b) Label definition")
    s = _read(art / "audit_slices.csv")
    if s is None: _missing(ax[2], "(c) Calibration by slice")
    else:
        keep = ["customer: first invoice in the data", "customer: earlier invoices, none old enough to know", "customer: some history old enough to know", "product: no history old enough to know", "product: some history old enough to know", "country: United Kingdom", "country: other"]
        s = s.set_index("slice").loc[[k for k in keep if k in set(s.slice)]]; lab = ["new customer", "customer, no old history", "customer with history", "new product", "product with history", "UK", "other country"][: len(s)]; y = np.arange(len(s))[::-1]
        ax[2].barh(y, s.predicted_over_observed, color=TEAL); ax[2].axvline(1.0, color=RED, lw=0.7, ls="--"); ax[2].set_yticks(y); ax[2].set_yticklabels(lab); ax[2].set_xlabel("predicted / observed rate"); ax[2].set_title("(c) Calibration by slice")
    fig.tight_layout(w_pad=1.2); p = out / "fig_graduate.png"; fig.savefig(p, bbox_inches="tight"); plt.close(fig); return p

def make_all(art, db, preds_path, out):
    art, out = pathlib.Path(art), pathlib.Path(out); out.mkdir(parents=True, exist_ok=True)
    return [fig_data(art, db, out), fig_results(art, preds_path, out), fig_graduate(art, out)]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--artifacts", default=str(ROOT / "artifacts")); ap.add_argument("--db", default=str(ROOT / "warehouse" / "retail.db"))
    ap.add_argument("--test-preds", default=str(ROOT / "data" / "processed" / "test_preds.pkl")); ap.add_argument("--out", default=str(ROOT / "paper" / "figures")); a = ap.parse_args()
    for p in make_all(a.artifacts, a.db, a.test_preds, a.out): print("wrote", p)

if __name__ == "__main__":
    main()
