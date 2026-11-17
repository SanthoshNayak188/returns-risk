# Predicting product line returns in online retail (UCI Online Retail II)

ES 678 individual project. Author: Santhosh Lakavath, roll number 26210073, IIT Gandhinagar.
Data: UCI Online Retail II (CC BY 4.0), about 1.07 million transaction lines from December 2009 to December 2011. The data is not included in this repository. To reproduce, download "Online Retail II" from the UCI Machine Learning Repository, unzip it, rename the file to online_retail_II.xlsx if it has another extension, and place it in data/raw/. Then follow the steps below.
Note:  The first two commits if  shows the Git author name "Malla Jeevan Kumar" because the Git user name was set wrongly in Colab. The commits are unchanged on purpose: their hashes and timestamps are the evidence that all choices were fixed before the test run (tag freeze-before-test). The author of this project is Santhosh Lakavath.
## Phase 1 (do this first, it takes minutes)
1. `python -m venv venv`, activate it, then `pip install -r requirements.txt`.
2. Download Online Retail II from the UCI repository (search "Online Retail II UCI"). It is free, needs no login, and the licence is CC BY 4.0.
   Unzip it and put `online_retail_II.xlsx` in `data/raw/` (or a CSV with the same columns; change `raw_path` in `config.yaml`).
3. `python -m pytest -q tests` should show `22 passed` (the tests use generated data, not the real file).
4. `python src/audit.py` and write one line in `decision_log.md` for every issue found. The first run reads the Excel file, which takes several
   minutes, and caches it as `data/interim/online_retail_II.csv`.
5. `python src/etl.py --fresh` builds the star schema, loads 30 trading days one at a time, then the rest, and checks every batch.
6. `python src/match.py` builds the return label.
7. `python src/label_audit.py` is the decision point. Read the table, the two gates and the recommendation. Then open
   `artifacts/label_audit_samples.csv` and read the matched and unmatched credit lines yourself.

## What the gates mean
Gate 1: at least 50 percent of credit lines can be matched to a purchase, and at least 90 percent of the matches found with the longest window (180 days)
lie inside the chosen window. Gate 2: at least 3,000 positive purchase lines in the development period and 800 in the test period.
If a gate fails, the audit says which test failed and what to try. Any change goes in `config.yaml` and `decision_log.md` before any model is compared.

Raw data, the warehouse file and credentials are never committed (see `.gitignore`).

## Phase 2a: features and first models (after the label audit passes)
8. `python src/features.py` builds 26 point-in-time features per purchase line and saves `data/processed/features.pkl`.
9. `python src/run_cv.py --sanity --boot 200` compares the three baselines, a small logistic regression, a full logistic regression and LightGBM on
   four time-ordered folds (development period only, 90 day embargo). `--compare-strategies` repeats it for the three imbalance strategies.
Every feature uses only information available when the invoice is placed; return-rate history uses only purchases at least 90 days old.

## Phase 2b: choosing the imbalance strategy, tuning and the model (development period only)
10. `python src/choose_strategy.py` applies the strategy rule to the three runs of `run_cv.py --compare-strategies`.
11. `python src/tune.py --trials 25` runs Optuna for logistic regression and LightGBM under the chosen strategy (three tuning seeds, half of the customers).
12. `python src/run_cv.py --models logistic,lightgbm --strategy-map artifacts/strategy_map.json --params artifacts/best_params.json --extra-seeds 2,3,4,5`
13. `python src/select_model.py` applies the rule fixed in the proposal: the simplest model within one standard error of the best mean fold average precision.
The five-feature logistic regression stays a baseline. The test period is not used anywhere in this phase.

## Phase 2c: calibration and the single scored test run
14. `python src/calibrate.py` compares three corrections of the inflated probabilities (analytic, Platt, isotonic) with forward chaining on the development folds and applies the rule.
15. Freeze: commit `config.yaml`, `decision_log.md` and `artifacts/*.json` and tag the commit `freeze-before-test`.
16. `python src/final.py --boot 1000` fits everything on the development period, scores the test period ONCE and writes the results, the saved model and a lock file. It refuses to run if a decision file is missing or the lock exists.

## Phase 3: graduate analyses and figures
17. `python src/analyses.py ablation` and `python src/analyses.py labels` (development folds only) and, after the test run, `python src/analyses.py audit` and `python src/analyses.py drift`.
18. `python src/make_figures.py` draws the three paper figures into `paper/figures/` from the saved results.
Run `final.py` BEFORE unpacking this phase, so that the code that scores the test period is exactly the frozen commit (`git status` shows no changes).

## AI-use disclosure

I used Claude (Anthropic) to plan the project, write and test the Python code (warehouse, label matching, features, models, evaluation, analyses and figures) and draft the paper. I downloaded the dataset and ran the data audit, the cleaning, all experiments and the evaluation myself in Google Colab, and every number in the paper comes from those outputs.

## Citations

Dataset: Online Retail II, UCI Machine Learning Repository, 2019 (CC BY 4.0). Libraries listed in requirements.txt: pandas, NumPy, SciPy, scikit-learn, LightGBM, Optuna, Matplotlib. Methods cited in the paper: Saito and Rehmsmeier 2015, Dal Pozzolo et al. 2015, Niculescu-Mizil and Caruana 2005.

## Notes

The artifacts folder holds small derived tables; label_audit_samples.csv and audit_errors.csv contain a few dozen example rows from the public dataset (CC BY 4.0), shown to document the label checks. The full data is not included.
