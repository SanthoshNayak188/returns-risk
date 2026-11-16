audit:
	python src/audit.py
etl:
	python src/etl.py --fresh
match:
	python src/match.py
label:
	python src/label_audit.py
test:
	python -m pytest -q tests
features:
	python src/features.py
cv:
	python src/run_cv.py --sanity
strategy:
	python src/choose_strategy.py
tune:
	python src/tune.py
select:
	python src/select_model.py
calibrate:
	python src/calibrate.py
final:
	python src/final.py --boot 1000
