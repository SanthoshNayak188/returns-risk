import subprocess, sys, pathlib
import match, etl, raw
from make_fake_data import make
from utils import load_config
SRC = pathlib.Path(__file__).resolve().parents[1] / "src"

def run(args):
    r = subprocess.run([sys.executable] + [str(a) for a in args], capture_output=True, text=True, cwd=str(SRC))
    assert r.returncode == 0, "%s failed:\n%s\n%s" % (args[0], r.stdout[-700:], r.stderr[-1500:]); return r.stdout

def test_features_and_comparison_run_from_the_command_line(tmp_path, fake_rs):
    import shutil; shutil.copy(fake_rs["dir"] / "w.db", tmp_path / "w.db")
    out = run(["features.py", "--db", tmp_path / "w.db", "--out", tmp_path / "f.pkl"]); assert "features 26" in out and (tmp_path / "f.pkl").exists()
    out = run(["run_cv.py", "--features", tmp_path / "f.pkl", "--models", "B0_prior,B1b_product_rate,logistic,lightgbm", "--sanity", "--boot", "30",
               "--out-dir", tmp_path / "art", "--oof", tmp_path / "oof.pkl"])
    assert "Permuted-label check" in out and "PASS" in out and "Signal check" in out and "Pooled over all folds" in out and "Customer-cluster bootstrap" in out
    out = run(["run_cv.py", "--features", tmp_path / "f.pkl", "--models", "B0_prior,logistic", "--compare-strategies", "--boot", "20",
               "--out-dir", tmp_path / "art", "--oof", tmp_path / "oof.pkl"])
    assert out.count("imbalance strategy") == 3
