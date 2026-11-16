import subprocess, sys, pathlib, json
from make_fake_data import make
SRC = pathlib.Path(__file__).resolve().parents[1] / "src"

def run(args):
    r = subprocess.run([sys.executable] + [str(a) for a in args], capture_output=True, text=True, cwd=str(SRC))
    assert r.returncode == 0, "%s failed:\n%s\n%s" % (args[0], r.stdout[-700:], r.stderr[-1500:])
    return r.stdout

def test_the_phase_one_scripts_run_in_order_from_the_command_line(tmp_path):
    make(tmp_path / "tx.csv", n_customers=150); art = tmp_path / "art"
    out = run(["audit.py", "--raw", tmp_path / "tx.csv", "--out-dir", art]); assert "credit_line_share" in out and (art / "audit_report.json").exists()
    out = run(["etl.py", "--raw", tmp_path / "tx.csv", "--db", tmp_path / "w.db", "--fresh"]); assert "failed reconciliation checks: 0" in out
    out = run(["match.py", "--db", tmp_path / "w.db"]); assert "Conservation violations (must be 0): 0" in out
    out = run(["label_audit.py", "--db", tmp_path / "w.db", "--out-dir", art])
    assert "Gate 1" in out and "Gate 2" in out and "Recommendation" in out
    assert all((art / f).exists() for f in ["label_audit.json", "label_audit_windows.csv", "label_audit_samples.csv"])
