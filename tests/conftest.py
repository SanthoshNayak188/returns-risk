import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src")); sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import pytest
from make_fake_data import make
import raw, etl
from utils import load_config

@pytest.fixture(scope="session")
def fake(tmp_path_factory):
    d = tmp_path_factory.mktemp("fake"); truth = make(d / "tx.csv", n_customers=300)
    df = raw.load_raw(d / "tx.csv"); conn = etl.connect(d / "w.db"); etl.build(conn, df, load_config(), replay_days=5)
    return {"df": df, "conn": conn, "truth": truth, "dir": d}

@pytest.fixture(scope="session")
def fake_rs(tmp_path_factory):
    """A warehouse with a PLANTED risk signal (risky products and customers), its label, and its features."""
    import match, features
    d = tmp_path_factory.mktemp("fake_rs"); make(d / "tx.csv", n_customers=500, invoices_per_day=10, risk_signal=True, return_rate=0.03, seed=5)
    cfg = load_config(); df = raw.load_raw(d / "tx.csv"); conn = etl.connect(d / "w.db"); etl.build(conn, df, cfg, replay_days=3)
    match.run_matching(conn, cfg["label"]["window_days"]); feats = features.build_features(conn, cfg)
    return {"conn": conn, "feats": feats, "cfg": cfg, "dir": d}
