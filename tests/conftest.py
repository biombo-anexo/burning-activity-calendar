import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from burncal.pipeline import build  # noqa: E402
from burncal.synthetic import generate  # noqa: E402


@pytest.fixture(scope="session")
def synth(tmp_path_factory):
    """Small synthetic world processed end to end. Returns dict with paths."""
    root = tmp_path_factory.mktemp("synth")
    raw, out = root / "raw", root / "processed"
    info = generate(raw, scale=0.3, seed=7)
    calib = build([raw / "MODIS_synthetic.csv"], [raw / "VIIRS_synthetic.csv"], out, label="TEST", log=lambda *_: None)
    return {"raw": raw, "out": out, "calib": calib, "info": info}
