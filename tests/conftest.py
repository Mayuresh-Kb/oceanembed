from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SAMPLE_NC = ROOT / "data" / "cmems_mod_glo_phy_my_0.083deg_P1D-m_1788172075365.nc"


@pytest.fixture(scope="session")
def sample_nc() -> Path:
    return SAMPLE_NC
