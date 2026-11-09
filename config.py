"""Loads system.yaml and checks it, so a typo fails early with a clear message."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"


def load_system(path: Path | str | None = None) -> dict:
    path = Path(path or ROOT / "system.yaml")
    with open(path, encoding="utf-8") as f:
        s = yaml.safe_load(f)
    loc, pan, ori = s["location"], s["panels"], s["orientation"]
    if not -90 <= loc["latitude"] <= 90 or not -180 <= loc["longitude"] <= 180:
        raise ValueError("latitude must be -90..90 and longitude -180..180")
    if pan["count"] <= 0 or pan["area_m2"] <= 0 or not 0 < pan["efficiency"] < 0.5:
        raise ValueError("panels: count and area must be positive, efficiency between 0 and 0.5")
    if not 0 <= ori["tilt_deg"] <= 90 or not 0 <= ori["azimuth_deg"] < 360:
        raise ValueError("orientation: tilt 0..90, azimuth 0..359")
    if not 0 <= s["losses"] < 1:
        raise ValueError("losses must be a fraction between 0 and 1")
    return s


def peak_kw(system: dict) -> float:
    """Nameplate power at standard test conditions (1000 W/m2)."""
    p = system["panels"]
    return p["count"] * p["area_m2"] * p["efficiency"]
