#!/usr/bin/env python3
"""Offline validation: CelesTrak sup-gp.php TLE/JSON vs ASTRA parsers + SGP4.

Run from repo root: python tools/validate_supgp_sample.py

Exits 0 if checks pass; prints diagnostics. Requires network.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import requests

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from astra.omm import parse_omm_json  # noqa: E402
from astra.orbit import propagate_orbit  # noqa: E402
from astra.tle import load_tle_catalog, parse_tle, validate_tle  # noqa: E402

SUP_GP = "https://celestrak.org/NORAD/elements/supplemental/sup-gp.php"
HEADERS = {
    "User-Agent": (
        "ASTRA-Core/validation (https://github.com/ISHANTARE/ASTRA; sup-gp check)"
    ),
}


def _first_tle_triplet(lines: list[str]) -> tuple[str, str, str]:
    """Return (name, l1, l2) from 3-line or multi-record TLE blob."""
    raw = [ln.rstrip() for ln in lines if ln.strip()]
    i = 0
    while i + 2 < len(raw):
        a, b, c = raw[i], raw[i + 1], raw[i + 2]
        if b.startswith("1 ") and c.startswith("2 ") and len(b) >= 64 and len(c) >= 64:
            name = a.strip() if not a.startswith("1 ") else f"NORAD-{b[2:7].strip()}"
            if not a.startswith("1 "):
                return (a.strip(), b, c)
            return (f"NORAD-{b[2:7].strip()}", b, c)
        i += 1
    raise ValueError("No 3-line TLE found in response")


def main() -> int:
    print("=== CelesTrak sup-gp.php data validation ===\n")

    # --- ISS (25544): TLE ---
    r_tle = requests.get(
        SUP_GP,
        params={"CATNR": "25544", "FORMAT": "TLE"},
        headers=HEADERS,
        timeout=60,
    )
    print(f"ISS TLE: HTTP {r_tle.status_code}, {len(r_tle.content)} bytes")
    r_tle.raise_for_status()
    name, l1, l2 = _first_tle_triplet(r_tle.text.splitlines())
    if not validate_tle(name, l1, l2):
        print("FAIL: ISS TLE checksum / format validation")
        return 1
    print(f"  Parsed name: {name!r}, NORAD from line1: {l1[2:7].strip()}")

    sat = parse_tle(name, l1, l2)
    st = propagate_orbit(sat, sat.epoch_jd, 0.0)
    r_km = math.sqrt(sum(x * x for x in st.position_km))
    alt_km = r_km - 6378.137
    print(f"  SGP4 r @ epoch: {r_km:.3f} km (alt ~{alt_km:.1f} km)")
    if not (6500 <= r_km <= 7000):
        print("FAIL: ISS radius out of plausible LEO band")
        return 1

    # --- ISS: JSON (OMM keywords) ---
    r_js = requests.get(
        SUP_GP,
        params={"CATNR": "25544", "FORMAT": "JSON"},
        headers=HEADERS,
        timeout=60,
    )
    print(f"\nISS JSON: HTTP {r_js.status_code}, {len(r_js.content)} bytes")
    r_js.raise_for_status()
    omms = parse_omm_json(r_js.text)
    if not omms:
        print("FAIL: empty OMM list")
        return 1
    o0 = omms[0]
    if o0.norad_id.strip() != "25544":
        print(f"FAIL: expected NORAD 25544, got {o0.norad_id!r}")
        return 1
    print(f"  OMM name: {o0.name!r}, epoch_jd={o0.epoch_jd:.8f}")
    st2 = propagate_orbit(o0, o0.epoch_jd, 0.0)
    r2 = math.sqrt(sum(x * x for x in st2.position_km))
    print(f"  SGP4 r @ epoch from OMM: {r2:.3f} km")

    dpos_km = math.sqrt(
        sum(
            (float(st.position_km[i]) - float(st2.position_km[i])) ** 2
            for i in range(3)
        )
    )
    print(f"  |r_tle - r_omm| @ epoch: {dpos_km * 1000:.2f} m")
    if dpos_km > 0.02:
        print(
            f"WARN: TLE vs OMM position delta {dpos_km:.4f} km — "
            "may differ if SupGP returns different ephemeris segments; check multi-epoch blob."
        )

    # --- GPS almanac (SupGP SOURCE), TLE: multiple satellites, MEO geometry ---
    r_gp = requests.get(
        SUP_GP,
        params={"SOURCE": "GPS-A", "FORMAT": "TLE"},
        headers=HEADERS,
        timeout=120,
    )
    print(f"\nSOURCE=GPS-A TLE: HTTP {r_gp.status_code}, {len(r_gp.content)} bytes")
    r_gp.raise_for_status()
    if "Invalid query" in r_gp.text:
        print(f"FAIL: {r_gp.text.strip()}")
        return 1
    tles = load_tle_catalog(r_gp.text.splitlines())
    print(f"  Loaded {len(tles)} SatelliteTLE via load_tle_catalog")
    if len(tles) < 20:
        print("FAIL: expected many GPS almanac entries")
        return 1
    # propagate first
    s0 = tles[0]
    g0 = propagate_orbit(s0, s0.epoch_jd, 0.0)
    rg = math.sqrt(sum(x * x for x in g0.position_km))
    alt_g = rg - 6378.137
    print(
        f"  First sat {s0.name.strip()}: r @ epoch {rg:.3f} km "
        f"(alt ~{alt_g:.1f} km)"
    )
    if not (25000 <= rg <= 29000):
        print("FAIL: GPS MEO radius implausible (expect ~26500 km)")
        return 1

    print("\n=== All sanity checks passed (SupGP data parses and propagates coherently). ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
