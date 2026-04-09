# -*- coding: utf-8 -*-
"""
Spacebook API Explorer — Checkpoint 1
======================================
Hits every Spacebook endpoint, saves raw samples to tools/spacebook_samples/,
and prints annotated schema analysis to stdout.

Run from the ASTRA root:
    python tools/explore_spacebook_apis.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

# ── Encoding safety on Windows ────────────────────────────────────────────────
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ── Config ────────────────────────────────────────────────────────────────────
SB_BASE   = "https://spacebook.com/api"
TIMEOUT   = 30
SAMPLES   = Path(__file__).parent / "spacebook_samples"
SAMPLES.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": "ASTRA-Core/schema-explorer (https://github.com/ISHANTARE/ASTRA)",
    "Accept": "application/json, text/plain, */*",
}

ISS_NORAD = 25544          # Known NORAD ID — used to test GUID lookup
RESULTS: list[tuple[str, bool, str]] = []   # (label, ok, notes)


# ── Helpers ───────────────────────────────────────────────────────────────────

def sep(title: str) -> None:
    print(f"\n{'='*68}")
    print(f"  {title}")
    print(f"{'='*68}")


def fetch(label: str, url: str, params: dict | None = None,
          save_as: str | None = None) -> requests.Response | None:
    """GET url, print outcome, optionally save raw body. Returns Response or None."""
    print(f"\n  >> {label}")
    print(f"     URL: {url}")
    if params:
        print(f"     Params: {params}")
    try:
        t0 = time.monotonic()
        r = requests.get(url, params=params, headers=HEADERS, timeout=TIMEOUT)
        elapsed = time.monotonic() - t0
        print(f"     HTTP {r.status_code} | {len(r.content):,} bytes | {elapsed:.2f}s")

        if r.status_code != 200:
            print(f"     [WARN] Non-200 response. Body: {r.text[:300]}")
            RESULTS.append((label, False, f"HTTP {r.status_code}"))
            return None

        if save_as:
            path = SAMPLES / save_as
            path.write_bytes(r.content)
            print(f"     [SAVED] → {path}")

        RESULTS.append((label, True, f"{len(r.content):,} bytes"))
        return r

    except requests.Timeout:
        print(f"     [TIMEOUT] after {TIMEOUT}s")
        RESULTS.append((label, False, "Timeout"))
        return None
    except Exception as exc:
        print(f"     [ERROR] {exc}")
        RESULTS.append((label, False, str(exc)))
        return None


def analyze_json(r: requests.Response, label: str, max_records: int = 3) -> dict | list | None:
    """Parse JSON, print schema of first record(s), return parsed object."""
    try:
        data = r.json()
    except Exception as exc:
        print(f"     [JSON PARSE FAIL] {exc}")
        print(f"     Raw (200 chars): {r.text[:200]}")
        return None

    if isinstance(data, list):
        print(f"     JSON array — {len(data):,} records")
        for i, rec in enumerate(data[:max_records]):
            print(f"\n     -- Record {i} keys: {list(rec.keys()) if isinstance(rec, dict) else type(rec)}")
            if isinstance(rec, dict):
                for k, v in list(rec.items())[:20]:
                    print(f"          {k:35s}: {repr(v)[:80]}")
        return data

    elif isinstance(data, dict):
        print(f"     JSON object — keys: {list(data.keys())[:20]}")
        for k, v in list(data.items())[:20]:
            print(f"          {k:35s}: {repr(v)[:80]}")
        return data

    else:
        print(f"     JSON type: {type(data)} — value: {repr(data)[:200]}")
        return data


def analyze_text(r: requests.Response, lines: int = 12) -> None:
    """Print first N lines of a text response."""
    for i, line in enumerate(r.text.splitlines()[:lines]):
        print(f"     {i+1:>3}: {line}")
    total = len(r.text.splitlines())
    if total > lines:
        print(f"     ... ({total - lines} more lines)")


def is_tle_text(text: str) -> bool:
    """Heuristic: first data line starts with object name, followed by 1/2 TLE lines."""
    lines = [l for l in text.splitlines() if l.strip()]
    if len(lines) < 3:
        return False
    return lines[1].strip().startswith("1 ") and lines[2].strip().startswith("2 ")


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — Catalog Endpoints (no GUID needed)
# ═══════════════════════════════════════════════════════════════════════════════
sep("PHASE 1 — Bulk Catalog Endpoints (no GUID required)")

# 1a. Standard TLE
r_tle = fetch("Standard TLE catalog", f"{SB_BASE}/entity/tle", save_as="tle_catalog.txt")
if r_tle:
    print(f"\n     TLE format check: {is_tle_text(r_tle.text)}")
    analyze_text(r_tle, lines=9)

# 1b. Historical TLE (use yesterday's date to guarantee data exists)
hist_date = "2026-04-08"
r_hist = fetch(f"Historical TLE ({hist_date})", f"{SB_BASE}/entity/tle/{hist_date}",
               save_as=f"historical_tle_{hist_date}.txt")
if r_hist:
    print(f"\n     TLE format check: {is_tle_text(r_hist.text)}")
    analyze_text(r_hist, lines=9)

# 1c. XP-TLE
r_xptle = fetch("XP-TLE catalog", f"{SB_BASE}/entity/xp-tle", save_as="xp_tle_catalog.txt")
if r_xptle:
    print(f"\n     TLE format check: {is_tle_text(r_xptle.text)}")
    analyze_text(r_xptle, lines=9)

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — Environmental Data
# ═══════════════════════════════════════════════════════════════════════════════
sep("PHASE 2 — Environmental Data")

# 2a. Space Weather Recent
r_sw_recent = fetch("Space Weather (recent)", f"{SB_BASE}/spaceweather/recent",
                    save_as="spaceweather_recent.txt")
if r_sw_recent:
    print(f"\n     Content-Type: {r_sw_recent.headers.get('content-type', 'unknown')}")
    analyze_text(r_sw_recent, lines=15)

# 2b. Space Weather Full
r_sw_full = fetch("Space Weather (full)", f"{SB_BASE}/spaceweather/full",
                  save_as="spaceweather_full.txt")
if r_sw_full:
    print(f"\n     Content-Type: {r_sw_full.headers.get('content-type', 'unknown')}")
    analyze_text(r_sw_full, lines=5)

# 2c. EOP Recent
r_eop_recent = fetch("EOP (recent)", f"{SB_BASE}/eop/recent", save_as="eop_recent.txt")
if r_eop_recent:
    print(f"\n     Content-Type: {r_eop_recent.headers.get('content-type', 'unknown')}")
    analyze_text(r_eop_recent, lines=15)

# 2d. EOP Full
r_eop_full = fetch("EOP (full)", f"{SB_BASE}/eop/full", save_as="eop_full.txt")
if r_eop_full:
    analyze_text(r_eop_full, lines=5)

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — Satellite Catalog (GUID map extraction)
# ═══════════════════════════════════════════════════════════════════════════════
sep("PHASE 3 — Satellite Catalog & GUID Extraction")

# 3a. Satcat JSON
r_satcat = fetch("Satellite Catalog (JSON)", f"{SB_BASE}/entity/satcat",
                 save_as="satcat.json")
iss_guid: str | None = None
if r_satcat:
    data = analyze_json(r_satcat, "satcat")
    # Hunt for ISS by NORAD ID
    if isinstance(data, list):
        for rec in data:
            if not isinstance(rec, dict):
                continue
            # Spacebook uses noradId without leading zeros
            nid = rec.get("noradId") or rec.get("norad_id") or rec.get("NORAD_CAT_ID")
            try:
                if int(str(nid)) == ISS_NORAD:
                    iss_guid = rec.get("id") or rec.get("guid") or rec.get("comspocId")
                    print(f"\n     [ISS FOUND] NORAD={ISS_NORAD}")
                    print(f"     Full record: {json.dumps(rec, indent=6)[:1500]}")
                    break
            except (TypeError, ValueError):
                continue
        if iss_guid:
            print(f"\n     ISS GUID: {iss_guid}")
        else:
            print(f"\n     [WARN] ISS (NORAD {ISS_NORAD}) not found in satcat. "
                  f"Checking first 3 record keys for clues...")
            if data:
                print(f"     First record: {json.dumps(data[0], indent=4)[:600]}")

# 3b. Satcat Details JSON
r_satcat_det = fetch("Satellite Catalog Details (JSON)", f"{SB_BASE}/entity/satcat/details",
                     save_as="satcat_details.json")
if r_satcat_det:
    ddata = analyze_json(r_satcat_det, "satcat/details")
    # Try to find ISS here too if not found above
    if iss_guid is None and isinstance(ddata, list):
        for rec in ddata:
            if not isinstance(rec, dict):
                continue
            nid = rec.get("noradId") or rec.get("norad_id") or rec.get("NORAD_CAT_ID")
            try:
                if int(str(nid)) == ISS_NORAD:
                    iss_guid = rec.get("id") or rec.get("guid") or rec.get("comspocId") or rec.get("COMSPOC_ID")
                    print(f"\n     [ISS FOUND in details] GUID: {iss_guid}")
                    print(f"     Full record: {json.dumps(rec, indent=6)[:1500]}")
                    break
            except (TypeError, ValueError):
                continue

# 3c. Satcat CSV (peek at headers)
r_satcat_csv = fetch("Satellite Catalog (CSV)", f"{SB_BASE}/entity/satcat/csv",
                     save_as="satcat.csv")
if r_satcat_csv:
    analyze_text(r_satcat_csv, lines=3)
    # Find COMSPOC_ID column
    header = r_satcat_csv.text.splitlines()[0] if r_satcat_csv.text.strip() else ""
    cols = [c.strip() for c in header.split(",")]
    print(f"\n     CSV columns ({len(cols)}): {cols}")
    if "COMSPOC_ID" in cols:
        print("     [OK] COMSPOC_ID column found — can map NORAD->GUID via CSV")
        # Find ISS row from CSV if GUID still unknown
        if iss_guid is None:
            norad_col = next((i for i, c in enumerate(cols) if "NORAD" in c.upper()), None)
            guid_col  = next((i for i, c in enumerate(cols) if "COMSPOC_ID" in c.upper()), None)
            if norad_col is not None and guid_col is not None:
                for row in r_satcat_csv.text.splitlines()[1:]:
                    parts = row.split(",")
                    try:
                        if int(parts[norad_col].strip()) == ISS_NORAD:
                            iss_guid = parts[guid_col].strip()
                            print(f"     [ISS GUID from CSV] {iss_guid}")
                            break
                    except (ValueError, IndexError):
                        continue

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 4 — Per-Object (GUID-based) Endpoints
# ═══════════════════════════════════════════════════════════════════════════════
sep("PHASE 4 — Per-Object Endpoints (GUID required)")

if iss_guid:
    print(f"\n  Using ISS GUID: {iss_guid}\n")

    # 4a. Synthetic Covariance
    r_cov = fetch("Synthetic Covariance (ISS)", f"{SB_BASE}/entity/synthetic-covariance/{iss_guid}",
                  save_as="iss_synthetic_covariance.json")
    if r_cov:
        analyze_json(r_cov, "synthetic-covariance")

    # 4b. Synthetic Covariance Plotly graph
    r_plotly = fetch("Synthetic Covariance Plotly (ISS)",
                     f"{SB_BASE}/entity/synthetic-covariance-plotly/{iss_guid}",
                     save_as="iss_covariance_plotly.json")
    if r_plotly:
        analyze_json(r_plotly, "covariance-plotly")

    # 4c. Reference Ephemeris (default format)
    r_eph = fetch("Reference Ephemeris (ISS)", f"{SB_BASE}/entity/reference-ephemerides/{iss_guid}",
                  save_as="iss_reference_ephemeris.txt")
    if r_eph:
        print(f"\n     Content-Type: {r_eph.headers.get('content-type', 'unknown')}")
        analyze_text(r_eph, lines=20)

    # 4d. Reference Ephemeris OCM format
    r_ocm = fetch("Reference Ephemeris OCM (ISS)",
                  f"{SB_BASE}/entity/reference-ephemerides/ocm/{iss_guid}",
                  save_as="iss_reference_ephemeris_ocm.txt")
    if r_ocm:
        print(f"\n     Content-Type: {r_ocm.headers.get('content-type', 'unknown')}")
        analyze_text(r_ocm, lines=30)

else:
    print("\n  [SKIP] ISS GUID not resolved — skipping per-object endpoints.")
    print("         Check satcat.json / satcat.csv to find the correct field name.")
    RESULTS.append(("Per-object GUID endpoints", False, "ISS GUID not resolved"))

# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
sep("SUMMARY")
passed = sum(1 for _, ok, _ in RESULTS if ok)
failed = len(RESULTS) - passed
print(f"\n  Endpoints hit: {len(RESULTS)}   Passed: {passed}   Failed: {failed}")
print()
for label, ok, notes in RESULTS:
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {label:55s} {notes}")

print(f"\n  Samples saved to: {SAMPLES.resolve()}")
print(f"  ISS GUID resolved: {iss_guid or 'NO — see satcat files'}")

# Save guid for downstream use
if iss_guid:
    guid_file = SAMPLES / "iss_guid.txt"
    guid_file.write_text(iss_guid, encoding="utf-8")
    print(f"  ISS GUID written to: {guid_file.resolve()}")

print()
