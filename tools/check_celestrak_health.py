# -*- coding: utf-8 -*-
"""ASTRA CelesTrak Health Check — tests all endpoints ASTRA depends on."""
import sys
import requests

# Force UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

HEADERS = {"User-Agent": "ASTRA-Core/diagnostic (https://github.com/ISHANTARE/ASTRA)"}
TIMEOUT = 20
RESULTS = []


def check(label, url, params=None, min_bytes=100, expect_text=None, bad_text=None):
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=TIMEOUT)
        size = len(r.content)
        ok = True
        issues = []

        if r.status_code != 200:
            ok = False
            issues.append(f"HTTP {r.status_code}")
        if size < min_bytes:
            ok = False
            issues.append(f"only {size} bytes (expected >= {min_bytes})")
        if expect_text and expect_text not in r.text:
            ok = False
            issues.append(f"missing expected content: '{expect_text}'")
        if bad_text and bad_text.lower() in r.text.lower():
            ok = False
            issues.append(f"contains bad content: '{bad_text}'")

        tag = "PASS" if ok else "FAIL"
        snippet = r.text[:130].strip().replace("\n", " ")
        RESULTS.append((label, ok, issues))
        print(f"  [{tag}] {label}")
        print(f"         HTTP {r.status_code} | {size:,} bytes")
        print(f"         {snippet}")
        if not ok:
            for i in issues:
                print(f"         *** {i}")
    except requests.Timeout:
        print(f"  [TOUT] {label}  =>  Timed out after {TIMEOUT}s")
        RESULTS.append((label, False, ["Timed out"]))
    except Exception as e:
        print(f"  [ERR ] {label}  =>  {e}")
        RESULTS.append((label, False, [str(e)]))
    print()


print("=" * 68)
print("  ASTRA CelesTrak Health Check")
print("=" * 68)

# ── 1. Main GP Catalog (gp.php) ────────────────────────────────
print("\n-- [1] Main GP Catalog: gp.php --\n")

check("gp.php | starlink | TLE",
      "https://celestrak.org/NORAD/elements/gp.php",
      params={"GROUP": "starlink", "FORMAT": "tle"},
      min_bytes=500, expect_text="STARLINK", bad_text="invalid query")

check("gp.php | starlink | OMM/JSON",
      "https://celestrak.org/NORAD/elements/gp.php",
      params={"GROUP": "starlink", "FORMAT": "json"},
      min_bytes=500, expect_text="OBJECT_NAME", bad_text="invalid query")

check("gp.php | active | TLE",
      "https://celestrak.org/NORAD/elements/gp.php",
      params={"GROUP": "active", "FORMAT": "tle"},
      min_bytes=1000, bad_text="invalid query")

check("gp.php | gps-ops | TLE",
      "https://celestrak.org/NORAD/elements/gp.php",
      params={"GROUP": "gps-ops", "FORMAT": "tle"},
      min_bytes=200, bad_text="invalid query")

check("gp.php | iridium-33-debris | TLE",
      "https://celestrak.org/NORAD/elements/gp.php",
      params={"GROUP": "iridium-33-debris", "FORMAT": "tle"},
      min_bytes=200, bad_text="invalid query")

check("gp.php | cosmos-2251-debris | TLE",
      "https://celestrak.org/NORAD/elements/gp.php",
      params={"GROUP": "cosmos-2251-debris", "FORMAT": "tle"},
      min_bytes=200, bad_text="invalid query")

# ── 2. Supplemental GP Catalog (sup-gp.php) ────────────────────
print("\n-- [2] Supplemental GP: sup-gp.php --\n")

check("sup-gp.php | starlink | TLE",
      "https://celestrak.org/NORAD/elements/supplemental/sup-gp.php",
      params={"FILE": "starlink", "FORMAT": "TLE"},
      min_bytes=500, expect_text="STARLINK", bad_text="invalid query")

check("sup-gp.php | starlink | JSON",
      "https://celestrak.org/NORAD/elements/supplemental/sup-gp.php",
      params={"FILE": "starlink", "FORMAT": "JSON"},
      min_bytes=500, expect_text="OBJECT_NAME", bad_text="invalid query")

check("sup-gp.php | GPS-A (gps-ops) | TLE",
      "https://celestrak.org/NORAD/elements/supplemental/sup-gp.php",
      params={"SOURCE": "GPS-A", "FORMAT": "TLE"},
      min_bytes=200, bad_text="invalid query")

# ── 3. Space Weather (data_pipeline.py fallback) ───────────────
print("\n-- [3] Space Weather: SW-All.csv --\n")

check("SW-All.csv | CelesTrak SpaceData",
      "https://celestrak.org/SpaceData/SW-All.csv",
      min_bytes=5000, expect_text="DATE", bad_text="not found")

# ── Summary ────────────────────────────────────────────────────
print("=" * 68)
print("  SUMMARY")
print("=" * 68)
passed = sum(1 for _, ok, _ in RESULTS if ok)
failed = len(RESULTS) - passed
print(f"  Passed : {passed}/{len(RESULTS)}")
print(f"  Failed : {failed}/{len(RESULTS)}")
print()
for label, ok, issues in RESULTS:
    tag = "PASS" if ok else "FAIL"
    suffix = ("  ==>  " + " | ".join(issues)) if not ok else ""
    print(f"  [{tag}] {label}{suffix}")
print("=" * 68)
sys.exit(0 if failed == 0 else 1)
