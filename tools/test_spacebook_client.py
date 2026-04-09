# -*- coding: utf-8 -*-
"""Checkpoint 3 — Live functional test for astra/spacebook.py.
Run from ASTRA root:  python tools/test_spacebook_client.py
"""
from __future__ import annotations
import sys
from datetime import datetime, timezone, timedelta

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from astra import spacebook

PASS = "[PASS]"
FAIL = "[FAIL]"
RESULTS = []

def check(label, fn, *args, **kwargs):
    try:
        result = fn(*args, **kwargs)
        print(f"\n  {PASS} {label}")
        print(f"         result type  : {type(result).__name__}")
        if isinstance(result, (list, dict)):
            print(f"         result len   : {len(result)}")
        elif isinstance(result, str):
            print(f"         result len   : {len(result)} chars")
            print(f"         preview      : {result[:120].replace(chr(10), ' ')}")
        elif isinstance(result, tuple):
            print(f"         result       : {result}")
        elif isinstance(result, bool):
            print(f"         result       : {result}")
        RESULTS.append((label, True, ""))
        return result
    except Exception as exc:
        print(f"\n  {FAIL} {label}")
        print(f"         error        : {type(exc).__name__}: {exc}")
        RESULTS.append((label, False, str(exc)))
        return None


print("=" * 65)
print("  Checkpoint 3 — astra.spacebook Live Function Tests")
print("=" * 65)

# 1. Connectivity probe
print("\n-- [1] Connectivity --")
check("is_available()", spacebook.is_available)

# 2. Space Weather
print("\n-- [2] Space Weather --")
t_jd_today = 2451545.0 + (datetime.now(timezone.utc) - datetime(2000,1,1,12,tzinfo=timezone.utc)).total_seconds() / 86400.0
sw = check("get_space_weather_sb(today)", spacebook.get_space_weather_sb, t_jd_today)
if sw:
    f107_obs, f107_adj, ap = sw
    print(f"         F10.7_obs={f107_obs:.1f}  F10.7_adj={f107_adj:.1f}  Ap={ap:.1f}")
    assert 50.0 < f107_obs < 500.0, f"Unrealistic F10.7_obs: {f107_obs}"
    assert 0.0 <= ap <= 400.0, f"Unrealistic Ap: {ap}"
    print(f"         [OK] Values are physically plausible.")

# Future date (should raise SpacebookError on miss)
t_jd_future = t_jd_today + 365
try:
    spacebook.get_space_weather_sb(t_jd_future)
    print(f"  {FAIL} get_space_weather_sb(future date) should raise SpacebookError")
    RESULTS.append(("get_space_weather_sb(future date) raises on missing", False, "No exception raised"))
except spacebook.SpacebookError:
    print(f"\n  {PASS} get_space_weather_sb(future date) correctly raises SpacebookError")
    RESULTS.append(("get_space_weather_sb(future date) raises on missing", True, ""))

# 3. EOP
print("\n-- [3] Earth Orientation Parameters --")
eop = check("get_eop_sb(today)", spacebook.get_eop_sb, t_jd_today)
if eop:
    xp, yp, dut1 = eop
    print(f"         xp={xp:.6f} arcsec  yp={yp:.6f} arcsec  dut1={dut1:.7f} s")
    assert abs(xp) < 2.0, f"Unrealistic polar motion xp: {xp}"
    assert abs(yp) < 2.0, f"Unrealistic polar motion yp: {yp}"
    assert abs(dut1) < 2.0, f"Unrealistic UT1-UTC: {dut1}"
    print(f"         [OK] EOP values within nominal range.")

# 4. GUID resolution
print("\n-- [4] GUID Resolution --")
ISS_NORAD = 25544
guid = check(f"get_norad_guid(ISS={ISS_NORAD})", spacebook.get_norad_guid, ISS_NORAD)
if guid:
    print(f"         GUID={guid}")
    assert len(guid) == 36, f"Unexpected GUID length: {len(guid)}"
    assert guid.count("-") == 4, f"Unexpected GUID format: {guid}"
    print(f"         [OK] UUID format validated.")

STARLINK_NORAD = 44714  # STARLINK-1008
guid_sl = check(f"get_norad_guid(Starlink={STARLINK_NORAD})", spacebook.get_norad_guid, STARLINK_NORAD)

# Test unknown NORAD ID raises SpacebookLookupError
try:
    spacebook.get_norad_guid(9999999)
    print(f"  {FAIL} get_norad_guid(unknown) should raise SpacebookLookupError")
    RESULTS.append(("get_norad_guid raises on missing", False, "No exception raised"))
except spacebook.SpacebookLookupError:
    print(f"\n  {PASS} get_norad_guid(unknown) correctly raises SpacebookLookupError")
    RESULTS.append(("get_norad_guid raises on missing", True, ""))
except Exception as exc:
    print(f"\n  {FAIL} get_norad_guid(unknown) raised wrong exception: {exc}")
    RESULTS.append(("get_norad_guid raises on missing", False, str(exc)))

# 5. XP-TLE catalog
print("\n-- [5] XP-TLE Catalog --")
xp_tles = check("fetch_xp_tle_catalog()", spacebook.fetch_xp_tle_catalog)
if xp_tles:
    print(f"         first TLE name : {xp_tles[0].name}")
    print(f"         first NORAD    : {xp_tles[0].norad_id}")
    # Find ISS in XP-TLE catalog
    iss_tle = next((t for t in xp_tles if t.norad_id == str(ISS_NORAD)), None)
    if iss_tle:
        print(f"         ISS found      : {iss_tle.name}  epoch_jd={iss_tle.epoch_jd:.4f}")
        print(f"         [OK] ISS present in XP-TLE catalog.")
    else:
        print(f"         [WARN] ISS (NORAD {ISS_NORAD}) not found in XP-TLE catalog.")

# 6. Standard TLE catalog
print("\n-- [6] Standard TLE Catalog --")
std_tles = check("fetch_tle_catalog()", spacebook.fetch_tle_catalog)
if std_tles:
    print(f"         first TLE name : {std_tles[0].name}")

# 7. Historical TLE
print("\n-- [7] Historical TLE --")
yesterday = datetime.now(timezone.utc) - timedelta(days=1)
hist_tles = check(f"fetch_historical_tle({yesterday.date()})", spacebook.fetch_historical_tle, yesterday)
if hist_tles:
    print(f"         count : {len(hist_tles)} TLEs for {yesterday.date()}")

# 8. Synthetic Covariance (STK format)
print("\n-- [8] Synthetic Covariance (ISS) --")
stk_text = check(f"fetch_synthetic_covariance_stk(ISS={ISS_NORAD})", spacebook.fetch_synthetic_covariance_stk, ISS_NORAD)
if stk_text:
    assert "stk.v" in stk_text, "Missing STK header"
    assert "EphemerisTimePosVel" in stk_text, "Missing ephemeris data block"
    assert "TEMEOfDate" in stk_text, "Missing TEME coordinate system tag"
    lines = [l.strip() for l in stk_text.splitlines() if l.strip() and not l.startswith("#")]
    data_lines = [l for l in lines if l and l[0].isdigit() or l[0] == "-"]
    print(f"         STK header   : present")
    print(f"         Coord system : TEMEOfDate (native ASTRA frame)")
    print(f"         Data lines   : ~{len(data_lines)}")
    print(f"         [OK] STK ephemeris structure validated.")

# 9. Satcat Details
print("\n-- [9] Satcat Physical Details (ISS) --")
details = check(f"fetch_satcat_details(ISS={ISS_NORAD})", spacebook.fetch_satcat_details, ISS_NORAD)
if details and "discosMsg" in details:
    msg = details["discosMsg"].get("comspocDerivedMessage", {})
    print(f"         name             : {details.get('name')}")
    print(f"         adoptedMass      : {msg.get('adoptedMass')} kg")
    print(f"         crossSection     : {msg.get('crossSectionMedian'):.2f} m2")
    print(f"         ballisticCoeff   : {msg.get('ballisticCoefficient')}")
    print(f"         solarRadPressure : {msg.get('solarRadiationPressure')}")
    print(f"         [OK] Physical parameters available for DragConfig.")

# ── Summary ──────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  SUMMARY")
print("=" * 65)
passed = sum(1 for _, ok, _ in RESULTS if ok)
failed = len(RESULTS) - passed
print(f"  Passed: {passed}/{len(RESULTS)}   Failed: {failed}/{len(RESULTS)}")
for label, ok, err in RESULTS:
    tag = "PASS" if ok else "FAIL"
    suf = f"  => {err}" if not ok else ""
    print(f"  [{tag}] {label}{suf}")
print("=" * 65)
sys.exit(0 if failed == 0 else 1)
