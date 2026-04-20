"""
Example 07: Spacebook / COMSPOC Data Pipeline
==============================================
Demonstrates ASTRA-Core's integration with the COMSPOC Spacebook platform
(https://spacebook.com) — a high-fidelity, unauthenticated SSA data service.

Spacebook provides several data products that are superior to the standard
CelesTrak / Space-Track catalog for precision work:

  - **XP-TLEs** — Extended-Precision TLEs, numerically refined by COMSPOC.
  - **Synthetic Covariance** — Per-satellite 6×6 observational covariance
    matrices from COMSPOC's SynCoPate system, enabling flight-grade P_c.
  - **Live Space Weather** — Real-time F10.7 and Ap indices (6-hour cache).
  - **Earth Orientation Parameters** — Live IERS polar motion and UT1-UTC
    for rigorous TEME↔ECEF frame transforms.

No account or credentials are required. All endpoints are freely accessible.

Environment variables:
  ASTRA_SPACEBOOK_ENABLED  Set to "false" to disable Spacebook (default: true)
  ASTRA_DATA_DIR           Cache directory (default: ~/.astra/data/)
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import astra
from astra.errors import SpacebookError, SpacebookLookupError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print('=' * 60)


def _ok(msg: str) -> None:
    print(f"  [OK]  {msg}")


def _warn(msg: str) -> None:
    print(f"  [!!]  {msg}")


def _info(msg: str) -> None:
    print(f"        {msg}")


# ---------------------------------------------------------------------------
# Step 1 — Connectivity probe
# ---------------------------------------------------------------------------

def step1_probe() -> bool:
    _section("1 · Spacebook connectivity probe")
    from astra import spacebook as sb

    if not sb.SPACEBOOK_ENABLED:
        _warn("Spacebook is disabled (ASTRA_SPACEBOOK_ENABLED=false).")
        _info("Set ASTRA_SPACEBOOK_ENABLED=true (or unset) to enable.")
        return False

    _info("Probing https://spacebook.com/api …")
    reachable = sb.is_available(timeout=5)
    if reachable:
        _ok("Spacebook API is reachable.")
    else:
        _warn("Spacebook is not reachable from this machine.")
        _info("Demo will fall back gracefully on network errors.")
    return reachable


# ---------------------------------------------------------------------------
# Step 2 — XP-TLE catalog
# ---------------------------------------------------------------------------

def step2_xp_tle() -> list:
    _section("2 · Fetch XP-TLE (Extended-Precision TLE) catalog")
    _info("Fetching … (cached for 6 h after first download)")
    try:
        xp_catalog = astra.fetch_xp_tle_catalog()
        _ok(f"Loaded {len(xp_catalog)} XP-TLE objects from Spacebook.")

        # Show first few objects
        print()
        print(f"  {'Name':<26} {'NORAD':>7}  {'Epoch JD':>14}  {'BSTAR':>12}")
        print("  " + "-" * 66)
        for sat in xp_catalog[:6]:
            print(
                f"  {sat.name:<26} {sat.norad_id:>7}  "
                f"{sat.epoch_jd:>14.6f}  {sat.bstar:>12.4e}"
            )

        return xp_catalog
    except SpacebookError as exc:
        _warn(f"XP-TLE fetch failed: {exc}")
        _info("Falling back to standard CelesTrak TLE catalog …")
        catalog = astra.fetch_celestrak_active()
        _ok(f"CelesTrak fallback: {len(catalog)} objects.")
        return catalog


# ---------------------------------------------------------------------------
# Step 3 — Live Space Weather
# ---------------------------------------------------------------------------

def step3_space_weather() -> None:
    _section("3 · Live Space Weather (F10.7 / Ap)")
    _info("Requesting today's F10.7 and Ap from Spacebook …")

    now_jd = 2451545.0 + (
        datetime.now(timezone.utc) - datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    ).total_seconds() / 86400.0

    try:
        f107_obs, f107_adj, ap = astra.get_space_weather_sb(now_jd)
        _ok("Spacebook live SW retrieved successfully.")
        _info(f"F10.7 observed : {f107_obs:.1f} SFU")
        _info(f"F10.7 adjusted : {f107_adj:.1f} SFU (81-day centred)")
        _info(f"Ap daily       : {ap:.1f}")

        # Show how atmospheric density differs at today's vs moderate activity
        rho_today   = astra.atmospheric_density_empirical(400.0, f107_obs, f107_adj, ap)
        rho_moderate = astra.atmospheric_density_empirical(400.0, 150.0, 150.0, 15.0)
        ratio = rho_today / rho_moderate if rho_moderate > 0 else float("nan")
        print()
        _info(f"NRLMSISE-00 density at 400 km:")
        _info(f"  Today    : {rho_today:.3e} kg/m³")
        _info(f"  Moderate : {rho_moderate:.3e} kg/m³  (F10.7=150, Ap=15)")
        _info(f"  Ratio    : {ratio:.2f}×  (affects drag magnitude proportionally)")
    except SpacebookError as exc:
        _warn(f"Spacebook SW unavailable: {exc}")
        _info("get_space_weather() will fall back to CelesTrak CSV automatically.")


# ---------------------------------------------------------------------------
# Step 4 — Earth Orientation Parameters
# ---------------------------------------------------------------------------

def step4_eop() -> None:
    _section("4 · Earth Orientation Parameters (EOP)")
    _info("Fetching today's polar motion and UT1-UTC from Spacebook …")

    now_jd = 2451545.0 + (
        datetime.now(timezone.utc) - datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    ).total_seconds() / 86400.0

    try:
        xp, yp, dut1 = astra.get_eop_sb(now_jd)
        _ok("Spacebook EOP retrieved successfully.")
        _info(f"Polar motion xp : {xp:+.6f} arcsec")
        _info(f"Polar motion yp : {yp:+.6f} arcsec")
        _info(f"UT1 − UTC       : {dut1:+.6f} s")
        _info("These values are injected into teme_to_ecef() for rigorous")
        _info("TEME → ECEF frame transformations in filter_region() and")
        _info("passes_over_location().")
    except SpacebookError as exc:
        _warn(f"Spacebook EOP unavailable: {exc}")
        _info("ASTRA will fall back to Skyfield IERS finals2000A data.")


# ---------------------------------------------------------------------------
# Step 5 — Synthetic Covariance for ISS (NORAD 25544)
# ---------------------------------------------------------------------------

def step5_synthetic_covariance(norad_id: str = "25544") -> np.ndarray | None:
    _section(f"5 · Synthetic Covariance for NORAD {norad_id} (ISS)")
    _info(f"Resolving COMSPOC GUID for NORAD {norad_id} …")
    try:
        stk_text = astra.fetch_synthetic_covariance_stk(norad_id)
        cov = astra.parse_stk_ephemeris(stk_text)

        if cov is None:
            _warn("No CovarianceTimePosVel block found in STK ephemeris.")
            return None

        _ok("6×6 synthetic covariance matrix retrieved from Spacebook SynCoPate.")
        print()
        print("  Covariance matrix P (km², km²/s, km²/s²) — lower-left displayed:")
        for i, row in enumerate(cov):
            vals = "  ".join(f"{v:+.3e}" for v in row[: i + 1])
            print(f"    [{i}]  {vals}")

        # Interpret 1-sigma position uncertainty
        sig_r = float(np.sqrt(cov[0, 0]))
        sig_t = float(np.sqrt(cov[1, 1]))
        sig_n = float(np.sqrt(cov[2, 2]))
        print()
        _info(f"1-σ position uncertainty (radial)    : {sig_r * 1000:.1f} m")
        _info(f"1-σ position uncertainty (transverse): {sig_t * 1000:.1f} m")
        _info(f"1-σ position uncertainty (normal)    : {sig_n * 1000:.1f} m")
        _info("This flight-grade covariance replaces estimate_covariance() for")
        _info("accurate P_c calculation via compute_collision_probability().")
        return cov

    except SpacebookLookupError as exc:
        _warn(f"NORAD {norad_id} not found in Spacebook SATCAT: {exc}")
        _info("Try astra.fetch_xp_tle_catalog() first to update the SATCAT cache.")
    except SpacebookError as exc:
        _warn(f"Synthetic covariance unavailable: {exc}")
    return None


# ---------------------------------------------------------------------------
# Step 6 — Propagate XP-TLEs and run a conjunction screen
# ---------------------------------------------------------------------------

def step6_xp_tle_conjunction(catalog: list) -> None:
    _section("6 · Conjunction screening with XP-TLE catalog subset")
    if len(catalog) < 2:
        _warn("Need at least 2 objects for conjunction analysis. Skipping.")
        return

    # Use a small subset for a fast demonstration
    subset = catalog[:200] if len(catalog) >= 200 else catalog
    objects = [astra.make_debris_object(sat) for sat in subset]
    leo = astra.filter_altitude(objects, min_km=200, max_km=2000)

    if len(leo) < 2:
        _warn(f"Only {len(leo)} LEO objects in subset — need ≥ 2. Skipping.")
        return

    sources = [obj.source for obj in leo]
    start_jd = sources[0].epoch_jd
    times_jd = start_jd + np.arange(0, 60, 5.0) / 1440.0   # 1 hour, 5-min steps

    _info(f"Propagating {len(leo)} LEO objects (SGP4) over 1-hour window …")
    trajectories = astra.propagate_many(sources, times_jd)

    _info("Running KD-tree conjunction screen (threshold = 5 km) …")
    elements_map = {obj.source.norad_id: obj for obj in leo}
    events = astra.find_conjunctions(
        trajectories,
        times_jd=times_jd,
        elements_map=elements_map,
        threshold_km=5.0,
        coarse_threshold_km=50.0,
    )

    _ok(f"Found {len(events)} conjunction event(s) in {len(leo)}-object LEO subset.")
    if events:
        print()
        print(f"  {'#':>3}  {'Object A':>10}  {'Object B':>10}  "
              f"{'Miss dist (km)':>15}  {'P_c':>10}  {'Risk':>8}")
        print("  " + "-" * 66)
        for i, ev in enumerate(events[:5]):
            pc_str = f"{ev.collision_probability:.2e}" if ev.collision_probability else "N/A"
            risk   = ev.risk_level or "UNKNOWN"
            print(
                f"  {i+1:>3}  {ev.object_a_id:>10}  {ev.object_b_id:>10}  "
                f"{ev.miss_distance_km:>15.3f}  {pc_str:>10}  {risk:>8}"
            )
        if len(events) > 5:
            print(f"  … and {len(events) - 5} more events.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("  ASTRA-Core  ·  Example 07: Spacebook / COMSPOC Pipeline")
    print("=" * 60)
    print()
    print("  Spacebook is unauthenticated — no account required.")
    print("  Data is cached locally (see ASTRA_DATA_DIR).")

    step1_probe()
    catalog = step2_xp_tle()
    step3_space_weather()
    step4_eop()
    step5_synthetic_covariance("25544")   # ISS
    step6_xp_tle_conjunction(catalog)

    print()
    print("✓ Spacebook pipeline demo complete.")
    print()
    print("Next steps:")
    print("  • Pass the covariance matrix from step5 into")
    print("    astra.compute_collision_probability() for flight-grade P_c.")
    print("  • Enable strict mode (astra.set_strict_mode(True)) so any")
    print("    Spacebook failure raises instead of silently falling back.")
    print("  • Set ASTRA_SPACEBOOK_ENABLED=false to force CelesTrak-only mode.")


if __name__ == "__main__":
    main()
