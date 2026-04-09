# astra/ocm.py
"""ASTRA Core — STK Ephemeris & Orbit State Parser.

Parses the AGI/STK ``.e`` (DotE) ephemeris format returned by the COMSPOC
Spacebook ``/api/entity/synthetic-covariance/{guid}`` endpoint, converting
numerically propagated state vectors into ASTRA ``NumericalState`` objects
for direct use in propagation validation, benchmarking, and conjunction
analysis.

Background
----------
The Spacebook Synthetic Covariance endpoint returns a **STK v12.0 DotE**
ephemeris file — not JSON. This was verified empirically on 2026-04-09.
The file contains:

- Scenario epoch (in STK calendar string format)
- Coordinate system: ``TEMEOfDate`` (the same frame ASTRA's RK87 uses internally)
- Distance unit: Kilometers
- An ``EphemerisTimePosVel`` data block with per-epoch 7-column rows:
  ``t_offset_sec  x_km  y_km  z_km  vx_km_s  vy_km_s  vz_km_s``

Since the coordinate frame (TEME) and units (km, km/s) match ASTRA's internal
conventions exactly, no frame rotation or unit conversion is needed.

Note on Reference Ephemerides
------------------------------
The Spacebook ``/api/entity/reference-ephemerides/{guid}`` and
``/api/entity/reference-ephemerides/ocm/{guid}`` endpoints returned HTTP 500
during our 2026-04-09 live testing. The Synthetic Covariance endpoint is
functionally equivalent (same COMSPOC numerical propagation, same state vectors)
and was used as the file source instead.

Usage Example
-------------
::

    from astra.spacebook import fetch_synthetic_covariance_stk
    from astra.ocm import parse_stk_ephemeris

    stk_text = fetch_synthetic_covariance_stk(norad_id=25544)   # ISS
    states = parse_stk_ephemeris(stk_text)

    print(f"{len(states)} state vectors")
    print(f"First epoch JD: {states[0].t_jd:.4f}")
    print(f"Position (km):  {states[0].position_km}")
    print(f"Velocity (km/s):{states[0].velocity_km_s}")
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from astra.errors import AstraError
from astra.log import get_logger
from astra.propagator import NumericalState

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# STK Month Name → Integer Map
# ---------------------------------------------------------------------------
_STK_MONTHS = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
    "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
}

# STK scenario epoch format: "8 Apr 2026 00:44:30.451"
_STK_EPOCH_RE = re.compile(
    r"(\d+)\s+([A-Za-z]{3})\s+(\d{4})\s+(\d{2}):(\d{2}):(\d{1,2}(?:\.\d+)?)"
)


def _parse_stk_epoch(epoch_str: str) -> float:
    """Parse an STK scenario epoch string into a Julian Date (UTC).

    Supported formats:
    - ``"8 Apr 2026 00:44:30.451"``
    - ``"10 Jan 2025 12:00:00"``

    Args:
        epoch_str: Raw epoch string from the STK file ``ScenarioEpoch`` line.

    Returns:
        Julian Date (UTC) as a float.

    Raises:
        AstraError: If the epoch string cannot be parsed.
    """
    m = _STK_EPOCH_RE.search(epoch_str)
    if not m:
        raise AstraError(
            f"Cannot parse STK scenario epoch: {epoch_str!r}. "
            "Expected format: 'D Mon YYYY HH:MM:SS.fff'"
        )
    day   = int(m.group(1))
    month = _STK_MONTHS.get(m.group(2).capitalize(), 0)
    year  = int(m.group(3))
    hour  = int(m.group(4))
    minute = int(m.group(5))
    second = float(m.group(6))

    if month == 0:
        raise AstraError(f"Unknown month abbreviation in STK epoch: {m.group(2)!r}")

    sec_int = int(second)
    microsecond = int(round((second - sec_int) * 1_000_000))

    dt = datetime(year, month, day, hour, minute, sec_int, microsecond, tzinfo=timezone.utc)

    # Convert datetime → Julian Date
    # JD = JDN + (hour - 12) / 24 + min / 1440 + sec / 86400
    # Using the standard algorithm:
    a = (14 - month) // 12
    y = year + 4800 - a
    m_n = month + 12 * a - 3
    jdn = (day + (153 * m_n + 2) // 5 + 365 * y + y // 4
           - y // 100 + y // 400 - 32045)
    # Fractional day
    day_frac = (hour - 12) / 24.0 + minute / 1440.0 + (sec_int + microsecond / 1_000_000) / 86400.0
    return float(jdn) + day_frac


def parse_stk_ephemeris(text: str) -> list[NumericalState]:
    """Parse an AGI/STK DotE ephemeris file into a list of NumericalState objects.

    The STK DotE format is returned by the Spacebook synthetic covariance
    endpoint. Each row in the ``EphemerisTimePosVel`` data block provides:
    ``t_sec  x_km  y_km  z_km  vx_km_s  vy_km_s  vz_km_s``

    where ``t_sec`` is seconds elapsed since the ``ScenarioEpoch``.

    The coordinate system is always ``TEMEOfDate`` (verified from live data),
    which is the same frame used by ASTRA's RK87 numerical propagator — no
    frame rotation is required.

    Args:
        text: Raw STK ephemeris file content as a string.

    Returns:
        List of ``NumericalState`` objects sorted by ascending Julian Date,
        with position in km and velocity in km/s in the TEME frame.

    Raises:
        AstraError: If the file is malformed (missing epoch or data block).
    """
    if not text or not text.strip():
        raise AstraError("STK ephemeris text is empty.")

    # ── 1. Validate file signature ─────────────────────────────────────────
    first_line = text.lstrip().split("\n")[0].strip()
    if not first_line.lower().startswith("stk.v"):
        raise AstraError(
            f"Not a valid STK ephemeris file. Expected 'stk.v...' header, "
            f"got: {first_line!r}"
        )

    # ── 2. Extract ScenarioEpoch ────────────────────────────────────────────
    epoch_jd: Optional[float] = None
    coord_system: str = "Unknown"
    distance_unit: str = "Unknown"
    n_points_expected: Optional[int] = None
    in_ephemeris_block = False
    in_data_block = False

    data_rows: list[tuple[float, float, float, float, float, float, float]] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()

        # Skip comment lines (but keep them for coordinate system detection)
        if not line:
            continue
        if line.startswith("#"):
            continue

        tokens_lower = line.lower()

        # Block markers
        if tokens_lower.startswith("begin ephemeris"):
            in_ephemeris_block = True
            continue
        if tokens_lower.startswith("end ephemeris"):
            in_ephemeris_block = False
            in_data_block = False
            continue

        if not in_ephemeris_block:
            continue

        # Header fields
        if tokens_lower.startswith("scenarioepoch"):
            # Format: "    ScenarioEpoch    8 Apr 2026 00:44:30.451"
            parts = line.split(None, 1)
            if len(parts) >= 2:
                epoch_jd = _parse_stk_epoch(parts[1])
            continue

        if tokens_lower.startswith("coordinatesystem"):
            parts = line.split(None, 1)
            if len(parts) >= 2:
                coord_system = parts[1].strip()
            continue

        if tokens_lower.startswith("distanceunit"):
            parts = line.split(None, 1)
            if len(parts) >= 2:
                distance_unit = parts[1].strip()
            continue

        if tokens_lower.startswith("numberofephemerispoints"):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    n_points_expected = int(parts[1])
                except ValueError:
                    pass
            continue

        # Data block marker
        if tokens_lower.startswith("ephemeristimeposvel"):
            in_data_block = True
            continue

        if tokens_lower.startswith("covariancetimeposvel"):
            in_data_block = False
            continue

        if not in_data_block:
            continue

        # Parse data row: t  x  y  z  vx  vy  vz
        fields = line.split()
        if len(fields) < 7:
            continue
        try:
            t_sec = float(fields[0])
            x     = float(fields[1])
            y     = float(fields[2])
            z     = float(fields[3])
            vx    = float(fields[4])
            vy    = float(fields[5])
            vz    = float(fields[6])
            data_rows.append((t_sec, x, y, z, vx, vy, vz))
        except ValueError:
            continue   # Non-numeric row (e.g. covariance section header)

    # ── 3. Validate parsed content ──────────────────────────────────────────
    if epoch_jd is None:
        raise AstraError(
            "STK ephemeris is missing 'ScenarioEpoch' field. "
            "Ensure the file is a valid STK DotE format."
        )
    if not data_rows:
        raise AstraError(
            "STK ephemeris 'EphemerisTimePosVel' block is empty or missing. "
            "No state vectors could be extracted."
        )

    # Unit guard: we only handle Kilometers (verified from Spacebook)
    if distance_unit.lower() not in ("kilometers", "km", ""):
        logger.warning(
            "STK ephemeris distance unit is '%s'; expected 'Kilometers'. "
            "Results may be in wrong units — verify with parse output.",
            distance_unit,
        )

    logger.debug(
        "STK ephemeris: epoch_jd=%.4f | coord=%s | unit=%s | points=%d (expected %s)",
        epoch_jd, coord_system, distance_unit, len(data_rows),
        str(n_points_expected) if n_points_expected else "?",
    )

    # ── 4. Build NumericalState list ────────────────────────────────────────
    states: list[NumericalState] = []
    sec_per_day = 86400.0

    for t_sec, x, y, z, vx, vy, vz in data_rows:
        t_jd = epoch_jd + t_sec / sec_per_day
        states.append(
            NumericalState(
                t_jd=t_jd,
                position_km=np.array([x, y, z], dtype=np.float64),
                velocity_km_s=np.array([vx, vy, vz], dtype=np.float64),
            )
        )

    # Sort by time (should already be in order, but be safe)
    states.sort(key=lambda s: s.t_jd)

    logger.info(
        "parse_stk_ephemeris: %d NumericalState objects extracted. "
        "Time span: %.2f hours.",
        len(states),
        (states[-1].t_jd - states[0].t_jd) * 24.0 if len(states) > 1 else 0.0,
    )
    return states
