# astra/tle.py
"""ASTRA Core TLE parsing and validation.

This module is the entry point for all data entering ASTRA Core. It handles
parsing, validation, and batch loading of Two-Line Element (TLE) sets from
raw text.
"""

from __future__ import annotations

import logging
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from astra.models import SatelliteState
from datetime import datetime, timedelta, timezone

from astra.errors import AstraError, InvalidTLEError
from astra.models import SatelliteTLE

logger = logging.getLogger(__name__)

# J2000 reference epoch for JD conversion (matching astra.time)
_J2000_JD = 2451545.0
_J2000_EPOCH = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def _compute_checksum(line: str) -> int:
    """Compute the TLE checksum for a given line (excluding the last char).

    The checksum is the sum of all digits, with minus signs counting as 1.
    All other characters are ignored. The result is modulo 10.
    """
    total = 0
    for char in line[:-1]:
        if char.isdigit():
            total += int(char)
        elif char == "-":
            total += 1
    return total % 10


def _parse_epoch_to_jd(epoch_str: str) -> float:
    """Convert TLE epoch string (YYDDD.FFFFFFFF) to Julian Date.

    Uses integer-split arithmetic to avoid floating-point rounding errors
    at the day/fraction boundary. The fractional day is accumulated
    via timedelta microseconds rather than floating-point multiplication.

    Years >= 57 are interpreted as 19YY, years < 57 as 20YY (standard TLE
    two-digit year convention).

    From calendar year 2057 onward, YY ≥ 57 maps to 19YY and collides with
    early-spacecraft epochs; for archival or long-horizon data prefer CCSDS OMM
    (full UTC epoch) over TLEs.
    """
    try:
        y = int(epoch_str[:2])
        day_of_year = float(epoch_str[2:])
    except ValueError as e:
        raise ValueError(f"Invalid epoch format: {epoch_str}") from e

    year = 2000 + y if y < 57 else 1900 + y

    # Integer day + fractional-day both accumulated through timedelta
    day_whole = int(day_of_year)  # e.g. 123
    day_frac = day_of_year - day_whole  # e.g. 0.45678
    # Convert fractional day to integer microseconds for precise accumulation
    frac_us = round(day_frac * 86400 * 1_000_000)
    dt = datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(
        days=day_whole - 1, microseconds=frac_us
    )
    delta = dt - _J2000_EPOCH
    return _J2000_JD + delta.total_seconds() / 86400.0


def check_tle_staleness(
    satellite: SatelliteState, target_jd: float | np.ndarray
) -> None:
    """Verify that the propagation time is within 30 days of the satellite epoch.

    SGP4 accuracy degrades exponentially over time. For mission-critical analysis,
    using TLEs older than 30 days is discouraged and blocked in STRICT mode.

    Args:
        satellite: The SatelliteTLE or SatelliteOMM object being propagated.
        target_jd: The target Julian Date(s) for propagation.

    Raises:
        PropagationError: If delta > 30 days and ASTRA_STRICT_MODE is True.
    """
    import numpy as np
    from astra import config
    from astra.errors import PropagationError

    delta_days = np.abs(np.asanyarray(target_jd) - satellite.epoch_jd)
    max_delta = np.max(delta_days)

    if max_delta > 30.0:
        msg = (
            f"TLE/OMM for {satellite.name} (NORAD {satellite.norad_id}) is stale "
            f"({max_delta:.2f} days from epoch). Max recommended SGP4 horizon is 30 days."
        )
        if config.ASTRA_STRICT_MODE:
            raise PropagationError(
                msg, norad_id=satellite.norad_id, t_jd=float(np.max(target_jd))
            )
        logger.warning(msg)


def parse_tle(name: str, line1: str, line2: str) -> SatelliteTLE:
    """Parse three raw TLE lines into a validated SatelliteTLE object.

    Args:
        name: Object name string.
        line1: TLE line 1.
        line2: TLE line 2.

    Returns:
        A fully populated SatelliteTLE instance.

    Raises:
        InvalidTLEError: If any validation fails.
    """
    # 1. Strip whitespace
    name = name.strip()
    line1 = line1.strip()
    line2 = line2.strip()

    # 2. Length checks (each line must be exactly 69 characters)
    if len(line1) != 69:
        raise InvalidTLEError(
            f"Line 1 length is {len(line1)}, expected 69",
            norad_id="UNKNOWN",
            object_name=name,
            invalid_line=line1,
            reason="L1_LENGTH",
        )
    if len(line2) != 69:
        raise InvalidTLEError(
            f"Line 2 length is {len(line2)}, expected 69",
            norad_id="UNKNOWN",
            object_name=name,
            invalid_line=line2,
            reason="L2_LENGTH",
        )

    # 3. Prefix checks (line 1 must start with "1 ", line 2 with "2 ")
    if not line1.startswith("1 "):
        raise InvalidTLEError(
            "Line 1 does not start with '1 '",
            norad_id="UNKNOWN",
            object_name=name,
            invalid_line=line1,
            reason="L1_PREFIX",
        )
    if not line2.startswith("2 "):
        raise InvalidTLEError(
            "Line 2 does not start with '2 '",
            norad_id="UNKNOWN",
            object_name=name,
            invalid_line=line2,
            reason="L2_PREFIX",
        )

    # 4. Checksum validation
    try:
        expected_cs1 = int(line1[68])
        if _compute_checksum(line1) != expected_cs1:
            raise InvalidTLEError(
                "Line 1 checksum mismatch",
                norad_id="UNKNOWN",
                object_name=name,
                invalid_line=line1,
                reason="L1_CHECKSUM",
            )
    except ValueError:
        raise InvalidTLEError(
            "Line 1 checksum character is not a digit",
            norad_id="UNKNOWN",
            object_name=name,
            invalid_line=line1,
            reason="L1_CHECKSUM",
        )

    try:
        expected_cs2 = int(line2[68])
        if _compute_checksum(line2) != expected_cs2:
            raise InvalidTLEError(
                "Line 2 checksum mismatch",
                norad_id="UNKNOWN",
                object_name=name,
                invalid_line=line2,
                reason="L2_CHECKSUM",
            )
    except ValueError:
        raise InvalidTLEError(
            "Line 2 checksum character is not a digit",
            norad_id="UNKNOWN",
            object_name=name,
            invalid_line=line2,
            reason="L2_CHECKSUM",
        )

    # 5. NORAD ID consistency
    norad_id = line1[2:7].strip()
    norad_id2 = line2[2:7].strip()
    if norad_id != norad_id2:
        raise InvalidTLEError(
            "NORAD ID mismatch between Line 1 and Line 2",
            norad_id=norad_id,
            object_name=name,
            invalid_line=line2,
            reason="ID_MISMATCH",
        )

    # 6. Extract epoch and convert to Julian Date
    epoch_str = line1[18:32].strip()
    try:
        epoch_jd = _parse_epoch_to_jd(epoch_str)
    except ValueError as e:
        raise InvalidTLEError(
            f"Failed to parse epoch: {e}",
            norad_id=norad_id,
            object_name=name,
            invalid_line=line1,
            reason="EPOCH_PARSE_ERROR",
        )

    # 7. Physical Bounds Checking (Input Poisoning Defense)
    try:
        ecco_str = line2[26:33].strip()
        if ecco_str:
            ecco = float("." + ecco_str)
            if ecco < 0.0 or ecco >= 1.0:
                raise ValueError(f"Eccentricity {ecco} out of bounds [0, 1) for SGP4")
        
        incl_str = line2[8:16].strip()
        if incl_str:
            incl = float(incl_str)
            if incl < 0.0 or incl > 180.0:
                raise ValueError(f"Inclination {incl} out of bounds [0, 180]")
    except ValueError as e:
        raise InvalidTLEError(
            f"Physical bounds violation in TLE: {e}",
            norad_id=norad_id,
            object_name=name,
            invalid_line=line2,
            reason="BOUNDS_VIOLATION",
        )

    # 8. Extract classification character (U=Unclassified, C=Classified, S=Secret)
    # NOTE: This is a SECURITY classification, not an object type.
    # Object type (PAYLOAD/DEBRIS/ROCKET_BODY) requires SATCAT lookup.
    classification_flag = line1[7]

    # 8. Default object_type to UNKNOWN — to be overridden by SATCAT enrichment
    object_type = "UNKNOWN"

    # 9. Instantiate and return
    return SatelliteTLE(
        norad_id=norad_id,
        name=name,
        line1=line1,
        line2=line2,
        epoch_jd=epoch_jd,
        object_type=object_type,
        classification_flag=classification_flag,
    )


def validate_tle(name: str, line1: str, line2: str) -> bool:
    """Non-destructive validation of TLE strings.

    Args:
        name: Object name string.
        line1: TLE line 1.
        line2: TLE line 2.

    Returns:
        True if TLE is well-formed and checksums pass, False otherwise.
    """
    try:
        parse_tle(name, line1, line2)
        return True
    except InvalidTLEError:
        return False


def _chunk_tle_lines(tle_lines: list[str]) -> list[tuple[str, str, str]]:
    """Group a list of TLE lines into triplets (name, line1, line2).

    Supports both 3-line format (with name) and 2-line format (auto-generates 'Unknown').
    Silently skips empty lines and invalid headers.
    """
    lines = [L.strip() for L in tle_lines if L.strip()]

    triplets = []
    i = 0
    n = len(lines)
    while i < n:
        if not lines[i].startswith("1 "):
            if (
                i + 1 < n
                and lines[i + 1].startswith("1 ")
                and i + 2 < n
                and lines[i + 2].startswith("2 ")
            ):
                triplets.append((lines[i], lines[i + 1], lines[i + 2]))
                i += 3
            else:
                i += 1
            continue

        if i + 1 < n and lines[i + 1].startswith("2 "):
            norad_id = lines[i][2:7].strip()
            synthetic_name = f"NORAD-{norad_id}" if norad_id else "Unknown"
            triplets.append((synthetic_name, lines[i], lines[i + 1]))
            i += 2
        else:
            i += 1
    return triplets


def load_tle_catalog(tle_lines: list[str]) -> list[SatelliteTLE]:
    """Parse a batch of TLE text lines into SatelliteTLE objects.

    Invalid TLEs are skipped with a logged warning.

    Args:
        tle_lines: A flat list of strings, typically expected to be in
            triplets: name, line1, line2.

    Returns:
        List of successfully parsed SatelliteTLE objects.

    Raises:
        AstraError: If total parse failure occurs (result is empty but input
            was non-empty).
    """
    if not tle_lines:
        return []

    triplets = _chunk_tle_lines(tle_lines)

    if not triplets and any(L.strip() for L in tle_lines):
        raise AstraError("Failed to parse any TLE triplets from input lines.")

    from astra.config import ASTRA_STRICT_MODE

    results: list[SatelliteTLE] = []

    for name, line1, line2 in triplets:
        try:
            sat = parse_tle(name, line1, line2)
            results.append(sat)
        except InvalidTLEError as e:
            if ASTRA_STRICT_MODE:
                # In strict mode, a single invalid TLE fails the entire load (SE-I).
                raise
            norad_id = e.norad_id or "UNKNOWN"
            logger.warning(
                f"Skipping invalid TLE for {name} (NORAD {norad_id}): {e.message}"
            )

    if not results and any(L.strip() for L in tle_lines):
        raise InvalidTLEError(
            "Total parse failure: no valid TLEs found in non-empty catalog input. "
            "Verify that input lines follow 3-line (name + L1 + L2) or 2-line (L1 + L2) format.",
            reason="TOTAL_PARSE_FAILURE",
        )

    return results
