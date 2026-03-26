# astra/tle.py
"""ASTRA Core TLE parsing and validation.

This module is the entry point for all data entering ASTRA Core. It handles
parsing, validation, and batch loading of Two-Line Element (TLE) sets from
raw text.
"""
from __future__ import annotations

import logging
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

    Years >= 57 are 1900s, years < 57 are 2000s.
    """
    try:
        y = int(epoch_str[:2])
        day_of_year = float(epoch_str[2:])
    except ValueError as e:
        raise ValueError(f"Invalid epoch format: {epoch_str}") from e

    year = 2000 + y if y < 57 else 1900 + y

    dt = datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=day_of_year - 1.0)
    delta = dt - _J2000_EPOCH
    return _J2000_JD + delta.total_seconds() / 86400.0


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
            norad_id="UNKNOWN", object_name=name, invalid_line=line1, reason="L1_LENGTH",
        )
    if len(line2) != 69:
        raise InvalidTLEError(
            f"Line 2 length is {len(line2)}, expected 69",
            norad_id="UNKNOWN", object_name=name, invalid_line=line2, reason="L2_LENGTH",
        )

    # 3. Prefix checks (line 1 must start with "1 ", line 2 with "2 ")
    if not line1.startswith("1 "):
        raise InvalidTLEError(
            "Line 1 does not start with '1 '",
            norad_id="UNKNOWN", object_name=name, invalid_line=line1, reason="L1_PREFIX",
        )
    if not line2.startswith("2 "):
        raise InvalidTLEError(
            "Line 2 does not start with '2 '",
            norad_id="UNKNOWN", object_name=name, invalid_line=line2, reason="L2_PREFIX",
        )

    # 4. Checksum validation
    try:
        expected_cs1 = int(line1[68])
        if _compute_checksum(line1) != expected_cs1:
            raise InvalidTLEError(
                "Line 1 checksum mismatch",
                norad_id="UNKNOWN", object_name=name, invalid_line=line1, reason="L1_CHECKSUM",
            )
    except ValueError:
        raise InvalidTLEError(
            "Line 1 checksum character is not a digit",
            norad_id="UNKNOWN", object_name=name, invalid_line=line1, reason="L1_CHECKSUM",
        )

    try:
        expected_cs2 = int(line2[68])
        if _compute_checksum(line2) != expected_cs2:
            raise InvalidTLEError(
                "Line 2 checksum mismatch",
                norad_id="UNKNOWN", object_name=name, invalid_line=line2, reason="L2_CHECKSUM",
            )
    except ValueError:
        raise InvalidTLEError(
            "Line 2 checksum character is not a digit",
            norad_id="UNKNOWN", object_name=name, invalid_line=line2, reason="L2_CHECKSUM",
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

    # 7. Extract object_type from classification character
    classification = line1[7]
    if classification == "U":
        object_type = "UNKNOWN"
    elif classification == "C":
        object_type = "PAYLOAD"
    elif classification == "D":
        object_type = "DEBRIS"
    elif classification == "R":
        object_type = "ROCKET_BODY"
    else:
        object_type = "UNKNOWN"

    # 8. Instantiate and return
    return SatelliteTLE(
        norad_id=norad_id,
        name=name,
        line1=line1,
        line2=line2,
        epoch_jd=epoch_jd,
        object_type=object_type,
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
    while i + 1 < len(lines):
        # Check for 2-line format at current position
        if lines[i].startswith("1 ") and lines[i+1].startswith("2 "):
            triplets.append(("Unknown", lines[i], lines[i+1]))
            i += 2
        # Check for 3-line format
        elif i + 2 < len(lines) and lines[i+1].startswith("1 ") and lines[i+2].startswith("2 "):
            triplets.append((lines[i], lines[i+1], lines[i+2]))
            i += 3
        else:
            # Skip invalid header or junk line
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

    results: list[SatelliteTLE] = []
    
    for name, line1, line2 in triplets:
        try:
            sat = parse_tle(name, line1, line2)
            results.append(sat)
        except InvalidTLEError as e:
            norad_id = e.norad_id or "UNKNOWN"
            logger.warning(
                f"Skipping invalid TLE for {name} (NORAD {norad_id}): {e.message}"
            )

    if not results and any(L.strip() for L in tle_lines):
        raise AstraError("Total parse failure: no valid TLEs found in non-empty catalog input.")

    return results
