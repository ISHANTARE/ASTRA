import pytest
from astra.tle import parse_tle, load_tle_catalog
from astra.errors import InvalidTLEError
from astra import config


def test_tle_checksum_validation_p2_7():
    """Verify that a single TLE with a bad checksum raises InvalidTLEError."""
    # This TLE is exactly 69 chars for both lines.
    name = "ISS (ZARYA)"
    line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
    line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"

    # Correct checksum for L1 is 7. Let's change it to 0.
    line1_bad = line1[:68] + "0"

    with pytest.raises(InvalidTLEError) as excinfo:
        parse_tle(name, line1_bad, line2)

    assert excinfo.value.reason == "L1_CHECKSUM"
    print("Direct parse_tle checksum validation PASSED.")


def test_tle_batch_strict_mode_p2_7():
    """Verify that a corrupt catalog fails the entire load in STRICT mode."""
    config.set_strict_mode(True)

    # Both satellites are exactly 69 chars. SAT 2 has a bad checksum on L1.
    catalog = [
        "ISS (ZARYA)",
        "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927",
        "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537",
        "CORRUPT SAT",
        "1 99999U 00000A   00000.00000000  00000-0  00000-0 0  00000"
        + " " * 9,  # Make it 69 chars
        "2 99999   0.0000   0.0000 0000000   0.0000   0.0000  0.0000000000000",
    ]

    with pytest.raises(InvalidTLEError):
        load_tle_catalog(catalog)
    print("Strict mode batch failure PASSED.")


def test_tle_batch_relaxed_mode_p2_7():
    """Verify that a corrupt catalog skips invalid records in RELAXED mode."""
    config.set_strict_mode(False)

    catalog = [
        "ISS (ZARYA)",
        "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927",
        "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537",
        "CORRUPT SAT",
        "1 99999U 00000A   00000.00000000  00000-0  00000-0 0  00000" + " " * 9,
        "2 99999   0.0000   0.0000 0000000   0.0000   0.0000  0.0000000000000",
    ]

    sats = load_tle_catalog(catalog)
    assert len(sats) == 1
    assert sats[0].norad_id == "25544"
    print("Relaxed mode batch resilience PASSED.")


if __name__ == "__main__":
    test_tle_checksum_validation_p2_7()
    test_tle_batch_strict_mode_p2_7()
    test_tle_batch_relaxed_mode_p2_7()
