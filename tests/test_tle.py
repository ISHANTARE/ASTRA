import pytest
from astra.errors import InvalidTLEError, AstraError
from astra import parse_tle, validate_tle, load_tle_catalog

def test_parse_valid_tle(iss_tle):
    assert iss_tle.norad_id == "25544"
    assert iss_tle.name == "ISS (ZARYA)"

def test_parse_epoch_jd(iss_tle):
    assert iss_tle.epoch_jd > 2400000.0

def test_line1_wrong_length():
    with pytest.raises(InvalidTLEError) as exc:
        parse_tle("NAME", "1 25544U 98067A   21001.00000000  .00001480  00000-0  34282-4 0  999", "2 25544  51.6442 284.1199 0001364 338.5498  21.5664 15.48922536 12341")
    assert exc.value.reason == "L1_LENGTH"

def test_line2_wrong_length():
    with pytest.raises(InvalidTLEError) as exc:
        parse_tle("NAME", "1 25544U 98067A   21001.00000000  .00001480  00000-0  34282-4 0  9990", "2 25544  51.6442 284.1199 0001364 338.5498  21.5664 15.48922536 1234")
    assert exc.value.reason == "L2_LENGTH"

def test_bad_checksum_line1():
    # Line1 correct checksum is 0, so 1 is wrong
    with pytest.raises(InvalidTLEError) as exc:
        parse_tle("NAME", "1 25544U 98067A   21001.00000000  .00001480  00000-0  34282-4 0  9991", "2 25544  51.6442 284.1199 0001364 338.5498  21.5664 15.48922536 12341")
    assert exc.value.reason == "L1_CHECKSUM"

def test_bad_checksum_line2():
    # Line2 correct checksum is 1, so 6 is wrong
    with pytest.raises(InvalidTLEError) as exc:
        parse_tle("NAME", "1 25544U 98067A   21001.00000000  .00001480  00000-0  34282-4 0  9990", "2 25544  51.6442 284.1199 0001364 338.5498  21.5664 15.48922536 12346")
    assert exc.value.reason == "L2_CHECKSUM"

def test_line1_bad_start():
    # Line starts with "2 " instead of "1 " — bad prefix
    with pytest.raises(InvalidTLEError) as exc:
        parse_tle("NAME", "2 25544U 98067A   21001.00000000  .00001480  00000-0  34282-4 0  9995", "2 25544  51.6442 284.1199 0001364 338.5498  21.5664 15.48922536 12341")
    assert exc.value.reason == "L1_PREFIX"

def test_validate_valid():
    assert validate_tle("NAME", "1 25544U 98067A   21001.00000000  .00001480  00000-0  34282-4 0  9990", "2 25544  51.6442 284.1199 0001364 338.5498  21.5664 15.48922536 12341")

def test_validate_invalid():
    assert not validate_tle("NAME", "bad", "bad")

def test_load_catalog_all_valid():
    lines = [
        "ISS", "1 25544U 98067A   21001.00000000  .00001480  00000-0  34282-4 0  9990", "2 25544  51.6442 284.1199 0001364 338.5498  21.5664 15.48922536 12341",
        "ISS", "1 25544U 98067A   21001.00000000  .00001480  00000-0  34282-4 0  9990", "2 25544  51.6442 284.1199 0001364 338.5498  21.5664 15.48922536 12341",
        "ISS", "1 25544U 98067A   21001.00000000  .00001480  00000-0  34282-4 0  9990", "2 25544  51.6442 284.1199 0001364 338.5498  21.5664 15.48922536 12341",
    ]
    catalog = load_tle_catalog(lines)
    assert len(catalog) == 3

def test_load_catalog_one_invalid():
    lines = [
        "ISS", "1 25544U 98067A   21001.00000000  .00001480  00000-0  34282-4 0  9990", "2 25544  51.6442 284.1199 0001364 338.5498  21.5664 15.48922536 12341",
        "BAD", "bad", "bad",
        "ISS", "1 25544U 98067A   21001.00000000  .00001480  00000-0  34282-4 0  9990", "2 25544  51.6442 284.1199 0001364 338.5498  21.5664 15.48922536 12341",
    ]
    catalog = load_tle_catalog(lines)
    assert len(catalog) == 2

def test_load_catalog_all_invalid():
    lines = ["BAD", "bad", "bad"]
    with pytest.raises(AstraError):
        load_tle_catalog(lines)

def test_load_catalog_empty():
    assert load_tle_catalog([]) == []
