import pytest
from datetime import datetime, timezone
import skyfield.timelib

from astra.time import convert_time


def test_convert_time_str_to_jd():
    jd = convert_time("2000-01-01T12:00:00Z", "jd")
    assert jd == 2451545.0


def test_convert_time_datetime_to_jd():
    dt = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    jd = convert_time(dt, "jd")
    assert jd == 2451545.0


def test_convert_time_float_to_datetime():
    dt = convert_time(2451545.0, "datetime")
    assert dt == datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def test_convert_time_iso_output():
    iso = convert_time(2451545.0, "iso")
    assert iso == "2000-01-01T12:00:00Z"


def test_convert_time_skyfield():
    ts = convert_time(2451545.0, "skyfield")
    assert isinstance(ts, skyfield.timelib.Time)


def test_convert_time_invalid_format():
    with pytest.raises(ValueError):
        convert_time(2451545.0, "invalid")


def test_convert_time_invalid_input():
    with pytest.raises(ValueError):
        convert_time([1, 2, 3], "jd")  # type: ignore
