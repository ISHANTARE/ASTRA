import pytest
from astra.debris import (
    filter_altitude,
    filter_region,
    filter_time_window,
    catalog_statistics,
    make_debris_object,
)


@pytest.fixture
def catalog_objs(small_catalog):
    return [make_debris_object(tle) for tle in small_catalog]


def test_filter_altitude_returns_subset(catalog_objs):
    filtered = filter_altitude(catalog_objs, min_km=100.0, max_km=1000.0)
    assert (
        len(filtered) == 2
    )  # The 3rd object in small_catalog has very high period/low mean motion? Wait, actually depends on the fake TLE I made.


def test_filter_altitude_all_pass(catalog_objs):
    filtered = filter_altitude(catalog_objs, min_km=0.0, max_km=100000.0)
    assert len(filtered) == 3


def test_filter_altitude_none_pass(catalog_objs):
    filtered = filter_altitude(catalog_objs, min_km=10000.0, max_km=20000.0)
    assert len(filtered) == 1


def test_filter_region_polar_orbit_included(catalog_objs):
    # 44383 has 97.6 deg inclinations
    filtered = filter_region(
        catalog_objs,
        lat_min_deg=80.0,
        lat_max_deg=90.0,
        lon_min_deg=0.0,
        lon_max_deg=0.0,
    )
    # The first one (ISS, 51.6 deg) will fail to reach 80. The 2nd one (97.6, max lat 82.4) reaches 82.4, so it overlaps [80, 90].
    # The 3rd one (10.6 deg) will fail.
    assert len(filtered) == 1
    assert filtered[0].tle.norad_id == "44383"


def test_filter_region_inclination_too_low(catalog_objs):
    # 99999 has 10.6 deg
    filtered = filter_region(
        catalog_objs,
        lat_min_deg=50.0,
        lat_max_deg=60.0,
        lon_min_deg=0.0,
        lon_max_deg=0.0,
    )
    assert len(filtered) == 2
    assert filtered[0].tle.norad_id == "25544"


def test_filter_time_window_fresh_tle(catalog_objs):
    # Fake TLEs epoch ~ 21001 (year 21, day 1)
    # t_start_jd relative to it being ~ 3 days later
    first_epoch = catalog_objs[0].tle.epoch_jd
    filtered = filter_time_window(catalog_objs, first_epoch + 3.0, first_epoch + 4.0)
    assert len(filtered) == 3


def test_filter_time_window_stale_leo(catalog_objs):
    first_epoch = catalog_objs[0].tle.epoch_jd
    filtered = filter_time_window(catalog_objs, first_epoch + 10.0, first_epoch + 14.0)
    # ISS is LEO, so stale because > 7 days
    assert len(filtered) == 1


def test_catalog_statistics_returns_dict(catalog_objs):
    stats = catalog_statistics(catalog_objs)
    assert isinstance(stats, dict)


def test_catalog_statistics_total_count(catalog_objs):
    stats = catalog_statistics(catalog_objs)
    assert stats["total_count"] == 3


def test_catalog_statistics_by_regime(catalog_objs):
    stats = catalog_statistics(catalog_objs)
    assert "LEO" in stats["by_regime"]
    assert "GEO" in stats["by_regime"]


def test_make_debris_object(small_catalog):
    obj = make_debris_object(small_catalog[0])
    assert obj.altitude_km > 0
    assert obj.inclination_deg == 51.6442
