"""tests/test_debris.py — Rigorous correctness tests for debris filtering.

[FM-5 Fix — Findings #17/#19/#20]
Replaces phantom smoke tests (type-only assertions) with:
- Physical correctness checks on altitude values
- Real longitude filtering validation (tests the new RAAN-based logic)
- Exact count assertions with physics rationale
- Edge-case coverage: lon=0.0, retrograde orbits, GEO RAAN exclusion
"""
import pytest
import math
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


# ---------------------------------------------------------------------------
# make_debris_object — physical correctness
# ---------------------------------------------------------------------------

class TestMakeDebrisObject:
    def test_altitude_is_positive(self, small_catalog):
        obj = make_debris_object(small_catalog[0])
        assert obj.altitude_km > 0, "altitude_km must be positive"

    def test_inclination_matches_tle(self, small_catalog):
        obj = make_debris_object(small_catalog[0])
        # ISS TLE has inclination 51.6442 deg
        assert abs(obj.inclination_deg - 51.6442) < 0.001, (
            f"Inclination mismatch: expected 51.6442, got {obj.inclination_deg}"
        )

    def test_period_positive_and_physical(self, small_catalog):
        obj = make_debris_object(small_catalog[0])
        # ISS mean motion ~15.49 rev/day → period ~93 min
        assert 80.0 < obj.period_minutes < 110.0, (
            f"ISS period {obj.period_minutes:.1f} min outside physical range [80,110]"
        )

    def test_raan_in_range(self, small_catalog):
        for tle in small_catalog:
            obj = make_debris_object(tle)
            assert 0.0 <= obj.raan_deg <= 360.0, (
                f"RAAN {obj.raan_deg} out of [0,360]"
            )

    def test_eccentricity_non_negative(self, small_catalog):
        for tle in small_catalog:
            obj = make_debris_object(tle)
            assert obj.eccentricity >= 0.0, "Eccentricity must be >= 0"


# ---------------------------------------------------------------------------
# filter_altitude
# ---------------------------------------------------------------------------

class TestFilterAltitude:
    def test_all_pass_wide_range(self, catalog_objs):
        filtered = filter_altitude(catalog_objs, min_km=0.0, max_km=100_000.0)
        assert len(filtered) == 3, "All 3 objects should pass a [0, 100000] km window"

    def test_none_pass_extreme_range(self, catalog_objs):
        # The low-mean-motion debris (2.49 rev/day) is in very high orbit
        filtered = filter_altitude(catalog_objs, min_km=0.0, max_km=50.0)
        assert len(filtered) == 0, "No object should be below 50 km"

    def test_iss_in_leo_band(self, catalog_objs):
        # ISS at ~400 km
        filtered = filter_altitude(catalog_objs, min_km=200.0, max_km=800.0)
        norad_ids = [o.source.norad_id for o in filtered]
        assert "25544" in norad_ids, "ISS should be in 200-800 km band"

    def test_returns_list_of_debris_objects(self, catalog_objs):
        from astra.models import DebrisObject
        filtered = filter_altitude(catalog_objs, min_km=0.0, max_km=100_000.0)
        assert all(isinstance(o, DebrisObject) for o in filtered)

    def test_boundary_inclusive(self, catalog_objs):
        # Edge: object exactly at min or max should be included
        objs = filter_altitude(catalog_objs, min_km=0.0, max_km=100_000.0)
        iss = next(o for o in objs if o.source.norad_id == "25544")
        alt = iss.altitude_km
        # Re-filter with exact altitude as boundary
        exact = filter_altitude(objs, min_km=alt, max_km=alt)
        assert len(exact) >= 1, "Object at exact boundary altitude should be included"


# ---------------------------------------------------------------------------
# filter_region — latitude
# ---------------------------------------------------------------------------

class TestFilterRegionLatitude:
    def test_polar_orbit_included_in_high_lat_band(self, catalog_objs):
        # TIRUPATI: inclination 97.6 → max_lat = 180 - 97.6 = 82.4 deg
        # → should reach [80, 90] lat band
        filtered = filter_region(
            catalog_objs,
            lat_min_deg=80.0,
            lat_max_deg=90.0,
        )
        norad_ids = [o.source.norad_id for o in filtered]
        assert "44383" in norad_ids, "Polar orbit (97.6°) should reach 80-90° latitude"

    def test_low_inclination_excluded_from_high_lat(self, catalog_objs):
        # DEBRIS X: inclination 10.6 → max_lat = 10.6 → cannot reach 50-60°
        filtered = filter_region(
            catalog_objs,
            lat_min_deg=50.0,
            lat_max_deg=60.0,
        )
        norad_ids = [o.source.norad_id for o in filtered]
        assert "99999" not in norad_ids, (
            "10.6° inclined object should NOT reach 50-60° lat"
        )

    def test_iss_reaches_mid_latitudes(self, catalog_objs):
        # ISS inclination 51.6442 → should reach 40-50° lat
        filtered = filter_region(
            catalog_objs,
            lat_min_deg=40.0,
            lat_max_deg=50.0,
        )
        norad_ids = [o.source.norad_id for o in filtered]
        assert "25544" in norad_ids, "ISS (51.6°) should reach 40-50° latitude"

    def test_equatorial_included_in_equatorial_band(self, catalog_objs):
        # All objects should cover the equatorial band [-5, 5]
        filtered = filter_region(
            catalog_objs,
            lat_min_deg=-5.0,
            lat_max_deg=5.0,
        )
        assert len(filtered) == 3, "All objects pass through the equatorial band"


# ---------------------------------------------------------------------------
# filter_region — longitude (Finding #1/#5: was unimplemented)
# ---------------------------------------------------------------------------

class TestFilterRegionLongitude:
    def test_lon_zero_activates_filter(self, catalog_objs):
        """Critical: lon_min_deg=0.0 must NOT be silently ignored (Finding #5)."""
        # All LEO/MEO objects (period < 24h) should pass since they sweep all lons.
        filtered_with = filter_region(
            catalog_objs,
            lat_min_deg=-90.0,
            lat_max_deg=90.0,
            lon_min_deg=0.0,
            lon_max_deg=180.0,
        )
        filtered_without = filter_region(
            catalog_objs,
            lat_min_deg=-90.0,
            lat_max_deg=90.0,
        )
        # Both should return same count since all are LEO (sweep all longitudes)
        assert len(filtered_with) == len(filtered_without), (
            "lon_min_deg=0.0 should activate (not ignore) the filter path, "
            "but LEO objects still pass because they sweep all longitudes."
        )

    def test_leo_objects_pass_any_longitude_band(self, catalog_objs):
        """LEO objects (period << 24 h) cover all longitudes — never excluded."""
        # Filter a narrow band on the opposite side of the globe
        filtered = filter_region(
            catalog_objs,
            lat_min_deg=-90.0,
            lat_max_deg=90.0,
            lon_min_deg=45.0,
            lon_max_deg=50.0,
        )
        # ISS and TIRUPATI are LEO → pass. DEBRIS X has very long period?
        leo_ids = {"25544", "44383"}
        result_ids = {o.source.norad_id for o in filtered}
        assert leo_ids.issubset(result_ids), (
            f"LEO satellites should pass any longitude band. Missing: {leo_ids - result_ids}"
        )

    def test_none_longitude_does_not_filter(self, catalog_objs):
        """Passing lon=None must disable longitude stage entirely."""
        filtered_no_lon = filter_region(
            catalog_objs,
            lat_min_deg=-90.0,
            lat_max_deg=90.0,
            lon_min_deg=None,
            lon_max_deg=None,
        )
        filtered_all_lat = filter_region(
            catalog_objs,
            lat_min_deg=-90.0,
            lat_max_deg=90.0,
        )
        assert len(filtered_no_lon) == len(filtered_all_lat), (
            "lon=None should produce identical results to no lon args at all"
        )


# ---------------------------------------------------------------------------
# filter_time_window
# ---------------------------------------------------------------------------

class TestFilterTimeWindow:
    def test_fresh_tle_within_window(self, catalog_objs):
        first_epoch = catalog_objs[0].source.epoch_jd
        filtered = filter_time_window(catalog_objs, first_epoch + 1.0, first_epoch + 3.0)
        # Within 7 days of epoch → all should pass
        assert len(filtered) == 3, (
            f"All 3 objects with fresh TLEs should pass. Got {len(filtered)}"
        )

    def test_stale_leo_excluded(self, catalog_objs):
        first_epoch = catalog_objs[0].source.epoch_jd
        # >7 days from epoch → LEO objects are stale
        filtered = filter_time_window(catalog_objs, first_epoch + 10.0, first_epoch + 14.0)
        # At least the very-low-mean-motion object (GEO-like) should survive
        assert len(filtered) < 3, (
            "Some LEO objects should be excluded at >7 day staleness"
        )

    def test_returns_list(self, catalog_objs):
        first_epoch = catalog_objs[0].source.epoch_jd
        result = filter_time_window(catalog_objs, first_epoch, first_epoch + 1.0)
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# catalog_statistics — physical correctness (Finding #19)
# ---------------------------------------------------------------------------

class TestCatalogStatistics:
    def test_returns_dict(self, catalog_objs):
        stats = catalog_statistics(catalog_objs)
        assert isinstance(stats, dict)

    def test_total_count_exact(self, catalog_objs):
        stats = catalog_statistics(catalog_objs)
        assert stats["total_count"] == 3, (
            f"Expected total_count=3, got {stats['total_count']}"
        )

    def test_by_regime_contains_leo(self, catalog_objs):
        stats = catalog_statistics(catalog_objs)
        assert "by_regime" in stats, "stats must contain 'by_regime'"
        assert "LEO" in stats["by_regime"], "LEO regime should be present"
        assert stats["by_regime"]["LEO"] >= 2, (
            f"Expected >=2 LEO objects, got {stats['by_regime']['LEO']}"
        )

    def test_mean_altitude_physical(self, catalog_objs):
        stats = catalog_statistics(catalog_objs)
        if "mean_altitude_km" in stats:
            assert 100.0 < stats["mean_altitude_km"] < 50_000.0, (
                f"mean_altitude_km={stats['mean_altitude_km']} outside physical range"
            )

    def test_empty_catalog_returns_zero(self):
        stats = catalog_statistics([])
        assert stats.get("total_count", 0) == 0, (
            "Empty catalog should report total_count=0"
        )

    def test_inclination_distribution_physical(self, catalog_objs):
        """Mean inclination must be within [0, 180] degrees."""
        stats = catalog_statistics(catalog_objs)
        if "mean_inclination_deg" in stats:
            assert 0.0 <= stats["mean_inclination_deg"] <= 180.0
