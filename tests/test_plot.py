"""tests/test_plot.py — Rigorous tests for visualization functions.

[FM-5 Fix — Finding #18]
Replaces smoke tests (isinstance-only) with data-level assertions:
- Verifies trace count and data shapes
- Tests plot_ground_track returns a Scattergeo figure with correct traces
- Tests wrap-around longitude handling
"""
import numpy as np
import pytest

from astra.plot import plot_trajectories


_RNG = np.random.default_rng(20260502)


class TestPlotTrajectories:
    def test_returns_plotly_figure(self):
        import plotly.graph_objects as go

        T = 10
        traj_a = np.zeros((T, 3))
        traj_a[:, 0] = np.linspace(6800.0, 7000.0, T)
        fig = plot_trajectories({"25544": traj_a})
        assert isinstance(fig, go.Figure), "Must return a Plotly Figure"

    def test_contains_earth_and_satellite_traces(self):
        """Figure must have at least 2 traces: Earth sphere + satellite path."""
        import plotly.graph_objects as go

        T = 10
        traj = np.zeros((T, 3))
        traj[:, 0] = np.linspace(6800.0, 7000.0, T)
        fig = plot_trajectories({"ISS": traj})
        assert [trace.name for trace in fig.data] == ["Earth", "NORAD ISS"]

    def test_satellite_trace_has_correct_data_length(self):
        """The satellite scatter trace must match the input trajectory length."""
        import plotly.graph_objects as go

        T = 20
        traj = _RNG.random((T, 3)) * 7000.0
        fig = plot_trajectories({"SAT1": traj})

        sat_trace = next(t for t in fig.data if isinstance(t, go.Scatter3d))
        assert sat_trace.name == "NORAD SAT1"
        np.testing.assert_allclose(np.asarray(sat_trace.x), traj[:, 0])
        np.testing.assert_allclose(np.asarray(sat_trace.y), traj[:, 1])
        np.testing.assert_allclose(np.asarray(sat_trace.z), traj[:, 2])

    def test_multiple_satellites(self):
        """Figure with 3 satellites should have 3 satellite traces + Earth."""
        T = 10
        trajs = {
            "SAT1": _RNG.random((T, 3)) * 7000.0,
            "SAT2": _RNG.random((T, 3)) * 7500.0,
            "SAT3": _RNG.random((T, 3)) * 8000.0,
        }
        fig = plot_trajectories(trajs)
        assert [trace.name for trace in fig.data] == [
            "Earth",
            "NORAD SAT1",
            "NORAD SAT2",
            "NORAD SAT3",
        ]

    def test_empty_trajectory_does_not_crash(self):
        """plot_trajectories with zero trajectories must return a valid figure."""
        import plotly.graph_objects as go

        fig = plot_trajectories({})
        assert isinstance(fig, go.Figure)
        assert [trace.name for trace in fig.data] == ["Earth"]


class TestPlotGroundTrack:
    @pytest.fixture
    def iss_tle(self):
        import astra
        return astra.parse_tle(
            "ISS (ZARYA)",
            "1 25544U 98067A   21001.00000000  .00001480  00000-0  34282-4 0  9990",
            "2 25544  51.6442 284.1199 0001364 338.5498  21.5664 15.48922536 12341",
        )

    @pytest.fixture(autouse=True)
    def offline_ground_track(self, monkeypatch):
        import astra.orbit as orbit

        def fake_propagate_trajectory(satellite, t_start_jd, t_end_jd, step_minutes):
            count = int(round((t_end_jd - t_start_jd) * 1440.0 / step_minutes)) + 1
            times = np.linspace(t_start_jd, t_end_jd, count)
            positions = np.column_stack(
                (
                    np.linspace(6800.0, 7000.0, count),
                    np.linspace(0.0, 500.0, count),
                    np.linspace(-250.0, 250.0, count),
                )
            )
            velocities = np.zeros_like(positions)
            return times, positions, velocities

        def fake_ground_track(positions_teme, times_jd):
            count = len(times_jd)
            split = max(1, count // 2)
            lats = np.linspace(-51.0, 51.0, count)
            lons = np.concatenate(
                (
                    np.linspace(160.0, 179.0, split, endpoint=True),
                    np.linspace(-179.0, -160.0, count - split, endpoint=True),
                )
            )
            alts = np.full(count, 420.0)
            return list(zip(lats, lons, alts))

        monkeypatch.setattr(orbit, "propagate_trajectory", fake_propagate_trajectory)
        monkeypatch.setattr(orbit, "ground_track", fake_ground_track)

    def test_returns_plotly_figure(self, iss_tle):
        import plotly.graph_objects as go
        from astra.plot import plot_ground_track

        t_start = iss_tle.epoch_jd
        t_end = t_start + 100.0 / 1440.0  # 100 minutes
        fig = plot_ground_track(iss_tle, t_start, t_end, step_s=60.0)
        assert isinstance(fig, go.Figure), "plot_ground_track must return a Plotly Figure"

    def test_has_scattergeo_trace(self, iss_tle):
        import plotly.graph_objects as go
        from astra.plot import plot_ground_track

        t_start = iss_tle.epoch_jd
        t_end = t_start + 100.0 / 1440.0
        fig = plot_ground_track(iss_tle, t_start, t_end, step_s=60.0)

        scattergeo = [t for t in fig.data if isinstance(t, go.Scattergeo)]
        assert len(scattergeo) == 2
        assert scattergeo[0].mode == "lines"
        assert scattergeo[1].name == "Endpoints"

    def test_ground_track_lat_bounds(self, iss_tle):
        """ISS latitudes must stay within inclination bounds [−51.6, +51.6]."""
        from astra.plot import plot_ground_track

        t_start = iss_tle.epoch_jd
        t_end = t_start + 200.0 / 1440.0  # 200 minutes — ~2 orbits
        fig = plot_ground_track(iss_tle, t_start, t_end, step_s=30.0)

        # Extract lat values from the ground-track line trace (first Scattergeo)
        import plotly.graph_objects as go
        line_trace = next(
            t for t in fig.data
            if isinstance(t, go.Scattergeo) and t.mode == "lines"
        )
        lats = [v for v in line_trace.lat if v is not None]
        assert len(lats) > 0, "Ground track must have latitude data points"

        max_lat = max(abs(l) for l in lats)
        assert max_lat <= 52.0, (
            f"ISS max |latitude| should be ≤52°, got {max_lat:.2f}°"
        )

    def test_ground_track_lon_wraps(self, iss_tle):
        """Longitude values must be in [−180, 180] range."""
        from astra.plot import plot_ground_track
        import plotly.graph_objects as go

        t_start = iss_tle.epoch_jd
        t_end = t_start + 200.0 / 1440.0
        fig = plot_ground_track(iss_tle, t_start, t_end, step_s=30.0)

        line_trace = next(
            t for t in fig.data
            if isinstance(t, go.Scattergeo) and t.mode == "lines"
        )
        lons = [v for v in line_trace.lon if v is not None]
        assert len(lons) > 0
        assert all(-180.0 <= lo <= 180.0 for lo in lons), (
            "All ground-track longitudes must be in [-180, 180]"
        )
        assert any(v is None for v in line_trace.lon), (
            "A two-orbit ISS ground track should include a wrap break at +/-180 degrees"
        )

    def test_accepts_omm_format(self, iss_tle):
        """plot_ground_track must accept OMM-backed SatelliteState (format-agnostic)."""
        import plotly.graph_objects as go
        from astra.plot import plot_ground_track
        from astra.omm import parse_omm_record

        rec = {
            "OBJECT_NAME": "ISS (ZARYA)",
            "OBJECT_ID": "1998-067A",
            "NORAD_CAT_ID": "25544",
            "OBJECT_TYPE": "PAYLOAD",
            "EPOCH": "2021-01-01T00:00:00.000000",
            "MEAN_MOTION": "15.48922536",
            "ECCENTRICITY": ".0001364",
            "INCLINATION": "51.6442",
            "RA_OF_ASC_NODE": "284.1199",
            "ARG_OF_PERICENTER": "338.5498",
            "MEAN_ANOMALY": "21.5664",
            "BSTAR": ".34282E-4",
            "RCS_SIZE": "LARGE",
        }
        iss_omm = parse_omm_record(rec)
        t_start = iss_omm.epoch_jd
        t_end = t_start + 100.0 / 1440.0
        fig = plot_ground_track(iss_omm, t_start, t_end, step_s=60.0)
        assert isinstance(fig, go.Figure)
        assert [trace.name for trace in fig.data] == ["ISS (ZARYA)", "Endpoints"]
