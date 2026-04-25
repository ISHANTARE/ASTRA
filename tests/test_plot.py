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
        assert len(fig.data) >= 2, (
            f"Expected >= 2 traces (Earth + satellite), got {len(fig.data)}"
        )

    def test_satellite_trace_has_correct_data_length(self):
        """The satellite scatter trace must match the input trajectory length."""
        import plotly.graph_objects as go

        T = 20
        traj = np.random.rand(T, 3) * 7000.0
        fig = plot_trajectories({"SAT1": traj})

        # Find a Scatter3d trace that is NOT the Earth sphere (which has many points)
        scatter_traces = [
            t for t in fig.data
            if isinstance(t, go.Scatter3d) and t.x is not None and len(t.x) == T
        ]
        assert len(scatter_traces) >= 1, (
            f"Expected a Scatter3d trace with {T} points, got none. "
            f"Traces: {[(type(t).__name__, len(t.x) if hasattr(t,'x') and t.x is not None else '?') for t in fig.data]}"
        )

    def test_multiple_satellites(self):
        """Figure with 3 satellites should have 3 satellite traces + Earth."""
        T = 10
        trajs = {
            "SAT1": np.random.rand(T, 3) * 7000.0,
            "SAT2": np.random.rand(T, 3) * 7500.0,
            "SAT3": np.random.rand(T, 3) * 8000.0,
        }
        fig = plot_trajectories(trajs)
        # At minimum: Earth + 3 satellite traces
        assert len(fig.data) >= 4, (
            f"Expected >= 4 traces for 3 satellites + Earth, got {len(fig.data)}"
        )

    def test_empty_trajectory_does_not_crash(self):
        """plot_trajectories with zero trajectories must return a valid figure."""
        import plotly.graph_objects as go

        fig = plot_trajectories({})
        assert isinstance(fig, go.Figure)


class TestPlotGroundTrack:
    @pytest.fixture
    def iss_tle(self):
        import astra
        return astra.parse_tle(
            "ISS (ZARYA)",
            "1 25544U 98067A   21001.00000000  .00001480  00000-0  34282-4 0  9990",
            "2 25544  51.6442 284.1199 0001364 338.5498  21.5664 15.48922536 12341",
        )

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
        assert len(scattergeo) >= 1, (
            "plot_ground_track must produce at least one Scattergeo trace"
        )

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
        assert all(-180.0 <= lo <= 180.0 for lo in lons), (
            "All ground-track longitudes must be in [-180, 180]"
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
