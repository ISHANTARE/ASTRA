from __future__ import annotations
"""ASTRA Core 3D Visualization Module.

Provides interactive 3D rendering of orbital trajectories and conjunction
events utilizing Plotly.
"""

from typing import Any

import numpy as np
import plotly.graph_objects as go

from astra.models import ConjunctionEvent, TrajectoryMap
from astra.constants import EARTH_RADIUS_KM

_EARTH_RADIUS_KM = EARTH_RADIUS_KM


def _add_earth_sphere(fig: go.Figure, resolution: int = 50) -> None:
    """Adds a 3D Earth wireframe or surface to the figure."""
    theta = np.linspace(0, 2 * np.pi, resolution)
    phi = np.linspace(0, np.pi, resolution)
    x = _EARTH_RADIUS_KM * np.outer(np.cos(theta), np.sin(phi))
    y = _EARTH_RADIUS_KM * np.outer(np.sin(theta), np.sin(phi))
    z = _EARTH_RADIUS_KM * np.outer(np.ones(resolution), np.cos(phi))

    fig.add_trace(
        go.Surface(
            x=x,
            y=y,
            z=z,
            colorscale="Blues",
            opacity=0.3,
            showscale=False,
            name="Earth",
        )
    )


def plot_trajectories(
    trajectories: TrajectoryMap,
    events: list[ConjunctionEvent] | None = None,
    title: str = "ASTRA Core: Orbital Trajectories & Conjunctions",
) -> go.Figure:
    """Creates an interactive 3D Plotly figure of the orbital tracks.

    Args:
        trajectories: Dict mapping NORAD IDs to (T, 3) position arrays in km.
        events: Optional list of ConjunctionEvents to highlight dynamically.
        title: Title of the rendered plot.

    Returns:
        go.Figure: A Plotly figure object ready to be shown via `fig.show()`.
    """
    fig = go.Figure()
    _add_earth_sphere(fig)

    # Plot each trajectory track
    for nid, traj in trajectories.items():
        # Sub-sample highly dense arrays if needed for renderer performance
        step = max(1, len(traj) // 500)

        fig.add_trace(
            go.Scatter3d(
                x=traj[::step, 0],
                y=traj[::step, 1],
                z=traj[::step, 2],
                mode="lines",
                line=dict(width=2),
                name=f"NORAD {nid}",
                opacity=0.7,
            )
        )

    # Highlight Conjunction TCA Points
    if events:
        for i, ev in enumerate(events):
            # Midpoint for label
            (ev.position_a_km + ev.position_b_km) / 2.0
            text_label = f"Risk: {ev.risk_level}<br>Miss: {ev.miss_distance_km:.2f} km"

            fig.add_trace(
                go.Scatter3d(
                    x=[ev.position_a_km[0], ev.position_b_km[0]],
                    y=[ev.position_a_km[1], ev.position_b_km[1]],
                    z=[ev.position_a_km[2], ev.position_b_km[2]],
                    mode="markers+lines+text",
                    marker=dict(size=5, color="red"),
                    line=dict(color="red", width=4, dash="dash"),
                    text=["", text_label],
                    textposition="top center",
                    name=f"TCA Match {i+1} ({ev.object_a_id} & {ev.object_b_id})",
                )
            )

    # Clean styling
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X (km TEME)",
            yaxis_title="Y (km TEME)",
            zaxis_title="Z (km TEME)",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, b=0, t=40),
    )

    return fig


def plot_ground_track(
    satellite: "Any",
    t_start_jd: float,
    t_end_jd: float,
    observers: "list[Any] | None" = None,
    step_s: float = 60.0,
    title: str = "ASTRA Core: Ground Track",
) -> "go.Figure":
    """Plot the ground track of a satellite on a 2-D world map.

    Propagates the satellite using SGP4 and projects positions to geodetic
    coordinates, then renders an interactive Plotly Scattergeo figure.

    The data pipeline is already fully implemented:
    ``ground_track()`` in ``orbit.py`` → ``teme_to_ecef()`` + ``ecef_to_geodetic_wgs84()``
    in ``frames.py``.  This function assembles the final visualisation layer.

    Args:
        satellite: A ``SatelliteTLE`` or ``SatelliteOMM`` instance.
        t_start_jd: Propagation start epoch as Julian Date.
        t_end_jd: Propagation end epoch as Julian Date.
        observers: Optional list of ``Observer`` objects to mark on the map.
        step_s: Time step between ground-track points in seconds (default 60 s).
        title: Figure title.

    Returns:
        Interactive Plotly ``go.Figure`` with a Scattergeo world-map trace.

    Example::

        import astra
        iss = astra.parse_tle("ISS", line1, line2)
        fig = astra.plot_ground_track(iss, t_start_jd, t_end_jd)
        fig.show()
    """
    from astra.orbit import propagate_trajectory, ground_track

    # Step 1: propagate trajectory to get TEME positions and Julian Date array.
    # propagate_trajectory(satellite, t_start_jd, t_end_jd, step_minutes) is the
    # current public API — ground_track() is a post-propagation TEME→geodetic converter.
    step_min = step_s / 60.0
    times_jd, positions_teme, _ = propagate_trajectory(
        satellite, t_start_jd, t_end_jd, step_minutes=step_min
    )

    # Step 2: convert TEME positions to geodetic (lat, lon, alt) tuples.
    points = ground_track(positions_teme, times_jd)

    if not points:
        fig = go.Figure()
        fig.update_layout(title=f"{title} (no data)")
        return fig

    lats = [p[0] for p in points]
    lons = [p[1] for p in points]
    alts = [p[2] for p in points]

    # Detect and break longitude wrap-arounds (>180 deg jump) for clean lines
    # Insert None to lift the pen when the track wraps around 180 deg.
    lat_plot: list = []
    lon_plot: list = []
    for i in range(len(lons)):
        if i > 0 and abs(lons[i] - lons[i - 1]) > 180.0:
            lat_plot.append(None)
            lon_plot.append(None)
        lat_plot.append(lats[i])
        lon_plot.append(lons[i])

    fig = go.Figure()

    fig.add_trace(
        go.Scattergeo(
            lat=lat_plot,
            lon=lon_plot,
            mode="lines",
            line=dict(width=1.5, color="royalblue"),
            name=getattr(satellite, "name", "Satellite"),
            hovertemplate=(
                "Lat: %{lat:.2f}°<br>Lon: %{lon:.2f}°<extra></extra>"
            ),
        )
    )

    # Mark start and end
    fig.add_trace(
        go.Scattergeo(
            lat=[lats[0], lats[-1]],
            lon=[lons[0], lons[-1]],
            mode="markers+text",
            marker=dict(size=8, color=["green", "red"], symbol="circle"),
            text=["Start", "End"],
            textposition="top center",
            name="Endpoints",
        )
    )

    # Mark observer locations if provided
    if observers:
        obs_lats = [o.latitude_deg for o in observers]
        obs_lons = [o.longitude_deg for o in observers]
        obs_names = [o.name for o in observers]
        fig.add_trace(
            go.Scattergeo(
                lat=obs_lats,
                lon=obs_lons,
                mode="markers+text",
                marker=dict(size=10, color="orange", symbol="triangle-up"),
                text=obs_names,
                textposition="top center",
                name="Ground Stations",
            )
        )

    fig.update_layout(
        title=title,
        geo=dict(
            showland=True,
            landcolor="rgb(230, 230, 230)",
            showocean=True,
            oceancolor="rgb(200, 220, 255)",
            showcoastlines=True,
            coastlinecolor="gray",
            showlakes=True,
            lakecolor="rgb(200, 220, 255)",
            showcountries=True,
            countrycolor="gray",
            projection_type="natural earth",
        ),
        margin=dict(l=0, r=0, b=0, t=40),
    )

    return fig
