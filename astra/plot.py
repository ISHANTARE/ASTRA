"""ASTRA Core 3D Visualization Module.

Provides interactive 3D rendering of orbital trajectories and conjunction
events utilizing Plotly.
"""

from __future__ import annotations

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
