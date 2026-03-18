import pytest
import numpy as np

from astra.plot import plot_trajectories

def test_plot_trajectories_returns_figure():
    import plotly.graph_objects as go
    
    # 10 step trajectory
    T = 10
    traj_a = np.zeros((T, 3))
    traj_a[:, 0] = np.linspace(100.0, 200.0, T)
    
    trajectories = {"25544": traj_a}
    
    fig = plot_trajectories(trajectories)
    
    assert isinstance(fig, go.Figure)
    # Figure should contain at least Earth surface trace and 1 scatter trace = 2
    assert len(fig.data) >= 2 
