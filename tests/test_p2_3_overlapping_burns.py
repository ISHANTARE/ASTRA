import pytest
import numpy as np
from astra.models import FiniteBurn, ManeuverFrame
from astra.propagator import propagate_cowell, NumericalState
from astra.errors import ManeuverError

def test_overlapping_burns_p2_3():
    """Verify that overlapping FiniteBurn sequences are detected and raised."""
    # J2000 epoch
    epoch = 2451545.0
    
    # Burn 1: [0, 100s]
    burn1 = FiniteBurn(
        epoch_ignition_jd=epoch,
        duration_s=100.0,
        thrust_N=10.0,
        isp_s=300.0,
        direction=(1.0, 0.0, 0.0),
        frame=ManeuverFrame.VNB
    )
    
    # Burn 2: [50s, 150s] -> Overlaps with Burn 1 by 50 seconds
    burn2 = FiniteBurn(
        epoch_ignition_jd=epoch + (50.0 / 86400.0),
        duration_s=100.0,
        thrust_N=10.0,
        isp_s=300.0,
        direction=(0.0, 1.0, 0.0),
        frame=ManeuverFrame.VNB
    )
    
    state0 = NumericalState(
        t_jd=epoch,
        position_km=np.array([7000.0, 0.0, 0.0]),
        velocity_km_s=np.array([0.0, 7.5, 0.0]),
        mass_kg=1000.0
    )
    
    # This should raise ManeuverError
    with pytest.raises(ManeuverError) as excinfo:
        propagate_cowell(state0, 200.0, maneuvers=[burn1, burn2])
    
    assert "Temporal overlap detected" in str(excinfo.value)
    print("Overlapping burn detection PASSED.")

def test_non_overlapping_burns_p2_3():
    """Verify that non-overlapping burns propagate correctly."""
    epoch = 2451545.0
    
    # Burn 1: [0, 100s]
    burn1 = FiniteBurn(
        epoch_ignition_jd=epoch,
        duration_s=100.0,
        thrust_N=10.0,
        isp_s=300.0,
        direction=(1.0, 0.0, 0.0),
        frame=ManeuverFrame.VNB
    )
    
    # Burn 2: [110s, 210s] -> 10s gap
    burn2 = FiniteBurn(
        epoch_ignition_jd=epoch + (110.0 / 86400.0),
        duration_s=100.0,
        thrust_N=10.0,
        isp_s=300.0,
        direction=(0.0, 1.0, 0.0),
        frame=ManeuverFrame.VNB
    )
    
    state0 = NumericalState(
        t_jd=epoch,
        position_km=np.array([7000.0, 0.0, 0.0]),
        velocity_km_s=np.array([0.0, 7.5, 0.0]),
        mass_kg=1000.0
    )
    
    # This should NOT raise error
    states = propagate_cowell(state0, 250.0, maneuvers=[burn1, burn2])
    assert len(states) > 0
    print("Non-overlapping burn propagation PASSED.")

if __name__ == "__main__":
    # If run directly without pytest
    test_overlapping_burns_p2_3()
    test_non_overlapping_burns_p2_3()
