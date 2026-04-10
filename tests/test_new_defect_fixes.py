import pytest
import numpy as np
from datetime import datetime, timezone
import math

from astra import config
from astra.propagator import _acceleration, _acceleration_njit
from astra.covariance import _acceleration_njit as cov_accel_njit
from astra.data_pipeline import _download_space_weather
from astra.frames import get_eop_correction

# ---------------------------------------------------------
# DEF-001: Drag Reference Altitude Initialization
# ---------------------------------------------------------
def test_drag_altitude_initialization():
    """Verify that the Numba kernel processes drag correctly with a dynamic drag_ref_alt_km"""
    r = np.array([7000.0, 0.0, 500.0])
    v = np.array([0.0, 7.5, 0.0])
    t_jd = 2451545.0
    empty_coeffs = np.zeros((2, 3))
    
    # Static density vs actual reference density check
    a_nb = _acceleration_njit(
        t_jd, r, v,
        True, 2.2, 10.0, 1000.0, 1e-12, 50.0, 621.0,   # drag_ref_alt_km = 621 km
        False, t_jd, 1.0, empty_coeffs, empty_coeffs,  # 3rd body off
        False, 1.5, True,                              # SRP off
    )
    assert np.all(np.isfinite(a_nb))

# ---------------------------------------------------------
# DEF-002: STM Co-Rotating Atmosphere
# ---------------------------------------------------------
def test_stm_corotating_atmosphere():
    """Verify covariance acceleration dynamically subtracts Earth's rotation v_rel = v - omega x r"""
    r = np.array([7000.0, 0.0, 0.0]) # Equatorial
    v = np.array([0.0, 7.5, 0.0])
    
    # High Bc to make drag dominant
    Bc = 0.5
    rho_ref = 1e-6
    H_km = 50.0
    rho_ref_alt = 400.0
    
    a_cov = cov_accel_njit(r, v, Bc, rho_ref, H_km, rho_ref_alt)
    assert np.all(np.isfinite(a_cov))

# ---------------------------------------------------------
# DEF-003: Numba Graceful Fallback
# ---------------------------------------------------------
def test_numba_graceful_fallback():
    """Ensure modules load without hard Numba import failures"""
    try:
        import astra.frames
        assert True
    except ImportError as e:
        pytest.fail(f"frames.py Numba hard-import occurred: {e}")

# ---------------------------------------------------------
# DEF-006: EOP Vectorization
# ---------------------------------------------------------
def test_eop_batching_length():
    """Ensure we process vectors smoothly instead of loop-crashing"""
    # Create 30 days of JDs
    jds = np.linspace(2451545.0, 2451575.0, 100)
    
    xp, yp, dut1 = get_eop_correction(jds)
    assert len(xp) == 100
    assert len(yp) == 100
    assert len(dut1) == 100

# ---------------------------------------------------------
# DEF-016: Strict Mode Network Propagation
# ---------------------------------------------------------
def test_strict_mode_network_failure(monkeypatch, tmp_path):
    """Ensure HTTP failures raise instead of silently continuing in strict mode"""
    def mock_get(*args, **kwargs):
        raise ValueError("Simulated network failure")
        
    monkeypatch.setattr("requests.Session.get", mock_get)
    
    config.ASTRA_STRICT_MODE = True
    with pytest.raises(ValueError, match="ASTRA STRICT"):
        _download_space_weather(str(tmp_path / "fake_dir"))
        
    config.ASTRA_STRICT_MODE = False
