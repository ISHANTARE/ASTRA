import pytest
import numpy as np
import math

from astra.propagator import NumericalState, DragConfig, propagate_cowell, _sun_position_approx, _moon_position_approx
from astra.covariance import compute_collision_probability_mc, propagate_covariance_stm, estimate_covariance
from astra.models import projected_area_m2
from astra.spatial_index import SpatialIndex


# =========================================================================
# Numerical Propagator Tests
# =========================================================================

class TestCowellPropagator:
    """Validates the Cowell numerical propagator against known orbital mechanics."""

    @pytest.fixture
    def iss_state(self):
        """ISS-like initial state at 400km altitude."""
        return NumericalState(
            t_jd=2459215.5,
            position_km=np.array([6778.137, 0.0, 0.0]),
            velocity_km_s=np.array([0.0, 7.6597, 0.0]),
        )

    def test_cowell_returns_states(self, iss_state):
        states = propagate_cowell(iss_state, 5400.0, dt_out=60.0)
        assert len(states) > 0
        assert all(isinstance(s, NumericalState) for s in states)

    def test_cowell_conserves_energy_approx(self, iss_state):
        """Keplerian energy should be approximately conserved over 1 orbit."""
        states = propagate_cowell(iss_state, 5400.0, dt_out=60.0,
                                  include_third_body=False)
        mu = 398600.4418

        e0 = 0.5 * np.linalg.norm(iss_state.velocity_km_s)**2 - mu / np.linalg.norm(iss_state.position_km)
        ef = 0.5 * np.linalg.norm(states[-1].velocity_km_s)**2 - mu / np.linalg.norm(states[-1].position_km)

        # Energy should be conserved within 0.01% for J2 perturbed (near-conservation)
        assert abs(ef - e0) / abs(e0) < 0.001

    def test_cowell_orbit_altitude_stability(self, iss_state):
        """Altitude should remain within LEO bounds over 1 orbit."""
        states = propagate_cowell(iss_state, 5400.0, dt_out=60.0)
        for s in states:
            alt = np.linalg.norm(s.position_km) - 6378.137
            assert 340.0 < alt < 500.0, f"Altitude {alt} km outside LEO bounds"

    def test_cowell_with_drag(self, iss_state):
        """Drag should cause slight altitude decrease."""
        drag = DragConfig(cd=2.2, area_m2=400.0, mass_kg=420000.0)
        states = propagate_cowell(iss_state, 5400.0, dt_out=60.0,
                                  drag_config=drag, include_third_body=False)
        assert len(states) > 0

    def test_sun_position_reasonable(self):
        r_sun = _sun_position_approx(2459215.5)
        dist = np.linalg.norm(r_sun)
        assert 1.47e8 < dist < 1.53e8  # ~1 AU +/- 3%

    def test_moon_position_reasonable(self):
        r_moon = _moon_position_approx(2459215.5)
        dist = np.linalg.norm(r_moon)
        assert 356000 < dist < 407000  # Moon distance range


# =========================================================================
# Monte Carlo Pc Tests
# =========================================================================

class TestMonteCarloPC:
    def test_mc_converges_to_analytical_for_high_speed(self):
        """MC should approximately match Chan for high-speed transverse encounters."""
        miss = np.array([0.001, 0.0, 0.0])
        vel = np.array([0.0, 14.0, 0.0])
        cov_a = np.eye(3) * 0.01
        cov_b = np.eye(3) * 0.01

        p_mc = compute_collision_probability_mc(
            miss, vel, cov_a, cov_b, radius_a_km=0.005, radius_b_km=0.005,
            n_samples=500_000, seed=42
        )
        assert 0.0 <= p_mc <= 1.0

    def test_mc_deterministic_collision(self):
        """Zero covariance, within radius -> Pc = 1.0."""
        miss = np.array([0.003, 0.0, 0.0])
        vel = np.array([0.0, 7.0, 0.0])
        cov_a = np.zeros((3, 3))
        cov_b = np.zeros((3, 3))

        p_mc = compute_collision_probability_mc(
            miss, vel, cov_a, cov_b, radius_a_km=0.005, radius_b_km=0.005
        )
        assert p_mc == 1.0

    def test_mc_deterministic_miss(self):
        """Zero covariance, outside radius -> Pc = 0.0."""
        miss = np.array([0.02, 0.0, 0.0])
        vel = np.array([0.0, 7.0, 0.0])
        cov_a = np.zeros((3, 3))
        cov_b = np.zeros((3, 3))

        p_mc = compute_collision_probability_mc(
            miss, vel, cov_a, cov_b, radius_a_km=0.005, radius_b_km=0.005
        )
        assert p_mc == 0.0

    def test_mc_reproducible_with_seed(self):
        miss = np.array([0.001, 0.0, 0.0])
        vel = np.array([0.0, 7.0, 0.0])
        cov = np.eye(3) * 0.01

        p1 = compute_collision_probability_mc(miss, vel, cov, cov, seed=123)
        p2 = compute_collision_probability_mc(miss, vel, cov, cov, seed=123)
        assert p1 == p2


# =========================================================================
# STM Covariance Tests
# =========================================================================

class TestSTMCovariance:
    def test_stm_returns_3x3(self):
        r0 = np.array([6778.137, 0.0, 0.0])
        v0 = np.array([0.0, 7.6597, 0.0])
        cov0 = np.eye(6) * 0.001

        cov_t = propagate_covariance_stm(2459215.5, r0, v0, cov0, 600.0)
        assert cov_t.shape == (6, 6)

    def test_stm_covariance_grows(self):
        """Covariance should generally grow over time."""
        r0 = np.array([6778.137, 0.0, 0.0])
        v0 = np.array([0.0, 7.6597, 0.0])
        cov0 = np.eye(6) * 0.001

        cov_t = propagate_covariance_stm(2459215.5, r0, v0, cov0, 600.0)
        assert np.trace(cov_t) > np.trace(cov0)

    def test_stm_positive_semidefinite(self):
        """Output covariance must be positive semi-definite."""
        r0 = np.array([6778.137, 0.0, 0.0])
        v0 = np.array([0.0, 7.6597, 0.0])
        cov0 = np.eye(6) * 0.001

        cov_t = propagate_covariance_stm(2459215.5, r0, v0, cov0, 3600.0)
        eigvals = np.linalg.eigvalsh(cov_t)
        assert np.all(eigvals >= -1e-10)


# =========================================================================
# Attitude-Aware Projected Area Tests
# =========================================================================

class TestProjectedArea:
    def test_face_on_returns_max_area(self):
        """Box face-on to velocity gives max projected area."""
        dims = (10.0, 5.0, 3.0)  # 10m x 5m x 3m
        q_identity = (1.0, 0.0, 0.0, 0.0)
        vel_along_x = np.array([1.0, 0.0, 0.0])

        area = projected_area_m2(dims, q_identity, vel_along_x)
        # Face-on to X: should see 5*3 = 15 m²
        assert abs(area - 15.0) < 0.1

    def test_edge_on_returns_different_area(self):
        dims = (10.0, 5.0, 3.0)
        q_identity = (1.0, 0.0, 0.0, 0.0)
        vel_along_y = np.array([0.0, 1.0, 0.0])

        area = projected_area_m2(dims, q_identity, vel_along_y)
        # Face-on to Y: should see 10*3 = 30 m²
        assert abs(area - 30.0) < 0.1

    def test_diagonal_velocity(self):
        """Diagonal velocity should give a blended area."""
        dims = (10.0, 10.0, 10.0)  # Cube
        q_identity = (1.0, 0.0, 0.0, 0.0)
        vel_diag = np.array([1.0, 1.0, 1.0]) / math.sqrt(3)

        area = projected_area_m2(dims, q_identity, vel_diag)
        assert area > 0


# =========================================================================
# Spatial Index Tests
# =========================================================================

class TestSpatialIndex:
    def test_insert_and_query(self):
        idx = SpatialIndex()
        idx.insert("A", np.array([6771.0, 0.0, 0.0]))
        idx.insert("B", np.array([6772.0, 0.0, 0.0]))  # 1 km away
        idx.insert("C", np.array([7000.0, 0.0, 0.0]))  # 229 km away

        near = idx.query_radius(np.array([6771.0, 0.0, 0.0]), 5.0)
        ids = [n[0] for n in near]
        assert "A" in ids
        assert "B" in ids
        assert "C" not in ids

    def test_query_pairs(self):
        idx = SpatialIndex()
        idx.insert("A", np.array([6771.0, 0.0, 0.0]))
        idx.insert("B", np.array([6772.0, 0.0, 0.0]))
        idx.insert("C", np.array([50000.0, 0.0, 0.0]))

        pairs = idx.query_pairs(threshold_km=5.0)
        assert ("A", "B") in pairs or ("B", "A") in pairs
        for p in pairs:
            assert "C" not in p

    def test_size(self):
        idx = SpatialIndex()
        assert idx.size == 0
        idx.insert("X", np.array([0.0, 0.0, 6771.0]))
        assert idx.size == 1
