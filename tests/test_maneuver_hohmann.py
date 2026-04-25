"""tests/test_maneuver.py — Tests for plan_hohmann and existing maneuver API.

[FM-4 Fix — Finding #14]
Validates the new plan_hohmann function against known orbital mechanics results:
- Delta-V magnitudes verified against Vallado formula
- Burn count, ordering, and direction
- Edge cases: equal radii, zero mass, negative thrust
- G0_STD usage (not hardcoded literal)
"""
import math
import pytest
import numpy as np

from astra.constants import EARTH_MU_KM3_S2, EARTH_EQUATORIAL_RADIUS_KM, G0_STD
from astra.maneuver import plan_hohmann
from astra.models import FiniteBurn, ManeuverFrame
from astra.errors import ManeuverError


Re = EARTH_EQUATORIAL_RADIUS_KM
mu = EARTH_MU_KM3_S2


class TestPlanHohmann:

    # ── Reference values ────────────────────────────────────────────────────
    # LEO 400 km → 600 km Hohmann transfer
    R1 = Re + 400.0   # 6778.137 km
    R2 = Re + 600.0   # 6978.137 km

    def _reference_dv1(self):
        a_t = (self.R1 + self.R2) / 2.0
        v1  = math.sqrt(mu / self.R1)
        vt1 = math.sqrt(mu * (2.0/self.R1 - 1.0/a_t))
        return abs(vt1 - v1) * 1000.0  # m/s

    def _reference_dv2(self):
        a_t = (self.R1 + self.R2) / 2.0
        v2  = math.sqrt(mu / self.R2)
        vt2 = math.sqrt(mu * (2.0/self.R2 - 1.0/a_t))
        return abs(v2 - vt2) * 1000.0  # m/s

    # ── Basic contract ───────────────────────────────────────────────────────

    def test_returns_two_burns(self):
        burns = plan_hohmann(
            r_initial_km=self.R1, r_target_km=self.R2,
            isp_s=300.0, mass_kg=1000.0, thrust_N=10.0,
            t_ignition_jd=2460000.5,
        )
        assert len(burns) == 2, f"Expected 2 burns, got {len(burns)}"
        assert isinstance(burns[0], FiniteBurn)
        assert isinstance(burns[1], FiniteBurn)

    def test_burn1_before_burn2(self):
        burns = plan_hohmann(
            r_initial_km=self.R1, r_target_km=self.R2,
            isp_s=300.0, mass_kg=1000.0, thrust_N=10.0,
            t_ignition_jd=2460000.5,
        )
        assert burns[0].epoch_ignition_jd < burns[1].epoch_ignition_jd, (
            "Burn 1 must ignite before Burn 2"
        )

    def test_both_burns_prograde(self):
        burns = plan_hohmann(
            r_initial_km=self.R1, r_target_km=self.R2,
            isp_s=300.0, mass_kg=1000.0, thrust_N=10.0,
            t_ignition_jd=2460000.5,
        )
        for i, burn in enumerate(burns):
            assert burn.direction[0] > 0, (
                f"Burn {i+1} direction must be prograde (V+ in VNB)"
            )

    def test_delta_v1_matches_vallado(self):
        """ΔV₁ derived from burn duration × thrust / (m_avg) ≈ reference."""
        burns = plan_hohmann(
            r_initial_km=self.R1, r_target_km=self.R2,
            isp_s=300.0, mass_kg=1000.0, thrust_N=10.0,
            t_ignition_jd=2460000.5,
        )
        ref_dv1 = self._reference_dv1()
        # Tsiolkovsky: ΔV = Isp * g0 * ln(m0/mf)
        # → mf = m0 * exp(-ΔV/(Isp*g0))
        mf = 1000.0 * math.exp(-ref_dv1 / (300.0 * G0_STD))
        prop_ref = 1000.0 - mf
        # burn duration = prop_mass / (thrust / (Isp * g0))
        mdot = 10.0 / (300.0 * G0_STD)
        dur_ref = prop_ref / mdot
        assert abs(burns[0].duration_s - dur_ref) < 0.5, (
            f"Burn 1 duration {burns[0].duration_s:.2f} s vs reference {dur_ref:.2f} s"
        )

    def test_coast_time_is_half_transfer_period(self):
        """Gap between burn1 cutoff and burn2 ignition ≈ half transfer ellipse period."""
        burns = plan_hohmann(
            r_initial_km=self.R1, r_target_km=self.R2,
            isp_s=300.0, mass_kg=1000.0, thrust_N=10.0,
            t_ignition_jd=2460000.5,
        )
        a_t = (self.R1 + self.R2) / 2.0
        T_half = math.pi * math.sqrt(a_t**3 / mu)  # seconds

        cutoff1_jd = burns[0].epoch_cutoff_jd
        ignition2_jd = burns[1].epoch_ignition_jd
        coast_s = (ignition2_jd - cutoff1_jd) * 86400.0

        assert abs(coast_s - T_half) < 2.0, (
            f"Coast time {coast_s:.1f} s vs expected half-period {T_half:.1f} s"
        )

    def test_positive_burn_durations(self):
        burns = plan_hohmann(
            r_initial_km=self.R1, r_target_km=self.R2,
            isp_s=300.0, mass_kg=1000.0, thrust_N=10.0,
            t_ignition_jd=2460000.5,
        )
        assert burns[0].duration_s > 0.0
        assert burns[1].duration_s > 0.0

    def test_uses_g0_std_not_hardcoded(self):
        """Burn duration must be consistent with G0_STD from constants."""
        burns = plan_hohmann(
            r_initial_km=self.R1, r_target_km=self.R2,
            isp_s=300.0, mass_kg=1000.0, thrust_N=10.0,
            t_ignition_jd=2460000.5,
        )
        # ΔV₁ recovered from burn duration (Tsiolkovsky)
        mdot = 10.0 / (300.0 * G0_STD)
        prop1 = burns[0].duration_s * mdot
        m_after = 1000.0 - prop1
        dv1_recovered = (300.0 * G0_STD) * math.log(1000.0 / m_after)
        ref_dv1 = self._reference_dv1()
        assert abs(dv1_recovered - ref_dv1) < 0.5, (
            f"ΔV₁ recovered ({dv1_recovered:.2f} m/s) ≠ reference ({ref_dv1:.2f} m/s). "
            "This suggests G0 constant drift."
        )

    def test_default_frame_is_vnb(self):
        burns = plan_hohmann(
            r_initial_km=self.R1, r_target_km=self.R2,
            isp_s=300.0, mass_kg=1000.0, thrust_N=10.0,
            t_ignition_jd=2460000.5,
        )
        assert burns[0].frame == ManeuverFrame.VNB
        assert burns[1].frame == ManeuverFrame.VNB

    # ── Propellant budget check ──────────────────────────────────────────────

    def test_total_propellant_below_initial_mass(self):
        burns = plan_hohmann(
            r_initial_km=self.R1, r_target_km=self.R2,
            isp_s=300.0, mass_kg=1000.0, thrust_N=10.0,
            t_ignition_jd=2460000.5,
        )
        mdot = 10.0 / (300.0 * G0_STD)
        total_prop = (burns[0].duration_s + burns[1].duration_s) * mdot
        assert total_prop < 1000.0, (
            f"Total propellant {total_prop:.2f} kg exceeds initial mass 1000 kg"
        )

    # ── Input validation / error cases ──────────────────────────────────────

    def test_raises_on_equal_radii(self):
        with pytest.raises(ManeuverError):
            plan_hohmann(
                r_initial_km=self.R1, r_target_km=self.R1,
                isp_s=300.0, mass_kg=1000.0, thrust_N=10.0,
                t_ignition_jd=2460000.5,
            )

    def test_raises_on_zero_thrust(self):
        with pytest.raises(ManeuverError):
            plan_hohmann(
                r_initial_km=self.R1, r_target_km=self.R2,
                isp_s=300.0, mass_kg=1000.0, thrust_N=0.0,
                t_ignition_jd=2460000.5,
            )

    def test_raises_on_zero_isp(self):
        with pytest.raises(ManeuverError):
            plan_hohmann(
                r_initial_km=self.R1, r_target_km=self.R2,
                isp_s=0.0, mass_kg=1000.0, thrust_N=10.0,
                t_ignition_jd=2460000.5,
            )

    def test_raises_on_negative_radius(self):
        with pytest.raises(ManeuverError):
            plan_hohmann(
                r_initial_km=-100.0, r_target_km=self.R2,
                isp_s=300.0, mass_kg=1000.0, thrust_N=10.0,
                t_ignition_jd=2460000.5,
            )

    def test_raises_on_zero_mass(self):
        with pytest.raises(ManeuverError):
            plan_hohmann(
                r_initial_km=self.R1, r_target_km=self.R2,
                isp_s=300.0, mass_kg=0.0, thrust_N=10.0,
                t_ignition_jd=2460000.5,
            )

    def test_deorbit_works(self):
        """Downward Hohmann (R2 < R1) should also return 2 valid burns."""
        burns = plan_hohmann(
            r_initial_km=self.R2, r_target_km=self.R1,  # reverse direction
            isp_s=300.0, mass_kg=1000.0, thrust_N=10.0,
            t_ignition_jd=2460000.5,
        )
        assert len(burns) == 2
        assert burns[0].duration_s > 0.0
        assert burns[1].duration_s > 0.0
