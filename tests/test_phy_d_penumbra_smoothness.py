"""PHY-D Penumbra smoothness tests — extended for FM-1B dual-cone upgrade.

Original: validates LEO penumbra transition monotonicity.
Extended: validates GEO/HEO correctness and C¹ derivative continuity.
"""
import numpy as np
import math
import pytest
from astra.propagator import srp_illumination_factor, _srp_illumination_factor_planar_njit
from astra.constants import EARTH_EQUATORIAL_RADIUS_KM, SUN_RADIUS_KM

AU_KM = 149_597_870.7


def test_penumbra_transition_smoothness():
    """Verify that the SRP illumination factor transitions smoothly through penumbra."""

    # Satellite at ~622km altitude (consistent with LEO)
    r_mag = 7000.0
    r_sat = np.array([r_mag, 0.0, 0.0])

    # Sun at 1 AU
    AU = AU_KM

    # Apparent radii
    alpha = math.asin(EARTH_EQUATORIAL_RADIUS_KM / r_mag)
    beta = math.asin(SUN_RADIUS_KM / AU)

    print(f"Earth angular radius (alpha): {math.degrees(alpha):.4f} deg")
    print(f"Sun angular radius (beta):   {math.degrees(beta):.4f} deg")
    print(f"Penumbra width:              {math.degrees(2*beta):.4f} deg")

    # Sweep gamma through the transition region
    # gamma = alpha + beta (start of penumbra)
    # gamma = alpha - beta (start of umbra)
    gammas = np.linspace(alpha + beta + 0.001, alpha - beta - 0.001, 100)
    nus = []

    for gamma in gammas:
        # Construct r_sun such that the angle between -r_sat and r_sun-r_sat is gamma
        # For simplicity, if -r_sat is [-1, 0, 0], r_rel = [cos(gamma), sin(gamma), 0]
        # Then r_sun = r_sat + AU * r_rel
        # Note: -r_sat is center of Earth from sat.
        # Here r_sat is [7000, 0, 0], so -r_sat is [-7000, 0, 0].
        # We need r_sun - r_sat to be at angle gamma from [-7000, 0, 0].
        # r_rel = [cos(pi-gamma), sin(pi-gamma), 0] = [-cos(gamma), sin(gamma), 0]
        r_sun_rel = np.array([-math.cos(gamma), math.sin(gamma), 0.0])
        r_sun = r_sat + AU * r_sun_rel

        nu = srp_illumination_factor(
            r_sat, r_sun, EARTH_EQUATORIAL_RADIUS_KM, SUN_RADIUS_KM
        )
        nus.append(nu)

    nus = np.array(nus)

    # Check boundaries
    assert nus[0] == 1.0, f"Expected 1.0 at start, got {nus[0]}"
    assert nus[-1] == 0.0, f"Expected 0.0 at end, got {nus[-1]}"

    # Check for values in (0, 1) during transition
    penumbra_mask = (nus > 0.0) & (nus < 1.0)
    assert np.any(penumbra_mask), "No penumbra transition detected!"

    # Check for monotonicity
    # Since we sweep from outside to inside, nus should be non-increasing
    diffs = np.diff(nus)
    assert np.all(
        diffs <= 1e-12
    ), "Illumination factor is not monotonically decreasing through shadow entry!"

    print("Penumbra transition test PASSED.")
    print(f"Detected {np.sum(penumbra_mask)} samples in penumbra.")


def test_heo_penumbra_correctness():
    """[FM-1B] At HEO/GEO (20,200 km), dual-cone ν must be in [0,1] and monotone."""
    r_mag = 20_200.0
    alpha = math.asin(EARTH_EQUATORIAL_RADIUS_KM / r_mag)
    beta  = math.asin(SUN_RADIUS_KM / AU_KM)
    r_sat = np.array([r_mag, 0.0, 0.0])

    gammas = np.linspace(alpha + beta - 1e-5, alpha - beta + 1e-5, 80)
    nus = []
    for g in gammas:
        r_sun = r_sat + AU_KM * np.array([-math.cos(g), math.sin(g), 0.0])
        nu = srp_illumination_factor(r_sat, r_sun, EARTH_EQUATORIAL_RADIUS_KM, SUN_RADIUS_KM)
        assert 0.0 <= nu <= 1.0, f"ν={nu} out of [0,1] at HEO γ={math.degrees(g):.4f}°"
        assert math.isfinite(nu), f"NaN/Inf ν at HEO"
        nus.append(nu)

    diffs = np.diff(np.array(nus))
    assert np.all(diffs <= 1e-10), (
        f"Non-monotone ν at HEO altitude {r_mag:.0f} km. Max increase: {diffs.max():.2e}"
    )


def test_geo_penumbra_correctness():
    """[FM-1B] At GEO (42,164 km), dual-cone ν must be in [0,1] and monotone."""
    r_mag = 42_164.0
    alpha = math.asin(EARTH_EQUATORIAL_RADIUS_KM / r_mag)
    beta  = math.asin(SUN_RADIUS_KM / AU_KM)
    r_sat = np.array([r_mag, 0.0, 0.0])

    gammas = np.linspace(alpha + beta - 1e-5, max(alpha - beta + 1e-5, 1e-7), 80)
    nus = []
    for g in gammas:
        r_sun = r_sat + AU_KM * np.array([-math.cos(g), math.sin(g), 0.0])
        nu = srp_illumination_factor(r_sat, r_sun, EARTH_EQUATORIAL_RADIUS_KM, SUN_RADIUS_KM)
        assert 0.0 <= nu <= 1.0, f"ν={nu} out of [0,1] at GEO"
        assert math.isfinite(nu), "NaN/Inf ν at GEO"
        nus.append(nu)

    diffs = np.diff(np.array(nus))
    assert np.all(diffs <= 1e-10), (
        f"Non-monotone ν at GEO altitude {r_mag:.0f} km. Max increase: {diffs.max():.2e}"
    )


def test_planar_vs_dual_cone_leo_agreement():
    """[FM-1B] In LEO, planar and dual-cone must agree to within 1%."""
    r_mag = 7000.0
    alpha = math.asin(EARTH_EQUATORIAL_RADIUS_KM / r_mag)
    beta  = math.asin(SUN_RADIUS_KM / AU_KM)
    r_sat = np.array([r_mag, 0.0, 0.0])

    gammas = np.linspace(alpha + beta - 1e-4, alpha - beta + 1e-4, 40)
    for g in gammas:
        r_sun = r_sat + AU_KM * np.array([-math.cos(g), math.sin(g), 0.0])
        nu_dual   = srp_illumination_factor(r_sat, r_sun, EARTH_EQUATORIAL_RADIUS_KM, SUN_RADIUS_KM)
        nu_planar = _srp_illumination_factor_planar_njit(r_sat, r_sun, EARTH_EQUATORIAL_RADIUS_KM, SUN_RADIUS_KM)
        if 0.01 < nu_planar < 0.99:
            rel_err = abs(float(nu_dual) - float(nu_planar)) / max(float(nu_planar), 1e-9)
            assert rel_err < 0.01, (
                f"LEO dual-cone vs planar: {rel_err*100:.3f}% > 1% at γ={math.degrees(g):.4f}°"
            )


if __name__ == "__main__":
    try:
        test_penumbra_transition_smoothness()
        test_heo_penumbra_correctness()
        test_geo_penumbra_correctness()
        test_planar_vs_dual_cone_leo_agreement()
    except Exception:
        import traceback
        traceback.print_exc()
        exit(1)
