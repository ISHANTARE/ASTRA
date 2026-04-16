import numpy as np
import math
from astra.propagator import srp_illumination_factor
from astra.constants import EARTH_EQUATORIAL_RADIUS_KM, SUN_RADIUS_KM


def test_penumbra_transition_smoothness():
    """Verify that the SRP illumination factor transitions smoothly through penumbra."""

    # Satellite at ~622km altitude (consistent with LEO)
    r_mag = 7000.0
    r_sat = np.array([r_mag, 0.0, 0.0])

    # Sun at 1 AU
    AU = 149597870.7

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
        np.array([math.cos(gamma), math.sin(gamma), 0.0])
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


if __name__ == "__main__":
    try:
        test_penumbra_transition_smoothness()
    except Exception:
        import traceback

        traceback.print_exc()
        exit(1)
