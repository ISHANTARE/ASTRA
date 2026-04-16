# astra/maneuver.py
"""ASTRA Core Maneuver Modeling — Frame Transformations & Thrust Application.

Implements the mathematics required to convert spacecraft-centric thrust
vectors (VNB / RTN) into the inertial frame and to validate maneuver
definitions before integration.

Frame definitions (all right-handed orthonormal triads)::

    VNB (Velocity, Normal, Binormal):
        V̂ = v / |v|
        N̂ = (r × v) / |r × v|
        B̂ = V̂ × N̂

    RTN (Radial, Transverse, Normal):
        R̂ = r / |r|
        N̂ = (r × v) / |r × v|
        T̂ = N̂ × R̂

References

- Vallado, D. A. (2013). *Fundamentals of Astrodynamics and Applications*, §4.7.
- Schaub & Junkins (2018). *Analytical Mechanics of Space Systems*, §14.2.
"""

from __future__ import annotations

import numpy as np

from astra.errors import ManeuverError
from astra.models import FiniteBurn, ManeuverFrame
from astra.log import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Frame Transformation Matrices
# ---------------------------------------------------------------------------


def rotation_vnb_to_inertial(
    r_eci: np.ndarray,
    v_eci: np.ndarray,
) -> np.ndarray:
    """Build the 3×3 rotation matrix from VNB to inertial (ECI/TEME).

    Columns of the returned matrix are the VNB unit vectors expressed
    in the inertial frame:

        T = [V̂ | N̂ | B̂]

    so that  a_inertial = T @ a_vnb.

    Args:
        r_eci: Shape (3,) inertial position [km].
        v_eci: Shape (3,) inertial velocity [km/s].

    Returns:
        Shape (3, 3) rotation matrix.

    Raises:
        ManeuverError: If position or velocity magnitudes are degenerate
            (< 1e-12 km or km/s), making the frame undefined.
    """
    v_mag = np.linalg.norm(v_eci)
    if v_mag < 1e-12:
        raise ManeuverError(
            "Cannot construct VNB frame: velocity magnitude is near-zero.",
            parameter="v_eci",
            value=float(v_mag),
        )

    h = np.cross(r_eci, v_eci)  # angular momentum vector
    h_mag = np.linalg.norm(h)
    if h_mag < 1e-12:
        raise ManeuverError(
            "Cannot construct VNB frame: position and velocity are nearly parallel "
            "(r ∥ v), so the orbital plane — and VNB normal — is undefined. "
            "Use a state with finite transverse velocity or a different frame.",
            parameter="h_mag",
            value=float(h_mag),
        )

    from astra.frames import _build_vnb_matrix_njit

    return _build_vnb_matrix_njit(r_eci, v_eci)  # type: ignore[no-any-return]


def rotation_rtn_to_inertial(
    r_eci: np.ndarray,
    v_eci: np.ndarray,
) -> np.ndarray:
    """Build the 3×3 rotation matrix from RTN to inertial (ECI/TEME).

    Columns of the returned matrix are the RTN unit vectors expressed
    in the inertial frame:

        T = [R̂ | T̂ | N̂]

    so that  a_inertial = T @ a_rtn.

    Args:
        r_eci: Shape (3,) inertial position [km].
        v_eci: Shape (3,) inertial velocity [km/s].

    Returns:
        Shape (3, 3) rotation matrix.

    Raises:
        ManeuverError: If position magnitude is degenerate (< 1e-12 km),
            making the frame undefined.
    """
    r_mag = np.linalg.norm(r_eci)
    if r_mag < 1e-12:
        raise ManeuverError(
            "Cannot construct RTN frame: position magnitude is near-zero.",
            parameter="r_eci",
            value=float(r_mag),
        )

    h = np.cross(r_eci, v_eci)
    h_mag = np.linalg.norm(h)
    if h_mag < 1e-12:
        raise ManeuverError(
            "Cannot construct RTN frame: position and velocity are nearly parallel "
            "(r ∥ v), so the orbital plane — and RTN normal — is undefined. "
            "Use a state with finite transverse velocity or a different frame.",
            parameter="h_mag",
            value=float(h_mag),
        )

    from astra.frames import _build_rtn_matrix_njit

    return _build_rtn_matrix_njit(r_eci, v_eci)  # type: ignore[no-any-return]


def frame_to_inertial(
    r_eci: np.ndarray,
    v_eci: np.ndarray,
    frame: ManeuverFrame,
) -> np.ndarray:
    """Return the appropriate frame-to-inertial rotation matrix.

    Convenience dispatcher that selects VNB or RTN based on the
    ``ManeuverFrame`` enum value.

    Args:
        r_eci: Shape (3,) inertial position [km].
        v_eci: Shape (3,) inertial velocity [km/s].
        frame: ManeuverFrame.VNB or ManeuverFrame.RTN.

    Returns:
        Shape (3, 3) rotation matrix  T  such that
        ``a_inertial = T @ a_frame``.
    """
    if frame is ManeuverFrame.VNB:
        return rotation_vnb_to_inertial(r_eci, v_eci)  # type: ignore[no-any-return]
    elif frame is ManeuverFrame.RTN:
        return rotation_rtn_to_inertial(r_eci, v_eci)  # type: ignore[no-any-return]
    else:
        raise ManeuverError(
            f"Unknown ManeuverFrame: {frame!r}",
            parameter="frame",
            value=str(frame),
        )


# ---------------------------------------------------------------------------
# Thrust Vector Computation (used inside ODE derivative)
# ---------------------------------------------------------------------------


def thrust_acceleration_inertial(
    r_eci: np.ndarray,
    v_eci: np.ndarray,
    mass_kg: float,
    burn: FiniteBurn,
) -> np.ndarray:
    """Compute the inertial thrust acceleration vector at a single instant.

    This function is called at *every* Runge-Kutta sub-step during a
    powered arc.  It re-computes the frame transformation matrix from
    the instantaneous position and velocity, ensuring that dynamically
    steered burns (e.g. gravity-turn, velocity-aligned orbit-raise)
    perfectly track the commanded attitude.

    Args:
        r_eci: Shape (3,) inertial position [km].
        v_eci: Shape (3,) inertial velocity [km/s].
        mass_kg: Instantaneous spacecraft mass [kg] (must be > 0).
        burn: Active ``FiniteBurn`` definition.

    Returns:
        Shape (3,) inertial acceleration [km/s²].

    Raises:
        ManeuverError: If mass is non-positive (propellant exhausted).
    """
    if mass_kg <= 0.0:
        raise ManeuverError(
            "Spacecraft mass has reached zero — propellant exhausted "
            "before engine cutoff.",
            parameter="mass_kg",
            value=mass_kg,
        )

    # Build dynamic rotation matrix from instantaneous state
    T = frame_to_inertial(r_eci, v_eci, burn.frame)

    # Direction in the body-centric frame (unit vector)
    d_frame = np.asarray(burn.direction, dtype=np.float64)

    # Rotate thrust direction into inertial frame
    d_inertial = T @ d_frame

    # Thrust acceleration:  a = F·d̂ / m   [N / kg = m/s²]
    # Convert to km/s²:  1 m/s² = 1e-3 km/s²
    a_thrust_km_s2 = (burn.thrust_N / mass_kg) * 1e-3 * d_inertial

    return a_thrust_km_s2  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Maneuver Validation
# ---------------------------------------------------------------------------


def validate_burn(burn: FiniteBurn, initial_mass_kg: float) -> None:
    """Pre-flight validation of a FiniteBurn definition.

    Checks physical consistency before handing the burn off to the
    integrator.  Raises ``ManeuverError`` on the first detected issue.

    Checks performed:
        1. Duration is strictly positive.
        2. Thrust is strictly positive.
        3. Specific impulse is strictly positive.
        4. Direction vector has unit magnitude (within 1e-6 tolerance).
        5. Total propellant consumed does not exceed available mass.

    Args:
        burn: The ``FiniteBurn`` to validate.
        initial_mass_kg: Spacecraft wet mass at ignition [kg].

    Raises:
        ManeuverError: Descriptive error on validation failure.
    """
    if burn.duration_s <= 0.0:
        raise ManeuverError(
            "Burn duration must be positive.",
            parameter="duration_s",
            value=burn.duration_s,
        )

    if burn.thrust_N <= 0.0:
        raise ManeuverError(
            "Thrust magnitude must be positive.",
            parameter="thrust_N",
            value=burn.thrust_N,
        )

    if burn.isp_s <= 0.0:
        raise ManeuverError(
            "Specific impulse must be positive.",
            parameter="isp_s",
            value=burn.isp_s,
        )

    d = np.asarray(burn.direction, dtype=np.float64)
    d_mag = np.linalg.norm(d)
    if abs(d_mag - 1.0) > 1e-6:
        raise ManeuverError(
            f"Thrust direction must be a unit vector (|d| = {d_mag:.8f}).",
            parameter="direction",
            value=tuple(burn.direction),
        )

    # Tsiolkovsky mass check
    propellant_consumed_kg = burn.mass_flow_rate_kg_s * burn.duration_s
    if propellant_consumed_kg > initial_mass_kg:
        raise ManeuverError(
            f"Burn consumes {propellant_consumed_kg:.2f} kg of propellant, "
            f"but spacecraft only has {initial_mass_kg:.2f} kg at ignition.",
            parameter="mass_budget",
            value=propellant_consumed_kg,
        )

    logger.info(
        f"Burn validated: F={burn.thrust_N:.1f} N, "
        f"Isp={burn.isp_s:.1f} s, duration={burn.duration_s:.1f} s, "
        f"propellant={propellant_consumed_kg:.2f} kg, "
        f"frame={burn.frame.value}"
    )


def validate_burn_sequence(burns: list[FiniteBurn]) -> None:
    """Ensure that a list of FiniteBurn objects does not contain temporal overlaps.

    This function detects unphysical "dual-thrust" arcs (remediates PHY-F/SE-G).
    It assumes the burns list is already sorted by ignition time.

    Args:
        burns: Sorted list of ``FiniteBurn`` objects.

    Raises:
        ManeuverError: If an overlap is detected between any two burns.
    """
    for i in range(len(burns) - 1):
        b1 = burns[i]
        b2 = burns[i + 1]
        if b1.epoch_cutoff_jd > b2.epoch_ignition_jd + 1e-12:
            raise ManeuverError(
                f"Temporal overlap detected between maneuver {i} and {i+1}. "
                f"Maneuver {i} cutoff: {b1.epoch_cutoff_jd:.8f} JD, "
                f"Maneuver {i+1} ignition: {b2.epoch_ignition_jd:.8f} JD.",
                parameter="maneuvers",
                value=len(burns),
            )
