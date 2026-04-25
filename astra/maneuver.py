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


# ---------------------------------------------------------------------------
# Hohmann Transfer Planner  [FM-4 Fix — Finding #14]
# ---------------------------------------------------------------------------


def plan_hohmann(
    r_initial_km: float,
    r_target_km: float,
    isp_s: float,
    mass_kg: float,
    thrust_N: float,
    t_ignition_jd: float,
    frame: ManeuverFrame = ManeuverFrame.VNB,
) -> list[FiniteBurn]:
    """Plan a two-burn Hohmann transfer between two circular orbits.

    Computes the two impulsive delta-V maneuvers required for a classical
    Hohmann transfer, then converts each impulsive delta-V to a finite-burn
    arc using the Tsiolkovsky rocket equation and the specified engine
    parameters.

    The first burn raises the apogee from ``r_initial_km`` to ``r_target_km``.
    The second burn circularises at the target altitude.  Both burns are
    prograde (direction = (1, 0, 0) in VNB frame).

    Assumptions:
        - Both initial and target orbits are **circular**.
        - Earth's gravitational parameter μ = 398600.4418 km³/s².
        - The transfer uses instantaneous acceleration (finite-burn duration
          is computed from thrust and Isp but the impulsive ΔV is exact).
        - Coasting time on the transfer arc (half the ellipse period) is
          computed and used to schedule the second burn epoch.

    Args:
        r_initial_km: Geocentric radius of the initial circular orbit (km).
            This is altitude + EARTH_EQUATORIAL_RADIUS_KM.
        r_target_km: Geocentric radius of the target circular orbit (km).
        isp_s: Engine specific impulse (seconds).
        mass_kg: Spacecraft mass at the start of the transfer (kg).
        thrust_N: Engine thrust in Newtons.
        t_ignition_jd: Julian Date of the first burn ignition.
        frame: ManeuverFrame for thrust direction (default VNB, prograde).

    Returns:
        List of two :class:`FiniteBurn` objects — [burn_1, burn_2].

    Raises:
        ManeuverError: If the orbit radii are non-positive, target equals
            initial (null transfer), thrust or Isp are non-positive, or the
            spacecraft runs out of propellant before completing the transfer.

    Example::

        import astra, math
        from astra.constants import EARTH_EQUATORIAL_RADIUS_KM as Re
        burns = astra.plan_hohmann(
            r_initial_km = Re + 400.0,   # 400 km LEO
            r_target_km  = Re + 600.0,   # 600 km target
            isp_s        = 300.0,
            mass_kg      = 1000.0,
            thrust_N     = 10.0,
            t_ignition_jd= 2460000.5,
        )
        traj = astra.propagate_cowell(state, maneuvers=burns, ...)
    """
    import math
    from astra.constants import EARTH_MU_KM3_S2, G0_STD

    # ── Input Validation ─────────────────────────────────────────────────────
    if r_initial_km <= 0.0:
        raise ManeuverError(
            f"r_initial_km must be positive, got {r_initial_km}.",
            parameter="r_initial_km",
            value=r_initial_km,
        )
    if r_target_km <= 0.0:
        raise ManeuverError(
            f"r_target_km must be positive, got {r_target_km}.",
            parameter="r_target_km",
            value=r_target_km,
        )
    if abs(r_target_km - r_initial_km) < 1e-3:
        raise ManeuverError(
            "r_initial_km and r_target_km are effectively equal — no transfer needed.",
            parameter="r_target_km",
            value=r_target_km,
        )
    if isp_s <= 0.0:
        raise ManeuverError(
            f"isp_s must be positive, got {isp_s}.",
            parameter="isp_s",
            value=isp_s,
        )
    if thrust_N <= 0.0:
        raise ManeuverError(
            f"thrust_N must be positive, got {thrust_N}.",
            parameter="thrust_N",
            value=thrust_N,
        )
    if mass_kg <= 0.0:
        raise ManeuverError(
            f"mass_kg must be positive, got {mass_kg}.",
            parameter="mass_kg",
            value=mass_kg,
        )

    mu = EARTH_MU_KM3_S2
    g0 = G0_STD  # m/s²

    # ── Orbital mechanics ────────────────────────────────────────────────────
    # Velocities on initial and target circular orbits (km/s)
    v_initial = math.sqrt(mu / r_initial_km)
    v_target  = math.sqrt(mu / r_target_km)

    # Semi-major axis of transfer ellipse
    a_transfer = (r_initial_km + r_target_km) / 2.0

    # Velocity at periapsis and apoapsis of transfer ellipse (vis-viva)
    v_transfer_peri = math.sqrt(mu * (2.0 / r_initial_km - 1.0 / a_transfer))
    v_transfer_apo  = math.sqrt(mu * (2.0 / r_target_km  - 1.0 / a_transfer))

    # Delta-V magnitudes (km/s → m/s for Tsiolkovsky)
    dv1_km_s = abs(v_transfer_peri - v_initial)   # first burn: raise apogee
    dv2_km_s = abs(v_target - v_transfer_apo)      # second burn: circularise

    dv1_m_s = dv1_km_s * 1000.0
    dv2_m_s = dv2_km_s * 1000.0

    # ── Tsiolkovsky mass budget ───────────────────────────────────────────────
    # Propellant consumed per burn: dm = m0 * (1 - exp(-dv / (Isp * g0)))
    m_after_burn1 = mass_kg * math.exp(-dv1_m_s / (isp_s * g0))
    prop1_kg      = mass_kg - m_after_burn1

    m_after_burn2 = m_after_burn1 * math.exp(-dv2_m_s / (isp_s * g0))
    prop2_kg      = m_after_burn1 - m_after_burn2
    total_prop_kg = prop1_kg + prop2_kg

    if total_prop_kg >= mass_kg:
        raise ManeuverError(
            f"Hohmann transfer requires {total_prop_kg:.2f} kg of propellant "
            f"but spacecraft only has {mass_kg:.2f} kg.",
            parameter="mass_kg",
            value=mass_kg,
        )

    logger.info(
        "Hohmann transfer planned: r_i=%.1f km → r_f=%.1f km | "
        "ΔV1=%.3f m/s | ΔV2=%.3f m/s | prop_total=%.2f kg",
        r_initial_km, r_target_km, dv1_m_s, dv2_m_s, total_prop_kg,
    )

    # ── Burn durations ────────────────────────────────────────────────────────
    # dm/dt = F / (Isp * g0)   →   dt = dm / (dm/dt) = dm * Isp * g0 / F
    mdot = thrust_N / (isp_s * g0)         # kg/s (mass flow rate)
    duration1_s = prop1_kg / mdot
    duration2_s = prop2_kg / mdot

    # ── Coast arc (transfer half-period) ─────────────────────────────────────
    T_transfer_s = math.pi * math.sqrt(a_transfer**3 / mu)  # half-period (s)

    # Burn 1 effective midpoint epoch (approx: start of burn + half duration)
    burn1_mid_s   = duration1_s / 2.0
    coast_start_s = duration1_s  # after burn 1 cutoff

    # Burn 2 ignition = burn 1 ignition + burn 1 duration + coast time
    t_ign2_jd = t_ignition_jd + (duration1_s + T_transfer_s) / 86400.0

    # ── Thrust direction: prograde (+V in VNB frame) ──────────────────────────
    prograde_dir = (1.0, 0.0, 0.0)  # VNB: V-axis is prograde

    burn1 = FiniteBurn(
        epoch_ignition_jd=t_ignition_jd,
        duration_s=duration1_s,
        thrust_N=thrust_N,
        isp_s=isp_s,
        direction=prograde_dir,
        frame=frame,
    )
    burn2 = FiniteBurn(
        epoch_ignition_jd=t_ign2_jd,
        duration_s=duration2_s,
        thrust_N=thrust_N,
        isp_s=isp_s,
        direction=prograde_dir,
        frame=frame,
    )

    return [burn1, burn2]
