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
    """Build the 3×3 rotation matrix whose rows are the VNB unit vectors in ECI.
    **Convention (important):**
    ``_build_vnb_matrix_njit`` stores the VNB basis vectors as *rows*:
        T[0, :] = V̂  (velocity unit vector)
        T[1, :] = N̂  (orbit-normal unit vector)
        T[2, :] = B̂  (binormal unit vector)
    This makes T an **VNB→ECI** rotation matrix, so:
        a_inertial  = T @ a_vnb      # VNB direction → ECI components

    Args:
        r_eci: Shape (3,) inertial position [km].
        v_eci: Shape (3,) inertial velocity [km/s].

    Returns:
        Shape (3, 3) matrix whose columns are [V̂, N̂, B̂] in ECI (i.e. VNB→ECI).
    Raises:
        ManeuverError: If velocity magnitude < 1e-12 km/s or if r ∥ v
            (angular momentum near-zero), making the VNB frame undefined.
    Example::
        T = rotation_vnb_to_inertial(r_eci, v_eci)
        # Prograde burn of 10 m/s:
        dv_eci = T @ np.array([0.01, 0.0, 0.0])  # V̂ direction → ECI
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
    return _build_vnb_matrix_njit(r_eci, v_eci).T  # type: ignore[no-any-return]
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
    return _build_rtn_matrix_njit(r_eci, v_eci).T  # type: ignore[no-any-return]
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
    # Build dynamic rotation matrix from instantaneous state.
    # T is the VNB→ECI (or RTN→ECI) rotation matrix.
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
# Hohmann Transfer Planner
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
    For raising transfers, both burns are prograde. For lowering transfers,
    both burns are retrograde because the transfer ellipse is slower than the
    initial circular orbit at apoapsis and faster than the final circular orbit
    at periapsis.
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
        frame: ManeuverFrame for thrust direction (default VNB).
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
    # ── Coast arc & Phasing ──────────────────────────────────────────────────
    T_transfer_s = math.pi * math.sqrt(a_transfer**3 / mu)  # half-period (s)
    # Burn 1 should be centered around the target apsis (t_ignition_jd),
    # meaning ignition starts half a burn duration before the apsis.
    t_ign1_jd = t_ignition_jd - (duration1_s / 2.0) / 86400.0
    # BL-11: Guard against burn centering that shifts ignition before propagation start.
    # If t_ign1_jd is before t_ignition_jd by more than the half-duration, it means
    # the burn window extends before the caller's epoch. The integrator will clamp
    # negative ign_s to 0, silently moving the burn to t=0 instead of the apsis.
    if t_ign1_jd < t_ignition_jd - (duration1_s / 86400.0):
        logger.warning(
            "Hohmann burn 1 centering shifts ignition epoch (JD %.6f) before the "
            "propagation start. The burn will be clamped to t=0 by the integrator, "
            "executing at epoch instead of the intended apsis. Consider starting "
            "propagation earlier or reducing burn duration.",
            t_ign1_jd,
        )
    # Burn 2 should be centered around the opposite apsis (exactly T_transfer_s later),
    # meaning its ignition starts half a burn duration before that.
    t_ign2_jd = t_ignition_jd + (T_transfer_s - duration2_s / 2.0) / 86400.0
    # ── Thrust direction ─────────────────────────────────────────────────────
    # Raising transfers accelerate along-track; lowering transfers decelerate.
    # Prograde is +V in VNB and +T in RTN. Retrograde is the signed inverse.
    direction_sign = 1.0 if r_target_km > r_initial_km else -1.0
    burn_dir = (
        (direction_sign, 0.0, 0.0)
        if frame == ManeuverFrame.VNB
        else (0.0, direction_sign, 0.0)
    )
    burn1 = FiniteBurn(
        epoch_ignition_jd=t_ign1_jd,
        duration_s=duration1_s,
        thrust_N=thrust_N,
        isp_s=isp_s,
        direction=burn_dir,
        frame=frame,
    )
    burn2 = FiniteBurn(
        epoch_ignition_jd=t_ign2_jd,
        duration_s=duration2_s,
        thrust_N=thrust_N,
        isp_s=isp_s,
        direction=burn_dir,
        frame=frame,
    )
    return [burn1, burn2]

# ---------------------------------------------------------------------------
# Bi-Elliptic Transfer Planner (AS-02a)
# ---------------------------------------------------------------------------
def plan_bielliptic(
    r_initial_km: float,
    r_target_km: float,
    r_intermediate_km: float,
    isp_s: float,
    mass_kg: float,
    thrust_N: float,
    t_ignition_jd: float,
    frame: ManeuverFrame = ManeuverFrame.VNB,
) -> list[FiniteBurn]:
    """Plan a three-burn bi-elliptic transfer between two circular orbits.

    A bi-elliptic transfer is more efficient than Hohmann when the ratio
    ``r_target / r_initial > 11.94`` (Vallado §6.3.2). It uses three burns:

    1. **Burn 1:** At initial orbit, raise apoapsis to ``r_intermediate``.
    2. **Burn 2:** At ``r_intermediate``, adjust periapsis to ``r_target``.
    3. **Burn 3:** At ``r_target``, circularize.

    Assumptions:
        - Initial and target orbits are **circular**.
        - Intermediate radius must exceed both ``r_initial`` and ``r_target``.
        - Earth's gravitational parameter μ = 398600.4418 km³/s².

    Args:
        r_initial_km: Geocentric radius of the initial circular orbit (km).
        r_target_km: Geocentric radius of the target circular orbit (km).
        r_intermediate_km: Geocentric radius of the intermediate apoapsis (km).
            Must be ≥ max(r_initial_km, r_target_km).
        isp_s: Engine specific impulse (seconds).
        mass_kg: Spacecraft mass at the start of the transfer (kg).
        thrust_N: Engine thrust in Newtons.
        t_ignition_jd: Julian Date of the first burn ignition.
        frame: ManeuverFrame for thrust direction (default VNB).

    Returns:
        List of three :class:`FiniteBurn` objects — [burn_1, burn_2, burn_3].

    Raises:
        ManeuverError: If radii are invalid, intermediate is too small,
            or the spacecraft runs out of propellant.

    Example::

        import astra
        from astra.constants import EARTH_EQUATORIAL_RADIUS_KM as Re
        burns = astra.plan_bielliptic(
            r_initial_km     = Re + 300.0,
            r_target_km      = Re + 35786.0,  # GEO
            r_intermediate_km= Re + 100000.0, # high intermediate
            isp_s            = 300.0,
            mass_kg          = 2000.0,
            thrust_N         = 50.0,
            t_ignition_jd    = 2460000.5,
        )
    """
    import math
    from astra.constants import EARTH_MU_KM3_S2, G0_STD

    # ── Input Validation ─────────────────────────────────────────────────────
    for name, val in [("r_initial_km", r_initial_km), ("r_target_km", r_target_km),
                      ("r_intermediate_km", r_intermediate_km)]:
        if val <= 0.0:
            raise ManeuverError(
                f"{name} must be positive, got {val}.",
                parameter=name, value=val,
            )
    if abs(r_target_km - r_initial_km) < 1e-3:
        raise ManeuverError(
            "r_initial_km and r_target_km are effectively equal — no transfer needed.",
            parameter="r_target_km", value=r_target_km,
        )
    if r_intermediate_km < max(r_initial_km, r_target_km) - 1e-3:
        raise ManeuverError(
            f"r_intermediate_km ({r_intermediate_km:.1f} km) must be ≥ "
            f"max(r_initial, r_target) = {max(r_initial_km, r_target_km):.1f} km.",
            parameter="r_intermediate_km", value=r_intermediate_km,
        )
    for name, val in [("isp_s", isp_s), ("thrust_N", thrust_N), ("mass_kg", mass_kg)]:
        if val <= 0.0:
            raise ManeuverError(
                f"{name} must be positive, got {val}.",
                parameter=name, value=val,
            )

    mu = EARTH_MU_KM3_S2
    g0 = G0_STD

    # ── Orbital mechanics ────────────────────────────────────────────────────
    v_initial = math.sqrt(mu / r_initial_km)
    v_target  = math.sqrt(mu / r_target_km)

    # Transfer ellipse 1: r_initial → r_intermediate
    a_transfer1 = (r_initial_km + r_intermediate_km) / 2.0
    v_t1_peri = math.sqrt(mu * (2.0 / r_initial_km - 1.0 / a_transfer1))
    v_t1_apo  = math.sqrt(mu * (2.0 / r_intermediate_km - 1.0 / a_transfer1))

    # Transfer ellipse 2: r_intermediate → r_target
    a_transfer2 = (r_intermediate_km + r_target_km) / 2.0
    v_t2_apo  = math.sqrt(mu * (2.0 / r_intermediate_km - 1.0 / a_transfer2))
    v_t2_peri = math.sqrt(mu * (2.0 / r_target_km - 1.0 / a_transfer2))

    # Delta-V magnitudes (km/s → m/s)
    dv1_m_s = abs(v_t1_peri - v_initial) * 1000.0       # burn 1: raise to intermediate
    dv2_m_s = abs(v_t2_apo - v_t1_apo) * 1000.0         # burn 2: adjust at intermediate
    dv3_m_s = abs(v_target - v_t2_peri) * 1000.0         # burn 3: circularize at target

    # ── Tsiolkovsky mass budget ───────────────────────────────────────────────
    m1 = mass_kg
    m2 = m1 * math.exp(-dv1_m_s / (isp_s * g0))
    m3 = m2 * math.exp(-dv2_m_s / (isp_s * g0))
    m4 = m3 * math.exp(-dv3_m_s / (isp_s * g0))

    total_prop_kg = m1 - m4
    if total_prop_kg >= mass_kg:
        raise ManeuverError(
            f"Bi-elliptic transfer requires {total_prop_kg:.2f} kg of propellant "
            f"but spacecraft only has {mass_kg:.2f} kg.",
            parameter="mass_kg", value=mass_kg,
        )

    logger.info(
        "Bi-elliptic transfer planned: r_i=%.1f → r_int=%.1f → r_f=%.1f km | "
        "ΔV1=%.3f | ΔV2=%.3f | ΔV3=%.3f m/s | prop_total=%.2f kg",
        r_initial_km, r_intermediate_km, r_target_km,
        dv1_m_s, dv2_m_s, dv3_m_s, total_prop_kg,
    )

    # ── Burn durations ────────────────────────────────────────────────────────
    mdot = thrust_N / (isp_s * g0)
    dur1 = (m1 - m2) / mdot
    dur2 = (m2 - m3) / mdot
    dur3 = (m3 - m4) / mdot

    # ── Coast arcs & phasing ──────────────────────────────────────────────────
    T_coast1 = math.pi * math.sqrt(a_transfer1**3 / mu)  # half-period of ellipse 1
    T_coast2 = math.pi * math.sqrt(a_transfer2**3 / mu)  # half-period of ellipse 2

    t_ign1_jd = t_ignition_jd - (dur1 / 2.0) / 86400.0
    t_ign2_jd = t_ignition_jd + (T_coast1 - dur2 / 2.0) / 86400.0
    t_ign3_jd = t_ign2_jd + (dur2 / 2.0 + T_coast2 - dur3 / 2.0) / 86400.0

    # ── Thrust directions ─────────────────────────────────────────────────────
    dir_prograde = (
        (1.0, 0.0, 0.0) if frame == ManeuverFrame.VNB
        else (0.0, 1.0, 0.0)
    )
    dir_retrograde = (
        (-1.0, 0.0, 0.0) if frame == ManeuverFrame.VNB
        else (0.0, -1.0, 0.0)
    )

    # Burn 1: always prograde (raise apoapsis)
    # Burn 2: depends on whether target is above or below initial
    # Burn 3: always retrograde at target (circularize from faster transfer orbit)
    burn1_dir = dir_prograde
    burn2_dir = dir_prograde if v_t2_apo > v_t1_apo else dir_retrograde
    burn3_dir = dir_retrograde if v_t2_peri > v_target else dir_prograde

    burns = []
    for t_ign, dur, direction in [
        (t_ign1_jd, dur1, burn1_dir),
        (t_ign2_jd, dur2, burn2_dir),
        (t_ign3_jd, dur3, burn3_dir),
    ]:
        burns.append(FiniteBurn(
            epoch_ignition_jd=t_ign,
            duration_s=dur,
            thrust_N=thrust_N,
            isp_s=isp_s,
            direction=direction,
            frame=frame,
        ))
    return burns


# ---------------------------------------------------------------------------
# Inclination Change Planner (AS-02b)
# ---------------------------------------------------------------------------
def plan_inclination_change(
    r_km: float,
    delta_inc_deg: float,
    isp_s: float,
    mass_kg: float,
    thrust_N: float,
    t_ignition_jd: float,
    frame: ManeuverFrame = ManeuverFrame.VNB,
) -> list[FiniteBurn]:
    """Plan a single-burn inclination change at the ascending/descending node.

    Uses the exact plane-change ΔV formula for circular orbits:

        ΔV = 2 · v_circular · sin(Δi / 2)

    where ``v_circular = √(μ / r)`` is the orbital velocity at radius ``r``.

    The burn is applied **normal** to the orbital plane (along the N-axis in
    both VNB and RTN frames). Positive ``delta_inc_deg`` increases inclination
    (burns at ascending node); negative decreases it (burns at descending node).

    Args:
        r_km: Geocentric orbit radius (km). Must be positive.
        delta_inc_deg: Desired inclination change (degrees). Can be negative.
        isp_s: Engine specific impulse (seconds).
        mass_kg: Spacecraft mass at burn start (kg).
        thrust_N: Engine thrust in Newtons.
        t_ignition_jd: Julian Date of burn ignition (should be at the
            ascending or descending node for optimal efficiency).
        frame: ManeuverFrame for thrust direction (default VNB).

    Returns:
        List containing one :class:`FiniteBurn` with thrust along the normal axis.

    Raises:
        ManeuverError: If radius, thrust, or Isp are non-positive, or if
            delta_inc_deg is zero, or if the burn requires more propellant
            than available.

    Example::

        import astra
        from astra.constants import EARTH_EQUATORIAL_RADIUS_KM as Re
        burns = astra.plan_inclination_change(
            r_km          = Re + 35786.0,  # GEO radius
            delta_inc_deg = -0.5,          # reduce inclination by 0.5°
            isp_s         = 300.0,
            mass_kg       = 3000.0,
            thrust_N      = 22.0,
            t_ignition_jd = 2460000.5,
        )
    """
    import math
    from astra.constants import EARTH_MU_KM3_S2, G0_STD

    # ── Input Validation ─────────────────────────────────────────────────────
    if r_km <= 0.0:
        raise ManeuverError(
            f"r_km must be positive, got {r_km}.",
            parameter="r_km", value=r_km,
        )
    if abs(delta_inc_deg) < 1e-10:
        raise ManeuverError(
            "delta_inc_deg is effectively zero — no plane change needed.",
            parameter="delta_inc_deg", value=delta_inc_deg,
        )
    for name, val in [("isp_s", isp_s), ("thrust_N", thrust_N), ("mass_kg", mass_kg)]:
        if val <= 0.0:
            raise ManeuverError(
                f"{name} must be positive, got {val}.",
                parameter=name, value=val,
            )

    mu = EARTH_MU_KM3_S2
    g0 = G0_STD

    # ── Plane change ΔV ──────────────────────────────────────────────────────
    v_circ = math.sqrt(mu / r_km)
    delta_inc_rad = math.radians(delta_inc_deg)
    dv_km_s = 2.0 * v_circ * abs(math.sin(delta_inc_rad / 2.0))
    dv_m_s = dv_km_s * 1000.0

    # ── Tsiolkovsky mass budget ───────────────────────────────────────────────
    m_after = mass_kg * math.exp(-dv_m_s / (isp_s * g0))
    prop_kg = mass_kg - m_after
    if prop_kg >= mass_kg:
        raise ManeuverError(
            f"Inclination change of {delta_inc_deg:.3f}° requires "
            f"{prop_kg:.2f} kg of propellant but spacecraft only has {mass_kg:.2f} kg.",
            parameter="mass_kg", value=mass_kg,
        )

    logger.info(
        "Inclination change planned: Δi=%.3f° at r=%.1f km | "
        "ΔV=%.3f m/s | prop=%.2f kg",
        delta_inc_deg, r_km, dv_m_s, prop_kg,
    )

    # ── Burn duration ─────────────────────────────────────────────────────────
    mdot = thrust_N / (isp_s * g0)
    duration_s = prop_kg / mdot

    # ── Normal thrust direction ───────────────────────────────────────────────
    # Normal axis: +N is along angular momentum (r × v).
    # Positive delta_inc → burn in +N direction.
    # Negative delta_inc → burn in -N direction.
    normal_sign = 1.0 if delta_inc_deg > 0.0 else -1.0

    # In VNB: N is axis 1. In RTN: N is axis 2.
    if frame == ManeuverFrame.VNB:
        direction = (0.0, normal_sign, 0.0)
    else:
        direction = (0.0, 0.0, normal_sign)

    burn = FiniteBurn(
        epoch_ignition_jd=t_ignition_jd,
        duration_s=duration_s,
        thrust_N=thrust_N,
        isp_s=isp_s,
        direction=direction,
        frame=frame,
    )
    return [burn]


# ---------------------------------------------------------------------------
# Delta-V Budget Calculator (AS-04)
# ---------------------------------------------------------------------------
from dataclasses import dataclass as _dataclass


@_dataclass(frozen=True)
class DeltaVBudget:
    """Pre-propagation ΔV budget summary for a sequence of finite burns.

    Provides the total ΔV, total propellant consumption, final mass, and
    per-burn breakdown without requiring a full numerical propagation.

    Attributes:
        total_delta_v_m_s: Total ΔV across all burns (m/s).
        total_propellant_kg: Total propellant consumed (kg).
        final_mass_kg: Spacecraft mass after all burns (kg).
        burns: Per-burn breakdown as a list of dicts, each containing:
            ``index``, ``delta_v_m_s``, ``propellant_kg``,
            ``mass_before_kg``, ``mass_after_kg``, ``duration_s``,
            ``epoch_ignition_jd``.
    """
    total_delta_v_m_s: float
    total_propellant_kg: float
    final_mass_kg: float
    burns: list[dict]


def compute_delta_v_budget(
    burns: list[FiniteBurn],
    initial_mass_kg: float,
) -> DeltaVBudget:
    """Compute a pre-propagation ΔV and propellant budget for a burn sequence.

    Uses the Tsiolkovsky rocket equation sequentially on each burn to compute
    ΔV, propellant consumed, and remaining mass — without running a full
    numerical propagation.

    The ΔV for each burn is computed as:

        ΔV = Isp · g₀ · ln(m_before / m_after)

    where ``m_after = m_before - F · duration / (Isp · g₀)``.

    Args:
        burns: Ordered list of :class:`FiniteBurn` objects. Burns are processed
            sequentially; mass depleted by burn N reduces the starting mass
            for burn N+1.
        initial_mass_kg: Spacecraft mass before the first burn (kg).

    Returns:
        :class:`DeltaVBudget` with total ΔV, propellant, final mass, and
        per-burn breakdown.

    Raises:
        ManeuverError: If ``initial_mass_kg`` is non-positive, if ``burns``
            is empty, or if the spacecraft runs out of mass mid-sequence.

    Example::

        import astra
        burns = astra.plan_hohmann(
            r_initial_km=6778.0, r_target_km=7178.0,
            isp_s=300.0, mass_kg=1000.0, thrust_N=10.0,
            t_ignition_jd=2460000.5,
        )
        budget = astra.compute_delta_v_budget(burns, initial_mass_kg=1000.0)
        print(f"Total ΔV: {budget.total_delta_v_m_s:.1f} m/s")
        print(f"Propellant: {budget.total_propellant_kg:.2f} kg")
        print(f"Final mass: {budget.final_mass_kg:.2f} kg")
    """
    import math
    from astra.constants import G0_STD

    if not burns:
        raise ManeuverError(
            "burns list is empty — nothing to budget.",
            parameter="burns", value=0,
        )
    if initial_mass_kg <= 0.0:
        raise ManeuverError(
            f"initial_mass_kg must be positive, got {initial_mass_kg}.",
            parameter="initial_mass_kg", value=initial_mass_kg,
        )

    g0 = G0_STD
    current_mass = initial_mass_kg
    total_dv = 0.0
    total_prop = 0.0
    burn_details: list[dict] = []

    for i, burn in enumerate(burns):
        mdot = burn.thrust_N / (burn.isp_s * g0)
        prop_kg = mdot * burn.duration_s
        mass_after = current_mass - prop_kg

        if mass_after <= 0.0:
            raise ManeuverError(
                f"Spacecraft runs out of mass during burn {i} "
                f"(mass_before={current_mass:.2f} kg, propellant_needed={prop_kg:.2f} kg).",
                parameter="initial_mass_kg", value=initial_mass_kg,
            )

        # Tsiolkovsky: ΔV = Isp * g0 * ln(m0 / mf)
        dv_m_s = burn.isp_s * g0 * math.log(current_mass / mass_after)

        burn_details.append({
            "index": i,
            "delta_v_m_s": dv_m_s,
            "propellant_kg": prop_kg,
            "mass_before_kg": current_mass,
            "mass_after_kg": mass_after,
            "duration_s": burn.duration_s,
            "epoch_ignition_jd": burn.epoch_ignition_jd,
        })

        total_dv += dv_m_s
        total_prop += prop_kg
        current_mass = mass_after

    return DeltaVBudget(
        total_delta_v_m_s=total_dv,
        total_propellant_kg=total_prop,
        final_mass_kg=current_mass,
        burns=burn_details,
    )
