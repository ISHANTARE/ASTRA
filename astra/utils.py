from __future__ import annotations
import math
from typing import Union, Optional, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from astra.models import OrbitalState
    from astra.propagator import NumericalState

from astra.constants import EARTH_EQUATORIAL_RADIUS_KM


def vincenty_distance(
    lat1_deg: float, lon1_deg: float, lat2_deg: float, lon2_deg: float
) -> float:
    """Compute exact geodetic distance between two points on the WGS-84 ellipsoid.

    Using Vincenty's formulae. Accurate to within 0.5mm for non-antipodal points.
    Replaces Haversine spherical approx.

    .. warning::
        For points that are nearly antipodal (opposite poles of the ellipsoid),
        the Vincenty series may fail to converge (indeterminate form). In this
        case the function falls back to the Haversine great-circle approximation.
        In STRICT_MODE a ``ValueError`` is raised instead of silently degrading.

    Args:
        lat1_deg, lon1_deg: WGS-84 geodetic coordinates of point A (degrees).
        lat2_deg, lon2_deg: WGS-84 geodetic coordinates of point B (degrees).

    Returns:
        Distance in kilometers.

    Raises:
        ValueError: In STRICT_MODE if the series fails to converge (antipodal pair).
    """
    a = EARTH_EQUATORIAL_RADIUS_KM
    f = 1 / 298.257223563  # WGS-84 flattening
    b = a * (1.0 - f)  # polar radius (km)

    L = math.radians(lon2_deg - lon1_deg)
    U1 = math.atan((1 - f) * math.tan(math.radians(lat1_deg)))
    U2 = math.atan((1 - f) * math.tan(math.radians(lat2_deg)))

    sinU1, cosU1 = math.sin(U1), math.cos(U1)
    sinU2, cosU2 = math.sin(U2), math.cos(U2)

    lam = L
    converged = False
    for _ in range(200):
        sinLam, cosLam = math.sin(lam), math.cos(lam)

        sinSigma = math.sqrt(
            (cosU2 * sinLam) ** 2 + (cosU1 * sinU2 - sinU1 * cosU2 * cosLam) ** 2
        )
        if sinSigma == 0:
            return 0.0  # Co-incident points

        cosSigma = sinU1 * sinU2 + cosU1 * cosU2 * cosLam
        sigma = math.atan2(sinSigma, cosSigma)

        sinAlpha = cosU1 * cosU2 * sinLam / sinSigma
        cosSqAlpha = 1 - sinAlpha**2

        cos2SigmaM = cosSigma - 2 * sinU1 * sinU2 / cosSqAlpha if cosSqAlpha != 0 else 0

        C = f / 16 * cosSqAlpha * (4 + f * (4 - 3 * cosSqAlpha))
        lam_prev = lam
        lam = L + (1 - C) * f * sinAlpha * (
            sigma
            + C * sinSigma * (cos2SigmaM + C * cosSigma * (-1 + 2 * cos2SigmaM**2))
        )

        if abs(lam - lam_prev) < 1e-12:
            converged = True
            break

    if not converged:
        # Antipodal / near-antipodal: Vincenty indeterminate → Haversine fallback
        from astra import config

        antipodal_msg = (
            f"vincenty_distance: series failed to converge — likely antipodal pair "
            f"({lat1_deg:.4f}°, {lon1_deg:.4f}°) ↔ ({lat2_deg:.4f}°, {lon2_deg:.4f}°). "
            "Falling back to Haversine great-circle approximation."
        )
        if config.ASTRA_STRICT_MODE:
            raise ValueError(f"[ASTRA STRICT] {antipodal_msg}")
        import logging

        logging.getLogger(__name__).warning(antipodal_msg)
        # Haversine fallback using WGS-84 semi-major axis (spherical approximation).
        R = EARTH_EQUATORIAL_RADIUS_KM
        phi1, phi2 = math.radians(lat1_deg), math.radians(lat2_deg)
        dphi = phi2 - phi1
        dlam = math.radians(lon2_deg - lon1_deg)
        hav = (
            math.sin(dphi / 2) ** 2
            + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
        )
        return R * 2 * math.asin(math.sqrt(hav))

    uSq = cosSqAlpha * (a**2 - b**2) / (b**2)
    A = 1 + uSq / 16384 * (4096 + uSq * (-768 + uSq * (320 - 175 * uSq)))
    B = uSq / 1024 * (256 + uSq * (-128 + uSq * (74 - 47 * uSq)))

    deltaSigma = (
        B
        * sinSigma
        * (
            cos2SigmaM
            + B
            / 4
            * (
                cosSigma * (-1 + 2 * cos2SigmaM**2)
                - B / 6 * cos2SigmaM * (-3 + 4 * sinSigma**2) * (-3 + 4 * cos2SigmaM**2)
            )
        )
    )

    return b * A * (sigma - deltaSigma)


def orbital_elements(
    source: Union[str, np.ndarray, "OrbitalState", "NumericalState"],
    v: Optional[np.ndarray] = None,
) -> dict[str, float]:
    """Extract Keplerian orbital elements from a TLE string or Cartesian state.

    Args:
        source: Either a raw TLE Line 2 string, a (3,) position vector [km],
                or an OrbitalState / NumericalState object.
        v: Optional (3,) velocity vector [km/s] if ``source`` is a position vector.

    Returns:
        Dictionary mapping element names to their values:
        - inclination_deg
        - raan_deg
        - eccentricity
        - arg_perigee_deg
        - mean_anomaly_deg (only for TLE)
        - true_anomaly_deg (only for Cartesian)
        - semimajor_axis_km
        - period_min
    """
    import numpy as np

    # 1. TLE string path (Legacy)
    if isinstance(source, str):
        line2 = source
        e = float("0." + line2[26:33].strip())
        mm_rev_day = float(line2[52:63].strip())
        p_min = orbit_period(mm_rev_day)

        # SGP4 Mean motion is in rev/day -> rad/s for semi-major axis
        n_rad_s = (mm_rev_day * 2.0 * math.pi) / 86400.0
        from astra.constants import EARTH_MU_KM3_S2

        a = (EARTH_MU_KM3_S2 / (n_rad_s**2)) ** (1.0 / 3.0)

        return {
            "inclination_deg": float(line2[8:16].strip()),
            "raan_deg": float(line2[17:25].strip()),
            "eccentricity": e,
            "arg_perigee_deg": float(line2[34:42].strip()),
            "mean_anomaly_deg": float(line2[43:51].strip()),
            "mean_motion_rev_per_day": mm_rev_day,
            "semimajor_axis_km": a,
            "period_min": p_min,
        }

    # 2. Cartesian state path (High-fidelity)
    r = None
    vel = None

    if hasattr(source, "position_km") and hasattr(source, "velocity_km_s"):
        # OrbitalState or NumericalState — both carry position_km and velocity_km_s
        r = np.asarray(source.position_km)
        vel = np.asarray(source.velocity_km_s)
    elif isinstance(source, np.ndarray) and v is not None:
        r = source
        vel = v
    else:
        raise TypeError(f"Invalid input to orbital_elements: {type(source)}")

    from astra.constants import EARTH_MU_KM3_S2

    mu = EARTH_MU_KM3_S2

    r_mag = np.linalg.norm(r)
    v_mag = np.linalg.norm(vel)

    # h = r x v
    h = np.cross(r, vel)
    np.linalg.norm(h)

    # n = k x h (Node vector)
    n = np.array([-h[1], h[0], 0.0])
    n_mag = np.linalg.norm(n)

    # Eccentricity vector
    e_vec = ((v_mag**2 - mu / r_mag) * r - np.dot(r, vel) * vel) / mu
    ecc = float(np.linalg.norm(e_vec))

    # Energy and Semi-major axis
    energy = 0.5 * v_mag**2 - mu / r_mag
    if abs(energy) > 1e-12:
        a = -mu / (2.0 * energy)
    else:
        a = float("inf")  # Parabolic

    # Inclination (SE-B Fix: atan2 for quadrant safety)
    inc = math.atan2(math.sqrt(h[0] ** 2 + h[1] ** 2), h[2])

    # RAAN (Omega)
    if n_mag > 1e-12:
        raan = math.atan2(n[1], n[0])
    else:
        raan = 0.0  # Equatorial

    # Argument of Perigee (omega)
    if n_mag > 1e-12 and ecc > 1e-12:
        omega = math.acos(np.clip(np.dot(n, e_vec) / (n_mag * ecc), -1.0, 1.0))
        if e_vec[2] < 0:
            omega = 2.0 * math.pi - omega
    else:
        omega = 0.0  # Circular or Equatorial

    # True Anomaly (nu)
    if ecc > 1e-12:
        nu = math.acos(np.clip(np.dot(e_vec, r) / (ecc * r_mag), -1.0, 1.0))
        if np.dot(r, vel) < 0:
            nu = 2.0 * math.pi - nu
    else:
        # For circular orbits, use argument of latitude
        if n_mag > 1e-12:
            nu = math.acos(np.clip(np.dot(n, r) / (n_mag * r_mag), -1.0, 1.0))
            if r[2] < 0:
                nu = 2.0 * math.pi - nu
        else:
            nu = 0.0

    p_min = 2.0 * math.pi * math.sqrt(abs(a) ** 3 / mu) / 60.0 if a > 0 else 0.0

    return {
        "inclination_deg": math.degrees(inc),
        "raan_deg": math.degrees(raan) % 360.0,
        "eccentricity": ecc,
        "arg_perigee_deg": math.degrees(omega) % 360.0,
        "true_anomaly_deg": math.degrees(nu) % 360.0,
        "semimajor_axis_km": a,
        "period_min": p_min,
    }


def orbit_period(mean_motion_rev_per_day: float) -> float:
    """Compute the orbital period from mean motion.

    Args:
        mean_motion_rev_per_day: Mean motion in revolutions per day.

    Returns:
        Orbital period in minutes.
    """
    if mean_motion_rev_per_day <= 0:
        return float("inf")
    return (24.0 * 60.0) / mean_motion_rev_per_day
