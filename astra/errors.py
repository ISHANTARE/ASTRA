# astra/errors.py
"""ASTRA Core exception hierarchy.

All custom exceptions for the ASTRA library. Every error condition
maps to a typed exception that carries full diagnostic information.

Hierarchy:
    AstraError (base)
    ├── InvalidTLEError       — TLE parsing and validation failures
    ├── PropagationError      — Orbit propagation failures (SGP4, Cowell IVP, etc.)
    ├── FilterError           — Invalid filter configuration
    ├── CoordinateError       — Coordinate frame conversion failures
    ├── ManeuverError         — Maneuver definition or execution failures
    ├── SpaceWeatherError     — Solar flux/Ap data unavailable in STRICT_MODE
    ├── EphemerisError        — JPL ephemeris data cannot be loaded in STRICT_MODE
    └── SpacebookError        — COMSPOC Spacebook API failures
        └── SpacebookLookupError  — NORAD ID cannot be resolved to a COMSPOC GUID
"""

from __future__ import annotations

from typing import Any, Optional


class AstraError(Exception):
    """Base class for all ASTRA Core exceptions.

    All ASTRA exceptions derive from this class, allowing callers
    to catch ``AstraError`` for broad handling or specific subclasses
    for targeted handling.

    Args:
        message: Human-readable error description.
        **context: Arbitrary diagnostic key-value pairs attached to the error.
    """

    def __init__(self, message: str, **context: Any) -> None:
        super().__init__(message)
        self.message: str = message
        self.context: dict[str, Any] = context

    def __str__(self) -> str:
        if self.context:
            ctx = ", ".join(
                f"{k}={v!r}" for k, v in self.context.items() if v is not None
            )
            return f"{self.message} [{ctx}]" if ctx else self.message
        return self.message


class InvalidTLEError(AstraError):
    """Raised when TLE parsing or validation fails.

    Args:
        message: Human-readable error description.
        norad_id: NORAD catalog number of the offending object, if known.
        object_name: Human-readable name, if known.
        invalid_line: The raw TLE line that caused the failure.
        reason: Machine-readable failure reason code (see docs).
    """

    def __init__(
        self,
        message: str,
        norad_id: Optional[str] = None,
        object_name: Optional[str] = None,
        invalid_line: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> None:
        super().__init__(
            message,
            norad_id=norad_id,
            object_name=object_name,
            invalid_line=invalid_line,
            reason=reason,
        )
        self.norad_id: Optional[str] = norad_id
        self.object_name: Optional[str] = object_name
        self.invalid_line: Optional[str] = invalid_line
        self.reason: Optional[str] = reason


class PropagationError(AstraError):
    """Raised when numerical orbit propagation fails (SGP4 or Cowell integrator).

    Args:
        message: Human-readable error description.
        norad_id: NORAD catalog number of the failing satellite (SGP4 paths).
        error_code: SGP4 internal error code when applicable (0 = success, 1–6 = failure).
        t_jd: Julian Date at which propagation failed.
    """

    def __init__(
        self,
        message: str,
        norad_id: Optional[str] = None,
        error_code: Optional[int] = None,
        t_jd: Optional[float] = None,
    ) -> None:
        super().__init__(
            message,
            norad_id=norad_id,
            error_code=error_code,
            t_jd=t_jd,
        )
        self.norad_id: Optional[str] = norad_id
        self.error_code: Optional[int] = error_code
        self.t_jd: Optional[float] = t_jd


class FilterError(AstraError):
    """Raised when filter parameters are invalid or contradictory.

    Args:
        message: Human-readable error description.
        parameter: Name of the offending parameter.
        value: The invalid value that was supplied.
    """

    def __init__(
        self,
        message: str,
        parameter: Optional[str] = None,
        value: Optional[Any] = None,
    ) -> None:
        super().__init__(message, parameter=parameter, value=value)
        self.parameter: Optional[str] = parameter
        self.value: Optional[Any] = value


class CoordinateError(AstraError):
    """Raised when coordinate frame conversion fails.

    Args:
        message: Human-readable error description.
        frame: The coordinate frame involved (e.g. ``"TEME"``, ``"GCRS"``).
    """

    def __init__(
        self,
        message: str,
        frame: Optional[str] = None,
    ) -> None:
        super().__init__(message, frame=frame)
        self.frame: Optional[str] = frame


class ManeuverError(AstraError):
    """Raised when maneuver definition or execution fails.

    Covers validation issues (non-unit direction vector, negative Isp,
    mass depletion exceeding available propellant) and runtime failures
    during the 7-DOF powered integration phase.

    Args:
        message: Human-readable error description.
        parameter: Name of the offending parameter, if applicable.
        value: The invalid value, if applicable.
    """

    def __init__(
        self,
        message: str,
        parameter: Optional[str] = None,
        value: Optional[Any] = None,
    ) -> None:
        super().__init__(message, parameter=parameter, value=value)
        self.parameter: Optional[str] = parameter
        self.value: Optional[Any] = value


class SpaceWeatherError(AstraError):
    """Raised when solar flux/Ap data is unavailable in STRICT_MODE.

    Triggered by ``get_space_weather()`` when the CelesTrak cache is empty
    or stale and ASTRA_STRICT_MODE is True. In relaxed mode, synthetic
    defaults (F10.7=150, Ap=15) are substituted with a WARNING log.
    """


class EphemerisError(AstraError):
    """Raised when JPL ephemeris data cannot be loaded in STRICT_MODE.

    Triggered by Sun/Moon position functions when the DE421 BSP file is
    missing and ASTRA_STRICT_MODE is True. In relaxed mode, the low-fidelity
    analytical approximation is substituted with a WARNING log.
    """


class SpacebookError(AstraError):
    """Raised when a COMSPOC Spacebook API call fails or returns unexpected data.

    This is the base class for all Spacebook-related failures. It is raised
    when an HTTP request to a Spacebook endpoint fails with a non-200 status,
    times out, or returns a response body that cannot be parsed by the
    expected format (TLE, fixed-width text, JSON, or STK ephemeris).

    Callers that wish to catch any Spacebook data error but still allow
    general ASTRA errors to propagate should catch ``SpacebookError`` directly.
    Callers that want to tolerate all data failures should catch ``AstraError``.

    Args:
        message: Human-readable error description.
        url: The Spacebook API URL that triggered the failure, if known.
        status_code: HTTP status code returned, if the request reached the server.
    """

    def __init__(
        self,
        message: str,
        url: Optional[str] = None,
        status_code: Optional[int] = None,
    ) -> None:
        super().__init__(message, url=url, status_code=status_code)
        self.url: Optional[str] = url
        self.status_code: Optional[int] = status_code


class SpacebookLookupError(SpacebookError):
    """Raised when a NORAD Catalog ID cannot be resolved to a COMSPOC GUID.

    COMSPOC assigns each tracked space object a UUID-based GUID (distinct
    from the public NORAD ID). Several Spacebook per-object API endpoints
    require this GUID. This error is raised when a NORAD ID passed to
    ``get_norad_guid()`` is not found in the Spacebook satellite catalog.

    Common causes:
    - The satellite is tracked by Space-Track but not yet by COMSPOC.
    - The satellite has decayed and was removed from the active catalog.
    - The catalog cache is stale; calling ``refresh_satcat_cache()``
      to force a re-download may resolve the issue.

    Args:
        message: Human-readable error description.
        norad_id: The NORAD ID that could not be resolved.
    """

    def __init__(
        self,
        message: str,
        norad_id: Optional[int] = None,
    ) -> None:
        super().__init__(message)
        self.norad_id: Optional[int] = norad_id
