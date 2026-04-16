from astra.errors import (
    AstraError,
    InvalidTLEError,
    PropagationError,
    SpaceWeatherError,
)


def test_astra_error_base():
    """Test message formatting and context serialization for base AstraError."""
    # Without context
    err1 = AstraError("Base error occurred")
    assert str(err1) == "Base error occurred"
    assert err1.context == {}

    # With context
    err2 = AstraError("Operation failed", target_id="12345", retry=True)
    assert "Operation failed" in str(err2)
    assert "[target_id='12345', retry=True]" in str(err2)
    assert err2.context["target_id"] == "12345"
    assert err2.context["retry"] is True


def test_invalid_tle_error():
    """Test InvalidTLEError metadata inheritance."""
    err = InvalidTLEError(
        "Checksum mismatch",
        norad_id="99999",
        invalid_line="1 99999U 00000A   ...",
        reason="CHECKSUM_FAIL",
    )
    assert err.norad_id == "99999"
    assert err.invalid_line is not None
    assert err.reason == "CHECKSUM_FAIL"
    assert "Checksum mismatch" in str(err)
    assert "norad_id='99999'" in str(err)

    # Assert isinstance AstraError
    assert isinstance(err, AstraError)


def test_propagation_error():
    """Test PropagationError metadata."""
    err = PropagationError(
        "SGP4 divergence", norad_id="12345", error_code=6, t_jd=2460000.5
    )
    assert err.norad_id == "12345"
    assert err.error_code == 6
    assert err.t_jd == 2460000.5
    assert isinstance(err, AstraError)


def test_space_weather_error_strict_mode():
    """Test SpaceWeatherError which is only raised in strict mode."""
    # Just testing instantiation, actual conditional raise is tested in data pipeline tests
    err = SpaceWeatherError("No F10.7 data available")
    assert str(err) == "No F10.7 data available"
    assert isinstance(err, AstraError)
