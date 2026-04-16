from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import astra.config as config
from astra.covariance import compute_collision_probability
from astra.ocm import parse_ocm_kvn, parse_ocm_xml
from astra.orbit import propagate_orbit
from astra.propagator import DragConfig, NumericalState, propagate_cowell
from astra.spatial_index import SpatialIndex
from astra.tle import parse_tle
from astra.version import __version__


def test_parse_ocm_xml_with_default_namespace() -> None:
    xml_text = """<?xml version=\"1.0\"?>
<ocm xmlns=\"urn:ccsds:schema:ndmxml\">
  <body>
    <segment>
      <metadata>
        <OBJECT_NAME>TEST</OBJECT_NAME>
      </metadata>
      <data>
        <stateVector>
          <EPOCH>2026-01-01T00:00:00Z</EPOCH>
          <X>7000</X><Y>0</Y><Z>0</Z>
          <X_DOT>0</X_DOT><Y_DOT>7.5</Y_DOT><Z_DOT>0</Z_DOT>
        </stateVector>
      </data>
    </segment>
  </body>
</ocm>
"""

    states = parse_ocm_xml(xml_text)
    assert len(states) == 1
    np.testing.assert_allclose(states[0].position_km, np.array([7000.0, 0.0, 0.0]))


def test_parse_ocm_kvn_day_of_year_epoch() -> None:
    kvn_text = """
EPOCH = 2026-107T13:14:15.000
X = 7000
Y = 0
Z = 0
X_DOT = 0
Y_DOT = 7.5
Z_DOT = 0
"""

    states = parse_ocm_kvn(kvn_text)
    assert len(states) == 1
    assert states[0].t_jd > 0.0


def test_parse_ocm_xml_multiple_segments() -> None:
    xml_text = """<?xml version=\"1.0\"?>
<ocm xmlns=\"urn:ccsds:schema:ndmxml\">
  <body>
    <segment>
      <metadata><OBJECT_NAME>TEST-A</OBJECT_NAME></metadata>
      <data>
        <stateVector>
          <EPOCH>2026-01-01T00:00:00Z</EPOCH>
          <X>7000</X><Y>0</Y><Z>0</Z>
          <X_DOT>0</X_DOT><Y_DOT>7.5</Y_DOT><Z_DOT>0</Z_DOT>
        </stateVector>
      </data>
    </segment>
    <segment>
      <metadata><OBJECT_NAME>TEST-B</OBJECT_NAME></metadata>
      <data>
        <stateVector>
          <EPOCH>2026-01-01T00:10:00Z</EPOCH>
          <X>7100</X><Y>10</Y><Z>0</Z>
          <X_DOT>0</X_DOT><Y_DOT>7.4</Y_DOT><Z_DOT>0</Z_DOT>
        </stateVector>
      </data>
    </segment>
  </body>
</ocm>
"""

    states = parse_ocm_xml(xml_text)
    assert len(states) == 2
    assert states[1].t_jd > states[0].t_jd


def test_covariance_nonpositive_det_fail_closed_relaxed() -> None:
    prev = config.ASTRA_STRICT_MODE
    config.ASTRA_STRICT_MODE = False
    try:
        pc = compute_collision_probability(
            miss_vector_km=np.array([0.1, 0.0, 0.1]),
            rel_vel_km_s=np.array([0.0, 7.5, 0.0]),
            cov_a=np.diag([-1.0, 1.0, 1.0]),
            cov_b=np.zeros((3, 3)),
        )
    finally:
        config.ASTRA_STRICT_MODE = prev
    assert pc == 1.0


def test_covariance_nonpositive_det_raises_strict() -> None:
    prev = config.ASTRA_STRICT_MODE
    config.ASTRA_STRICT_MODE = True
    try:
        with pytest.raises(ValueError, match="ASTRA STRICT"):
            compute_collision_probability(
                miss_vector_km=np.array([0.1, 0.0, 0.1]),
                rel_vel_km_s=np.array([0.0, 7.5, 0.0]),
                cov_a=np.diag([-1.0, 1.0, 1.0]),
                cov_b=np.zeros((3, 3)),
            )
    finally:
        config.ASTRA_STRICT_MODE = prev


def test_orbit_ut1_fallback_logs_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    import astra.data_pipeline as data_pipeline
    import astra.orbit as orbit_module

    sat = parse_tle(
        "ISS (ZARYA)",
        "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927",
        "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537",
    )

    monkeypatch.setattr(
        data_pipeline,
        "get_ut1_utc_correction",
        lambda _t: (_ for _ in ()).throw(RuntimeError("ut1 unavailable")),
    )

    warning_calls: list[str] = []
    monkeypatch.setattr(
        orbit_module.logger,
        "warning",
        lambda msg, *args, **kwargs: warning_calls.append(str(msg)),
    )

    prev = config.ASTRA_STRICT_MODE
    config.ASTRA_STRICT_MODE = False
    try:
        state = propagate_orbit(sat, sat.epoch_jd, 1.0)
    finally:
        config.ASTRA_STRICT_MODE = prev

    assert state.error_code in (0, 1, 2, 3, 4, 5, 6)
    assert warning_calls
    assert any("falling back to UTC propagation" in msg for msg in warning_calls)


def test_spatial_index_uses_symmetric_per_object_radius() -> None:
    class _FakeTree:
        def __init__(self) -> None:
            self.radii: list[float] = []

        def query_ball_point(self, _point: np.ndarray, r: float) -> list[int]:
            self.radii.append(float(r))
            return []

    idx = SpatialIndex()
    idx._positions = {
        "LOW": np.array([0.0, 0.0, 0.0]),
        "HIGH": np.array([1.0, 0.0, 0.0]),
    }
    idx._ids = ["LOW", "HIGH"]
    idx._excursions = {"LOW": 1.0, "HIGH": 1000.0}
    idx._max_excursion = 1000.0
    fake_tree = _FakeTree()
    idx._tree = fake_tree

    idx.query_pairs(threshold_km=5.0)

    assert len(fake_tree.radii) == 2
    assert fake_tree.radii[0] == pytest.approx(7.0)
    assert fake_tree.radii[1] == pytest.approx(2005.0)


def test_use_empirical_drag_flag_changes_behavior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Force a dramatic empirical density to make the behavior difference clear.
    import astra.data_pipeline as data_pipeline

    monkeypatch.setattr(
        data_pipeline, "get_space_weather", lambda _t: (150.0, 150.0, 15.0)
    )
    monkeypatch.setattr(
        data_pipeline,
        "atmospheric_density_empirical",
        lambda alt_km, f107_obs, f107_adj, ap_daily: 1e-5,
    )

    state0 = NumericalState(
        t_jd=2461000.0,
        position_km=np.array([6578.137, 0.0, 0.0]),
        velocity_km_s=np.array([0.0, 7.8, 0.0]),
        mass_kg=250.0,
    )
    drag = DragConfig(cd=2.2, area_m2=5.0, mass_kg=250.0, model="Jacchia")

    states_emp = propagate_cowell(
        state0=state0,
        duration_s=120.0,
        dt_out=120.0,
        drag_config=drag,
        include_third_body=False,
        use_de=False,
        use_empirical_drag=True,
    )
    states_static = propagate_cowell(
        state0=state0,
        duration_s=120.0,
        dt_out=120.0,
        drag_config=drag,
        include_third_body=False,
        use_de=False,
        use_empirical_drag=False,
    )

    dv = float(
        np.linalg.norm(states_emp[-1].velocity_km_s - states_static[-1].velocity_km_s)
    )
    assert dv > 1e-8


def test_runtime_version_matches_pyproject() -> None:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")

    version_line = next(
        line for line in text.splitlines() if line.strip().startswith("version =")
    )
    project_version = version_line.split("=", 1)[1].strip().strip('"')
    assert __version__ == project_version
