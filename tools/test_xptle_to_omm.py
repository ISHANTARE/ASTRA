import astra.omm
from astra.models import SatelliteTLE

tle = SatelliteTLE(
    norad_id="25544",
    name="ISS",
    line1="1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927",
    line2="2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537",
    epoch_jd=2454738.0,
    object_type="PAYLOAD"
)
object.__setattr__(tle, "_spacebook_source", "spacebook_xptle")

omms = astra.omm.xptle_to_satellite_omm([tle])
assert len(omms) == 1
omm = omms[0]
assert omm.norad_id == "25544"
assert omm.name == "ISS"
assert omm.inclination_rad > 0.901
assert omm.mean_motion_rad_min > 0.068
assert getattr(omm, "_spacebook_source", None) == "spacebook_xptle"

# Let's test that we can convert it to state for validation
import numpy as np
from astra.orbit import tle_to_propagatable_state
# Note: tle_to_propagatable_state handles both TLE and OMM objects due to duck typing or explicit checks.
t_state, v_state = astra.orbit.tle_to_propagatable_state(omm, np.array([omm.epoch_jd, omm.epoch_jd+1]))
print("Propagated correctly!")
print(t_state)
