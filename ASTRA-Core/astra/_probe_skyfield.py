import numpy as np
import skyfield
print(skyfield.__version__)
try:
    from skyfield.sgp4lib import TEME_to_ITRF
    print("TEME_to_ITRF exists")
except ImportError:
    print("No TEME_to_ITRF")

try:
    from skyfield.framelib import teme
    print("teme frame exists")
except ImportError:
    print("No teme frame")
