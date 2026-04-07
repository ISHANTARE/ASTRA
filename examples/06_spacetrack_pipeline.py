"""ASTRA-Core: Space-Track Pipeline Example

This example demonstrates the authenticated Space-Track.org data pipeline.
It requires a valid Space-Track account. The best practice is to set credentials
in environment variables prior to running the script.

For example (in your shell):
  export SPACETRACK_USER=your_email@domain.com
  export SPACETRACK_PASS=your_password
  
(On Windows Command Prompt use `setx` instead of `export`, then restart your terminal)

If you don't have credentials set, the engine will raise an AstraError with instructions.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import astra

def main():
    print("==========================================================")
    print(" ASTRA Space-Track Data Pipeline Demonstration")
    print("==========================================================\n")

    # For demonstration purposes, we will mock the credentials if they are missing
    # so the file doesn't crash on automated CI. However, the requests would fail 
    # if actually sent with dummy credentials. 
    user = os.environ.get("SPACETRACK_USER")
    if not user:
        print("[WARNING] Space-Track credentials not found.")
        print("Set SPACETRACK_USER and SPACETRACK_PASS environment variables.")
        print("Using test credentials (WILL LIKELY FAIL NETWORK REQUEST)")
        os.environ["SPACETRACK_USER"] = "demo@example.com"
        os.environ["SPACETRACK_PASS"] = "demo-password"

    print("1. Authenticating and Fetching 'starlink' (OMM format)...")
    try:
        starlinks_omm = astra.fetch_spacetrack_group("starlink", format="json")
        print(f"   [SUCCESS] Retrieved {len(starlinks_omm)} active Starlink OMM records.")
        if starlinks_omm:
            first = starlinks_omm[0]
            print(f"   [INFO] First record: {first.name.strip()} (NORAD {first.norad_id})")
            print(f"   [INFO] RCS (m²): {first.rcs_m2}, Mass (kg): {first.mass_kg}")
    except astra.errors.AstraError as e:
        print(f"   [ERROR] {e}")


    print("\n2. Re-fetching 'starlink' (TLE format) using cached session...")
    try:
        # Since _SESSION_CACHE works behind the scenes, this will skip the /ajaxauth/login round trip
        starlinks_tle = astra.fetch_spacetrack_group("starlink", format="tle")
        print(f"   [SUCCESS] Retrieved {len(starlinks_tle)} active Starlink TLE records.")
    except astra.errors.AstraError as e:
        print(f"   [ERROR] {e}")

    
    print("\n3. Testing SATCAT Endpoint...")
    try:
        satcat_data = astra.fetch_spacetrack_satcat(norad_ids=["25544"]) # ISS
        if satcat_data:
            iss_meta = satcat_data[0]
            print(f"   [SUCCESS] Retrieved SATCAT metadata for ISS:")
            print(f"             Name: {iss_meta.get('SATNAME')}")
            print(f"             Launch Date: {iss_meta.get('LAUNCH')}")
            print(f"             Decay Date: {iss_meta.get('DECAY', 'Not Decayed')}")
            print(f"             Country: {iss_meta.get('COUNTRY')}")
    except astra.errors.AstraError as e:
        print(f"   [ERROR] {e}")

    print("\n4. Logging out to clear session cache...")
    astra.spacetrack_logout()
    print("   [SUCCESS] Session cache cleared.")

if __name__ == "__main__":
    main()
