Quickstart Guide
================

This guide demonstrates the three core workflows requested by users to get started with ASTRA-Core.

1. Parsing TLE Data
-------------------
You can easily parse standard Two-Line Element (TLE) strings into a rigorous ``SatelliteTLE`` dataclass which auto-computes the Julian Date epoch internally.

.. code-block:: python

    import astra
    
    line1 = "1 25544U 98067A   26085.12345678  .00001234  00000-0  12345-4 0  9999"
    line2 = "2 25544  51.6400 123.4567 0005678  90.1234 270.5678 15.50000000123456"
    
    iss_tle = astra.SatelliteTLE.from_strings(line1, line2, name="ISS (ZARYA)")
    print(f"Epoch JD: {iss_tle.epoch_jd}")


2. Propagating Orbits
---------------------
Use the vectorized ``propagate_many`` function to simulate the orbit over discrete time steps.

.. code-block:: python

    import numpy as np

    # Predict the orbit over the next 2 hours in 5-minute increments
    time_steps_min = np.arange(0.0, 120.0, 5.0)
    times_jd = iss_tle.epoch_jd + (time_steps_min / 1440.0)

    # Returns an optimized mapping of NORAD IDs to (T, 3) position arrays in the TEME frame
    trajectories = astra.propagate_many([iss_tle], times_jd)
    iss_positions = trajectories[iss_tle.norad_id]
    

3. Conjunction Risk Analysis
----------------------------
Find close approaches between objects in an O(N log N) highly-optimized screening pass.

.. code-block:: python

    events = astra.find_conjunctions(
        trajectories=trajectories, 
        times_jd=times_jd, 
        elements_map={iss_tle.norad_id: astra.make_debris_object(iss_tle)}, 
        threshold_km=5.0
    )

    for event in events:
        print(f"Collision probability: {event.collision_probability}")


4. Retrieving Live CelesTrak Catalogs
-------------------------------------
Instead of manually typing TLE lines, you can directly query live space catalogs (e.g., active satellites, space stations, space weather) via the built-in catalog loading utility.

.. code-block:: python

    # Load the active satellite catalog from CelesTrak (approx 9000 objects)
    catalog = astra.load_tle_catalog(target="active")

    # Map NORAD IDs to SatelliteTLE objects
    catalog_map = {sat.norad_id: sat for sat in catalog}
    print(f"Loaded {len(catalog_map)} active satellites.")


5. Ground Station Visibility
----------------------------
ASTRA-Core can predict overhead visibility passes (AOS, TCA, LOS) for any observer coordinate globally.

.. code-block:: python

    from datetime import datetime, timezone
    
    # Define an observer in Los Angeles
    observer = astra.Observer(lat_deg=34.0522, lon_deg=-118.2437, alt_km=0.071)
    
    now = datetime.now(timezone.utc)
    target_start_jd = float(astra.convert_time(now, "jd"))
    
    # Calculate passes over the next 24 hours
    passes = astra.passes_over_location(
        satellite=iss_tle, 
        observer=observer, 
        t_start_jd=target_start_jd,
        t_end_jd=target_start_jd + 1.0, 
        step_minutes=1.0
    )
    
    for p in passes:
        print(f"Acquisition of Signal (JD): {p.aos_jd}")
        print(f"Peak Elevation: {np.degrees(p.max_elevation_rad):.1f} degrees")
