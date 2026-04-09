# Spacebook API — Verified Schema Notes
# Generated: 2026-04-09 by Checkpoint 1 live exploration
# All schemas verified against live data from https://spacebook.com/api

---

## Endpoint Status (Live Test Results)

| Endpoint | Status | Size | Notes |
|---|---|---|---|
| `/api/entity/tle` | ✅ 200 | 4.2 MB | Standard 3-line TLE format |
| `/api/entity/tle/YYYY-MM-DD` | ✅ 200 | 4.2 MB | Same format, historical date |
| `/api/entity/xp-tle` | ✅ 200 | 4.1 MB | Same 3-line TLE format, higher precision |
| `/api/spaceweather/recent` | ✅ 200 | 338 KB | CSSI/CelesTrak text format |
| `/api/spaceweather/full` | ✅ 200 | 3.4 MB | Same format, full history |
| `/api/eop/recent` | ✅ 200 | 215 KB | IERS Bulletin A text format |
| `/api/eop/full` | ✅ 200 | 2.4 MB | Same format, full history |
| `/api/entity/satcat` | ✅ 200 | 19.2 MB | JSON array, 68k+ records |
| `/api/entity/satcat/details` | ✅ 200 | 64.8 MB | JSON array, richer per-object data |
| `/api/entity/satcat/csv` | ✅ 200 | 10.1 MB | CSV, 17 columns, UTF-8 BOM |
| `/api/entity/synthetic-covariance/{GUID}` | ✅ 200 | 2.6 MB | **STK .e ephemeris format (NOT JSON!)** |
| `/api/entity/synthetic-covariance-plotly/{GUID}` | ✅ 200 | 451 KB | **HTML Plotly page (NOT JSON!)** |
| `/api/entity/reference-ephemerides/{GUID}` | ❌ 500 | 0 B | Internal server error |
| `/api/entity/reference-ephemerides/ocm/{GUID}` | ❌ 500 | 0 B | Internal server error |

**ISS GUID:** `bf72c797-cee3-45b2-8de1-5bc16ac62ea8`

---

## Critical Discoveries (Endpoint Surprises)

### 1. Synthetic Covariance — NOT JSON! It's STK Ephemeris format
The endpoint name says "synthetic-covariance" but the response body is an
**AGI/STK `.e` (DotE) ephemeris file** with inline covariance:

```
stk.v.12.0

BEGIN Ephemeris

# DotE file creation date: 9 Apr 2026 06:06:02.333 UTC

    NumberOfEphemerisPoints       5057

    ScenarioEpoch                 8 Apr 2026 00:44:30.451

# Satellite ID = 25544
# This is a SynCoPate ephemeris w/6x6 covariance info

# Epoch in JDate format: 2461138.53091 [UTC]
# Epoch in MJDate format: 61138.03091 [UTC]

    InterpolationMethod           Lagrange
    InterpolationSamplesM1        5
    CentralBody                   Earth
    CoordinateSystem              TEMEOfDate
    CovarianceInterpolationMethod TwoBodyQuadraticArithmeticBlending
    CovarianceFormat              LowerTriangular
    DistanceUnit                  Kilometers

    EphemerisTimePosVel

0.000000000000E+000 774.717061 5963.473044 3170.689005 -5.000237872 3.220150945 -4.824942174
...
```

- **Coordinate system:** TEME (perfect for ASTRA which uses TEME internally!)
- **Distance unit:** Kilometers (native ASTRA units)
- **Format:** `time_sec_from_scenario_epoch  x  y  z  vx  vy  vz`
- This IS the reference ephemeris — it's purely state vectors (no direct 6x6 matrix rows shown in the header, but covariance section follows the EphemerisTimePosVel block)
- **This is functionally equivalent to an OCM Reference Ephemeris**

### 2. Covariance-Plotly endpoint returns an HTML page
The `synthetic-covariance-plotly/{GUID}` endpoint returns a full HTML page with
an embedded Plotly graph — NOT raw JSON data. Only useful for browser display.

### 3. Reference Ephemerides return HTTP 500
Both `/api/entity/reference-ephemerides/{GUID}` and the OCM variant return
HTTP 500 with empty body. This appears to be a Spacebook server-side issue.
**Workaround:** The `synthetic-covariance/{GUID}` endpoint provides state vectors
in STK `.e` format — which is functionally equivalent for ASTRA's needs.

---

## Space Weather Format (CSSI / CelesTrak compatible)

```
DATATYPE CSSISpaceWeather
VERSION 1.3
UPDATED 2026 Apr 09 15:45:01 UTC
...
# FORMAT(I4,I3,I3,I5,I3,8I3,I4,8I4,I4,F4.1,I2,I4,F6.1,I2,5F6.1)
# yy mm dd BSRN ND Kp Kp Kp Kp Kp Kp Kp Kp Sum Ap Ap Ap Ap Ap Ap Ap Ap Avg Cp C9 ISN F10.7 Q Ctr81 Lst81 F10.7 Ctr81 Lst81
NUM_OBSERVED_POINTS 1925
BEGIN OBSERVED
2021 01 01 2556 10  0  3  7  3  3 13  7  7  43   0   2   3   2   2   5   3   3   2 0.0 0  24  77.7 0  80.4  83.5  80.4  82.9  85.4
...
BEGIN DAILY_PREDICTED
...
BEGIN MONTHLY_PREDICTED
...
END
```

**Parsing strategy:**
- Skip all `#` comment lines and header keyword lines
- Parse `BEGIN OBSERVED` / `BEGIN DAILY_PREDICTED` / `BEGIN MONTHLY_PREDICTED` blocks
- Columns (fixed-width, space-delimited):
  - Col 0: year (4 digits), Col 1: month, Col 2: day → date key
  - Col 24 (Adj F10.7): `fields[23]` — this is F10.7 ADJ
  - Col 26 (Obs F10.7): `fields[28]` by character position
  - Col 20 (Avg Ap): `fields[20]`
- **Identical format to CelesTrak SW.** Reuse existing `_parse_sw_csv` with minor adaptation

**Key difference from CelesTrak:** Spacebook SW is **fixed-width space-delimited text**,
not CSV. Existing `data_pipeline._parse_sw_csv()` parses CelesTrak's CSV format.
A separate `_parse_sw_text()` function is needed for fixed-width parsing.

---

## EOP Format (IERS Bulletin A compatible)

```
VERSION 1.1
UPDATED 2026 Apr 09 09:45:01 UTC
# FORMAT(I4,I3,I3,I6,2F10.6,2F11.7,4F10.6,I4)
#   Date    MJD      x         y       UT1-UTC      LOD       dPsi    dEpsilon     dX        dY    DAT
# (0h UTC)           "         "          s          s          "        "          "         "     s
NUM_OBSERVED_POINTS 1924
BEGIN OBSERVED
2021 01 01 59215  0.068684  0.304042 -0.1753654 -0.0005999 ...
```

**Parsing strategy:**
- Skip `#` lines and keyword headers
- `BEGIN OBSERVED` / `BEGIN PREDICTED` blocks — same pattern as SW
- Columns (space-delimited):
  - `year month day MJD xp yp UT1_UTC LOD dPsi dEps dX dY DAT`
  - `xp` = polar motion X (arcseconds) — col index 4
  - `yp` = polar motion Y (arcseconds) — col index 5
  - `UT1_UTC` = UT1-UTC (seconds) — col index 6
- **MJD key** = `int(fields[3])` → matches Julian Date via `MJD = JD - 2400000.5`

---

## Satellite Catalog JSON Schema (`/api/entity/satcat`)

```json
[
  {
    "id": "bf72c797-cee3-45b2-8de1-5bc16ac62ea8",   ← COMSPOC GUID
    "intlId": "1998-067A",                           ← International Designator
    "name": "ISS (ZARYA)",
    "noradId": 25544,                                ← NO leading zeros
    "launchDate": "1998-11-20T00:00:00",
    "objectType": "PAYLOAD",
    "owner": "CIS",
    "launchSite": "TTMTR",
    "period": 92.97,
    "inclination": 51.63,
    "apogee": 424,
    "perigee": 416,
    "rcsSize": "LARGE"
  }
]
```

**GUID field name:** `"id"` (not `"guid"` or `"comspocId"`)
**NORAD field name:** `"noradId"` (integer, no leading zeros)

---

## Satellite Catalog Details JSON Schema (`/api/entity/satcat/details`)

```json
[
  {
    "id": "bf72c797-cee3-45b2-8de1-5bc16ac62ea8",
    "name": "International Space Station",
    "noradId": 25544,
    "intlId": "1998-067A",
    "discosMsg": {
      "comspocDerivedMessage": {
        "adoptedMass": 450000.0,
        "adoptedSpan": 108.5,
        "ballisticCoefficient": 0.0291455,
        "hbrEncap": 66.0910168,
        "hbrHiMeters": 51.449757,
        "hbrLoMeters": 21.528102,
        "hbrMeanMeters": 36.48893,
        "hbrMedianMeters": 43.561822,
        "solarRadiationPressure": 0.0172224,
        "crossSectionMedian": 5961.587774,
        "shapeNumber": 30
      }
    },
    "jMcDowellMsg": {
      "mass": 20281,
      "dryMass": 19000,
      "length": 12.6,
      "diameter": 4.2,
      "span": 23.9,
      ...
    }
  }
]
```

**Key fields for ASTRA DragConfig:**
- `discosMsg.comspocDerivedMessage.adoptedMass` → `DragConfig.mass_kg`
- `discosMsg.comspocDerivedMessage.ballisticCoefficient` → derived `drag_area_m2`
- `discosMsg.comspocDerivedMessage.crossSectionMedian` → `DragConfig.area_m2`
- `discosMsg.comspocDerivedMessage.solarRadiationPressure` → `DragConfig.cr`

---

## Satellite Catalog CSV Schema (`/api/entity/satcat/csv`)

Column header (UTF-8 BOM prefix — strip with `lstrip('\ufeff')`):
```
OBJECT_NAME, OBJECT_ID, NORAD_CAT_ID, OBJECT_TYPE, OPS_STATUS_CODE,
OWNER, LAUNCH_DATE, LAUNCH_SITE, DECAY_DATE, PERIOD, INCLINATION,
APOGEE, PERIGEE, ORBIT_CENTER, ORBIT_TYPE, COMSPOC_ID, RCS_SIZE
```

**GUID column:** `COMSPOC_ID` (col index 15)
**NORAD column:** `NORAD_CAT_ID` (col index 2)

---

## STK Ephemeris Format (`/api/entity/synthetic-covariance/{GUID}`)

```
stk.v.12.0

BEGIN Ephemeris
    NumberOfEphemerisPoints       5057
    ScenarioEpoch                 8 Apr 2026 00:44:30.451
    CoordinateSystem              TEMEOfDate
    DistanceUnit                  Kilometers
    EphemerisTimePosVel

<time_sec>  <x_km>  <y_km>  <z_km>  <vx_km_s>  <vy_km_s>  <vz_km_s>
0.000000000000E+000  774.717061  5963.473044  3170.689005  -5.000237872  3.220150945  -4.824942174
...
END Ephemeris
```

**Parsing strategy for `parse_stk_ephemeris()`:**
1. Find `ScenarioEpoch` line → parse datetime → convert to JD
2. Find `EphemerisTimePosVel` marker line
3. Read all subsequent numeric lines until `END Ephemeris`
4. Each line: `t_offset_sec  x  y  z  vx  vy  vz`
5. Epoch JD = scenario_epoch_jd + t_offset_sec / 86400.0
6. Return `list[NumericalState]` in TEME frame (perfect — no frame conversion needed for ASTRA)

---

## Summary: Parser Requirements

| Data | Format | Parser to write |
|---|---|---|
| Space Weather | Fixed-width CSSI text | `_parse_sw_text()` in `spacebook.py` |
| EOP | Fixed-width IERS text | `_parse_eop_text()` in `spacebook.py` |
| Satcat | JSON array | `_load_satcat_guid_map()` in `spacebook.py` |
| TLE / XP-TLE | Standard 3-line TLE | Reuse `astra.tle.load_tle_catalog()` |
| STK Ephemeris | STK .e text | `parse_stk_ephemeris()` in `astra/ocm.py` (replaces OCM) |
| Satcat Details | JSON array | `fetch_satcat_details()` in `spacebook.py` |
