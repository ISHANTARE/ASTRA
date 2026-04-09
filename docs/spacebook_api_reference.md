# Spacebook API Reference — Section 4B (Complete Transcription)

> Source: https://spacebook.com/userguide#section4B  
> Transcribed from screenshots captured 2026-04-09.  
> All text reproduced verbatim. Download script examples are included exactly as shown.

---

## TLE:

```
https://spacebook.com/api/entity/tle
```

**Download Scripts**

**Windows PowerShell:**
```powershell
Invoke-WebRequest -Uri "https://spacebook.com/api/entity/tle" -OutFile "output.txt"
```

**Bash:**
```bash
curl -o output.txt https://spacebook.com/api/entity/tle
```

---

## Historical TLE:

```
https://spacebook.com/api/entity/tle/YYYY-MM-DD
```

**Download Scripts**

**Windows PowerShell:**
```powershell
Invoke-WebRequest -Uri "https://spacebook.com/api/entity/tle/2025-02-12" -OutFile "output.txt"
```

**Bash:**
```bash
curl -o output.txt https://spacebook.com/api/entity/tle/2025-02-12
```

---

## XP-TLE:

```
https://spacebook.com/api/entity/xp-tle
```

**Download Scripts**

**Windows PowerShell:**
```powershell
Invoke-WebRequest -Uri "https://spacebook.com/api/entity/xp-tle" -OutFile "output.txt"
```

**Bash:**
```bash
curl -o output.txt https://spacebook.com/api/entity/xp-tle
```

---

## Space Weather Full:

```
https://spacebook.com/api/spaceweather/full
```

**Download Scripts**

**Windows PowerShell:**
```powershell
Invoke-WebRequest -Uri "https://spacebook.com/api/spaceweather/full" -OutFile "output.txt"
```

**Bash:**
```bash
curl -o output.txt https://spacebook.com/api/spaceweather/full
```

---

## Space Weather Recent:

```
https://spacebook.com/api/spaceweather/recent
```

**Download Scripts**

**Windows PowerShell:**
```powershell
Invoke-WebRequest -Uri "https://spacebook.com/api/spaceweather/recent" -OutFile "output.txt"
```

**Bash:**
```bash
curl -o output.txt https://spacebook.com/api/spaceweather/recent
```

---

## EOP Full:

```
https://spacebook.com/api/eop/full
```

**Download Scripts**

**Windows PowerShell:**
```powershell
Invoke-WebRequest -Uri "https://spacebook.com/api/eop/full" -OutFile "output.txt"
```

**Bash:**
```bash
curl -o output.txt https://spacebook.com/api/eop/full
```

---

## EOP Recent:

```
https://spacebook.com/api/eop/recent
```

**Download Scripts**

**Windows PowerShell:**
```powershell
Invoke-WebRequest -Uri "https://spacebook.com/api/eop/recent" -OutFile "output.txt"
```

**Bash:**
```bash
curl -o output.txt https://spacebook.com/api/eop/recent
```

---

## Satellite Catalog SD file:

```
https://spacebook.com/api/entity/satcat/sd
```

**Download Scripts**

**Windows PowerShell:**
```powershell
Invoke-WebRequest -Uri "https://spacebook.com/api/entity/satcat/sd" -OutFile "output.txt"
```

**Bash:**
```bash
curl -o output.txt https://spacebook.com/api/entity/satcat/sd
```

---

## Satellite Catalog CSV file:

```
https://spacebook.com/api/entity/satcat/csv
```

**Download Scripts**

**Windows PowerShell:**
```powershell
Invoke-WebRequest -Uri "https://spacebook.com/api/entity/satcat/csv" -OutFile "output.txt"
```

**Bash:**
```bash
curl -o output.txt https://spacebook.com/api/entity/satcat/csv
```

---

## Satellite Catalog JSON file:

```
https://spacebook.com/api/entity/satcat
```

**Download Scripts**

**Windows PowerShell:**
```powershell
Invoke-WebRequest -Uri "https://spacebook.com/api/entity/satcat" -OutFile "output.txt"
```

**Bash:**
```bash
curl -o output.txt https://spacebook.com/api/entity/satcat
```

---

## Satellite Catalog Details JSON file:

```
https://spacebook.com/api/entity/satcat/details
```

**Download Scripts**

**Windows PowerShell:**
```powershell
Invoke-WebRequest -Uri "https://spacebook.com/api/entity/satcat/details" -OutFile "output.txt"
```

**Bash:**
```bash
curl -o output.txt https://spacebook.com/api/entity/satcat/details
```

---

## Satellite Catalog Details CSV file:

```
https://spacebook.com/api/entity/satcat/details/csv
```

**Download Scripts**

**Windows PowerShell:**
```powershell
Invoke-WebRequest -Uri "https://spacebook.com/api/entity/satcat/details/csv" -OutFile "output.txt"
```

**Bash:**
```bash
curl -o output.txt https://spacebook.com/api/entity/satcat/details/csv
```

---

## Object GUIDs

The API for the individual object data requires a GUID which is assigned to each object. The format for the endpoints is given below.

- **TLE-based ephemeris with synthetic covariance:**

```
https://spacebook.com/api/entity/synthetic-covariance/GUID
```

**Download Scripts**

**Windows PowerShell:**
```powershell
Invoke-WebRequest -Uri "https://spacebook.com/api/entity/synthetic-covariance/PUT_GUID_HERE" -OutFile "output.txt"
```

**Bash:**
```bash
curl -o output.txt https://spacebook.com/api/entity/synthetic-covariance/PUT_GUID_HERE
```

---

- **Synthetic covariance graph:**

```
https://spacebook.com/api/entity/synthetic-covariance-plotly/GUID
```

**Download Scripts**

**Windows PowerShell:**
```powershell
Invoke-WebRequest -Uri "https://spacebook.com/api/entity/synthetic-covariance-plotly/PUT_GUID_HERE" -OutFile "output.txt"
```

**Bash:**
```bash
curl -o output.txt https://spacebook.com/api/entity/synthetic-covariance-plotly/PUT_GUID_HERE
```

---

- **Reference ephemeris:**

```
https://spacebook.com/api/entity/reference-ephemerides/GUID
```

**Download Scripts**

**Windows PowerShell:**
```powershell
Invoke-WebRequest -Uri "https://spacebook.com/api/entity/reference-ephemerides/PUT_GUID_HERE" -OutFile "output.txt"
```

**Bash:**
```bash
curl -o output.txt https://spacebook.com/api/entity/reference-ephemerides/PUT_GUID_HERE
```

---

- **Reference ephemeris OCM format:**

```
https://spacebook.com/api/entity/reference-ephemerides/ocm/GUID
```

**Download Scripts**

**Windows PowerShell:**
```powershell
Invoke-WebRequest -Uri "https://spacebook.com/api/entity/reference-ephemerides/ocm/PUT_GUID_HERE" -OutFile "output.txt"
```

**Bash:**
```bash
curl -o output.txt https://spacebook.com/api/entity/reference-ephemerides/ocm/PUT_GUID_HERE
```

---

> You can find the unique GUID for each object in the Satellite Catalog JSON file, in the "id" field. Note that when you look up objects in this JSON file by NORAD id, i.e Space-Track id, the file does not use leading zeros for the "noradId" field. You can also find the GUID for an object in the Satellite Catalog csv file under the COMSPOC_ID column.
