```mermaid
flowchart TD

%% ═══════════════════════════════════════════════════════════════════
%%  ASTRA — COMPLETE FUNCTIONAL FLOW GRAPH
%%  Every node = a function or decision gate.
%%  Every edge = a data/control transfer.
%% ═══════════════════════════════════════════════════════════════════

%% ─────────────────────────────────────────────
%%  ENTRY POINT — PROCESS STARTUP
%% ─────────────────────────────────────────────
    START(["`**import astra**
    Process Entry`"])
    START --> BANNER["_show_banner()
    Print version + mode to stderr"]
    BANNER --> READENV["os.environ.get()
    Read ASTRA_STRICT_MODE
    ASTRA_SPACEBOOK_ENABLED
    ASTRA_DATA_DIR
    SPACETRACK_USER / PASS"]
    READENV --> MODECHECK{"`ASTRA_STRICT_MODE
    env var set?`"}
    MODECHECK -- "Yes → True" --> STRICT_ON["set_strict_mode(True)
    Acquire _STRICT_MODE_LOCK
    ASTRA_STRICT_MODE = True"]
    MODECHECK -- "No / False" --> STRICT_OFF["set_strict_mode(False)
    ASTRA_STRICT_MODE = False
    Warn: data will be estimated"]
    STRICT_ON --> SB_CHECK{"`ASTRA_SPACEBOOK_ENABLED
    == 'false' ?`"}
    STRICT_OFF --> SB_CHECK
    SB_CHECK -- "false" --> SB_DISABLED["SPACEBOOK_ENABLED = False
    All Spacebook calls → SpacebookError"]
    SB_CHECK -- "true (default)" --> SB_ENABLED["SPACEBOOK_ENABLED = True"]
    SB_DISABLED --> READY(["`**Engine Ready**
    Public API Exposed`"])
    SB_ENABLED --> READY

%% ─────────────────────────────────────────────
%%  USER REQUEST ROUTER
%% ─────────────────────────────────────────────
    READY --> USEROP{"`User Calls
    Which Operation?`"}
    USEROP --> OP_DATA["DATA INGESTION
    fetch_celestrak_*/
    fetch_spacetrack_*/
    fetch_*_sb()"]
    USEROP --> OP_PROP["ORBIT PROPAGATION
    propagate_orbit()
    propagate_many()
    propagate_cowell()"]
    USEROP --> OP_CONJ["CONJUNCTION ANALYSIS
    find_conjunctions()"]
    USEROP --> OP_FILT["DEBRIS FILTERING
    apply_filters()"]
    USEROP --> OP_VIS["VISIBILITY
    passes_over_location()"]
    USEROP --> OP_MAN["MANEUVER PLANNING
    validate_burn_sequence()
    propagate_cowell() + burns"]

%% ═══════════════════════════════════════════════════════════════════
%%  SECTION A — DATA INGESTION
%% ═══════════════════════════════════════════════════════════════════

%% ─────────────────────────────────────────────
%%  A1: DATA SOURCE SELECTOR
%% ─────────────────────────────────────────────
    OP_DATA --> DSRC{"`Which Data
    Source?`"}
    DSRC --> DS_CT["CelesTrak
    (No auth required)"]
    DSRC --> DS_ST["Space-Track.org
    (Authenticated)"]
    DSRC --> DS_SB["Spacebook / COMSPOC
    (Unauthenticated)"]

%% ─────────────────────────────────────────────
%%  A2: CELESTRAK BRANCH
%% ─────────────────────────────────────────────
    DS_CT --> CT_FMT{"`Format?`"}
    CT_FMT -- "format='tle'" --> CT_TLE["fetch_celestrak_group(group, 'tle')
    fetch_celestrak_active('tle')
    fetch_celestrak_comprehensive('tle')"]
    CT_FMT -- "format='json'" --> CT_OMM["fetch_celestrak_group_omm(group)
    fetch_celestrak_active_omm()
    fetch_celestrak_comprehensive_omm()"]
    CT_TLE & CT_OMM --> CT_RAW["_fetch_group_raw(group, fmt)
    Build URL: gp.php?GROUP=x&FORMAT=y
    _supplemental_params(group, fmt)"]
    CT_RAW --> CT_REQ["requests.get(url, headers=_HEADERS, timeout=20)
    User-Agent: ASTRA-Core/{version}"]
    CT_REQ --> CT_RESP{"`HTTP Response
    Check`"}
    CT_RESP -- "403 + rate-limit msg" --> CT_RATE["raise AstraError
    'CelesTrak rate limit reached'
    log_error()"]
    CT_RESP -- "5xx / empty body / invalid query" --> CT_SUPP{"`sup_params
    available?`"}
    CT_SUPP -- "No (group=active)" --> CT_FAIL["raise AstraError
    'gp.php failed, no supplemental'
    log_error()"]
    CT_SUPP -- "Yes" --> CT_SUPPREQ["_fetch_supplemental_raw(group, fmt, params)
    GET sup-gp.php?FILE=x&FORMAT=y
    timeout=20s"]
    CT_SUPPREQ --> CT_SUPPCHK{"`sup-gp response
    valid?`"}
    CT_SUPPCHK -- "Empty / invalid" --> CT_SUPFAIL["raise AstraError
    'sup-gp.php returned no data'
    log_error()"]
    CT_SUPPCHK -- "OK 200" --> CT_PARSE
    CT_RESP -- "200 OK" --> CT_PARSE["_parse_response(text, fmt)
    Route to TLE or JSON parser"]
    CT_PARSE --> CT_PARSEFMT{"`fmt?`"}
    CT_PARSEFMT -- "tle" --> CT_LOADTLE["load_tle_catalog(lines)
    Split text → lines
    parse_tle() per satellite
    validate_tle() per satellite"]
    CT_PARSEFMT -- "json" --> CT_LOADOMM["parse_omm_json(text)
    json.loads(text)
    parse_omm_record(rec) per entry
    validate_omm(rec)"]
    CT_LOADTLE --> CT_OUT["list[SatelliteTLE]
    → Physics Pipeline"]
    CT_LOADOMM --> CT_OUT2["list[SatelliteOMM]
    → Physics Pipeline"]

%% ─────────────────────────────────────────────
%%  A3: SPACE-TRACK BRANCH
%% ─────────────────────────────────────────────
    DS_ST --> ST_CREDS["_get_credentials()
    os.environ.get('SPACETRACK_USER')
    os.environ.get('SPACETRACK_PASS')"]
    ST_CREDS --> ST_CREDCHK{"`Credentials
    present?`"}
    ST_CREDCHK -- "No" --> ST_CREDFAIL["raise AstraError
    Print setup instructions
    (setx SPACETRACK_USER ...)
    log_error()"]
    ST_CREDCHK -- "Yes" --> ST_SESSION["_create_session(user, pass)
    Acquire _SESSION_LOCK
    Check _SESSION_CACHE[user]"]
    ST_SESSION --> ST_CACHED{"`Session
    Cached?`"}
    ST_CACHED -- "Yes" --> ST_REUSE["Return cached session
    (cookie still valid)"]
    ST_CACHED -- "No" --> ST_LOGIN["session.post(ST_LOGIN_URL)
    data={'identity': u, 'password': p}
    timeout=30s"]
    ST_LOGIN --> ST_LOGINCHK{"`Login
    Response`"}
    ST_LOGINCHK -- "401 / 'Failed' in text" --> ST_AUTHFAIL["raise AstraError
    'Authentication failed'
    log_error()"]
    ST_LOGINCHK -- "Non-2xx" --> ST_NETFAIL["raise AstraError
    'Service temporarily unavailable'
    log_error()"]
    ST_LOGINCHK -- "200 OK" --> ST_CACHESTORE["_SESSION_CACHE[username] = session
    log.info('ST session authenticated')"]
    ST_REUSE & ST_CACHESTORE --> ST_QUERY["_query_spacetrack(session, group, fmt)
    Map group → filter string (_ST_MAP)
    Build URL: gp?/OBJECT_NAME/~~X/FORMAT/json
    session.get(url, timeout=60)"]
    ST_QUERY --> ST_QUERYCHK{"`Response
    Check`"}
    ST_QUERYCHK -- "Network error" --> ST_QUERYFAIL["raise AstraError
    'Failed to fetch ST group'
    log_error()"]
    ST_QUERYCHK -- "Empty body" --> ST_EMPTYFAIL["raise AstraError
    'Empty response for group'
    log_error()"]
    ST_QUERYCHK -- "OK" --> ST_RATELIM["Check X-RateLimit-Remaining header
    if remaining < 10 → log.warning()"]
    ST_RATELIM --> ST_PAGEGUARD["Pagination Guard:
    record_count > 50000 → log.warning()
    'Results may be truncated'"]
    ST_PAGEGUARD --> ST_PARSESP["_parse_spacetrack_response(text, fmt)"]
    ST_PARSESP --> ST_PARSEFMT{"`fmt?`"}
    ST_PARSEFMT -- "json" --> ST_PARSEOMM["parse_omm_json(text)
    Normalize JSON → SatelliteOMM"]
    ST_PARSEFMT -- "tle" --> ST_PARSEJTLE["load_tle_catalog(lines)
    → SatelliteTLE list"]
    ST_PARSEOMM & ST_PARSEJTLE --> ST_OUT["list[SatelliteOMM] or list[SatelliteTLE]
    → Physics Pipeline"]

%% ─────────────────────────────────────────────
%%  A4: SPACEBOOK BRANCH
%% ─────────────────────────────────────────────
    DS_SB --> SB_ENABLEDCHK{"`SPACEBOOK_ENABLED?`"}
    SB_ENABLEDCHK -- "False" --> SB_DISABLEDEX["raise SpacebookError
    'Spacebook disabled (env var)'
    log_error()"]
    SB_ENABLEDCHK -- "True" --> SB_WHAT{"`Which Spacebook
    Endpoint?`"}

    SB_WHAT --> SB_TLE["fetch_tle_catalog()
    _fetch_tle_endpoint(url=_SB_TLE_URL,
      cache='tle_catalog.txt', ttl=6h)"]
    SB_WHAT --> SB_XPTLE["fetch_xp_tle_catalog()
    _fetch_tle_endpoint(url=_SB_XPTLE_URL,
      cache='xp_tle_catalog.txt', ttl=6h)"]
    SB_WHAT --> SB_HIST["fetch_historical_tle(date)
    _fetch_tle_endpoint(url=_SB_HISTORICAL_TLE/YYYY-MM-DD,
      cache=historical_tle_YYYY-MM-DD.txt, ttl=∞)"]
    SB_WHAT --> SB_SW["get_space_weather_sb(t_jd)"]
    SB_WHAT --> SB_EOP["get_eop_sb(t_jd)"]
    SB_WHAT --> SB_SCAT["fetch_satcat_details(norad_id)"]
    SB_WHAT --> SB_COV["fetch_synthetic_covariance_stk(norad_id)"]

    %% Common TLE endpoint fetch
    SB_TLE & SB_XPTLE & SB_HIST --> SB_CACHEAGE["_cache_age_hours(path)
    stat().st_mtime → age in hours"]
    SB_CACHEAGE --> SB_CACHEHIT{"`Cache fresh?
    (age < TTL)`"}
    SB_CACHEHIT -- "Yes" --> SB_CACHEREAD["path.read_text()
    Skip network call"]
    SB_CACHEHIT -- "No / missing" --> SB_SBGET["_sb_get(url, timeout=30)
    requests.get(url, headers, timeout)"]
    SB_SBGET --> SB_SBGETCHK{"`HTTP
    Response`"}
    SB_SBGETCHK -- "Timeout" --> SB_TOUT["raise SpacebookError
    'Timed out after 30s'
    log_error()"]
    SB_SBGETCHK -- "Network error" --> SB_NETERR["raise SpacebookError
    'Spacebook network error'
    log_error()"]
    SB_SBGETCHK -- "Non-200" --> SB_HTTPERR["raise SpacebookError
    'HTTP {status_code}'
    log_error()"]
    SB_SBGETCHK -- "200 OK" --> SB_WRITECACHE["path.write_text(resp.text)
    _sb_cache_path(filename)"]
    SB_CACHEREAD & SB_WRITECACHE --> SB_TLEPARSE["load_tle_catalog(text.splitlines())
    parse_tle() + validate_tle() per sat"]
    SB_TLEPARSE --> SB_TLEOUT["list[SatelliteTLE]
    source tag: 'spacebook_tle' /
    'spacebook_xptle' / 'spacebook_hist_{date}'"]

    %% Spacebook Space Weather
    SB_SW --> SBSW_LOADED{"`_sw_loaded?`"}
    SBSW_LOADED -- "No" --> SB_LOADSWFN["_load_sw(force_full=False)
    Check sw_recent.txt & sw_full.txt"]
    SBSW_LOADED -- "Yes" --> SBSW_STALE{"`Cache older
    than 6h?`"}
    SBSW_STALE -- "Yes" --> SBSW_BGREFRESH["threading.Thread(_refresh, daemon=True).start()
    Background refresh without blocking"]
    SBSW_STALE -- "No" --> SBSW_LOOKUP
    SB_LOADSWFN --> SBSW_FILESTALE{"`recent file
    age?`"}
    SBSW_FILESTALE -- "< 6h" --> SBSW_READCACHE["Read sw_recent.txt from disk"]
    SBSW_FILESTALE -- "> 6h" --> SBSW_DLRECENT["_sb_get(_SB_SW_RECENT_URL)
    Download recent space weather"]
    SBSW_DLRECENT --> SBSW_WRITERCENT["recent_path.write_text()"]
    SBSW_FILESTALE2{"`full file
    > 7 days?`"}
    SBSW_WRITERCENT --> SBSW_FILESTALE2
    SBSW_FILESTALE2 -- "Yes" --> SBSW_DLFULL["_sb_get(_SB_SW_FULL_URL)
    Download full sw history"]
    SBSW_FILESTALE2 -- "No" --> SBSW_PARSESW
    SBSW_DLFULL --> SBSW_WRITEFULL["full_path.write_text()"]
    SBSW_WRITEFULL --> SBSW_PARSESW
    SBSW_READCACHE --> SBSW_PARSESW
    SBSW_BGREFRESH --> SBSW_LOOKUP
    SBSW_PARSESW["_parse_sw_text(text)
    Split lines → BEGIN/END blocks
    Extract year/month/day/ap/f107_adj/f107_obs
    Build dict: 'YYYY-MM-DD' → (f107_obs, f107_adj, ap)
    Acquire _SW_LOCK → update _sw_cache
    _sw_loaded = True, _sw_last_success = now()"] --> SBSW_LOOKUP
    SBSW_LOOKUP["JD → date_str via timedelta from J2000
    Acquire _SW_LOCK
    _sw_cache.get(date_str)"] --> SBSW_HIT{"`Date in
    cache?`"}
    SBSW_HIT -- "Yes" --> SBSW_RETURN["return (f107_obs, f107_adj, ap_daily)"]
    SBSW_HIT -- "No" --> SBSW_MISS["raise SpacebookError
    'No SW data for {date}'"]

    %% Spacebook EOP
    SB_EOP --> SBEOP_LOADED{"`_eop_loaded?`"}
    SBEOP_LOADED -- "No" --> SB_LOADEOPFN["_load_eop()
    Check eop_recent.txt (24h TTL)"]
    SBEOP_LOADED -- "Yes" --> SBEOP_LOOKUP
    SB_LOADEOPFN --> SBEOP_RECAGE{"`recent EOP
    age < 24h?`"}
    SBEOP_RECAGE -- "Yes" --> SBEOP_READCACHE["Read eop_recent.txt"]
    SBEOP_RECAGE -- "No" --> SBEOP_DLRECENT["_sb_get(_SB_EOP_RECENT_URL)
    Download IERS EOP data"]
    SBEOP_DLRECENT --> SBEOP_WRITEREC["recent_path.write_text()"]
    SBEOP_READCACHE & SBEOP_WRITEREC --> SBEOP_FULLAGE{"`full EOP
    > 7 days?`"}
    SBEOP_FULLAGE -- "Yes" --> SBEOP_DLFULL["_sb_get(_SB_EOP_FULL_URL)
    Merge full + recent (recent wins)"]
    SBEOP_FULLAGE -- "No" --> SBEOP_PARSETEXT
    SBEOP_DLFULL --> SBEOP_PARSETEXT
    SBEOP_PARSETEXT["_parse_eop_text(text)
    Split BEGIN/END blocks
    Fields: MJD / xp / yp / dut1
    Build dict: MJD(int) → (xp, yp, dut1)
    Acquire _EOP_LOCK → update _eop_cache"] --> SBEOP_LOOKUP
    SBEOP_LOOKUP["JD → MJD = int(t_jd - 2400000.5)
    Acquire _EOP_LOCK
    Direct lookup _eop_cache[MJD]"] --> SBEOP_HIT{"`Exact MJD
    found?`"}
    SBEOP_HIT -- "Yes" --> SBEOP_RETURN["return (xp, yp, dut1)"]
    SBEOP_HIT -- "No" --> SBEOP_INTERP["Binary search bracketing MJDs
    Linear interpolation:
    frac = (MJD-lo)/(hi-lo)
    xp = xp0 + frac*(xp1-xp0)
    yp, dut1 same"]
    SBEOP_INTERP --> SBEOP_HIT2{"`Cache
    empty?`"}
    SBEOP_HIT2 -- "Yes" --> SBEOP_ZERO["log.warning()
    return (0.0, 0.0, 0.0)"]
    SBEOP_HIT2 -- "No" --> SBEOP_RETURN

    %% Spacebook GUID resolution for per-object endpoints
    SB_SCAT & SB_COV --> SB_GUID_RESOLVE["get_norad_guid(norad_id)
    Acquire _GUID_DOWNLOAD_LOCK
    Check _guid_loaded"]
    SB_GUID_RESOLVE --> SB_GUIDLOADED{"`_guid_loaded?`"}
    SB_GUIDLOADED -- "No" --> SB_LOADGUID["_load_satcat_guid_map()
    _sb_cache_path('satcat.json')
    age_h = _cache_age_hours()"]
    SB_GUIDLOADED -- "Yes" --> SB_GUIDLOOKUP
    SB_LOADGUID --> SB_GUIDAGE{"`cache age
    > 12h?`"}
    SB_GUIDAGE -- "Yes → need_download=True" --> SB_GUIDDL["_sb_get(_SB_SATCAT_JSON_URL, timeout=60)
    Download satcat.json"]
    SB_GUIDAGE -- "No → load from disk" --> SB_GUIDREAD["json.loads(cache_path.read_text())"]
    SB_GUIDDL --> SB_GUIDWDLCHK{"`Download
    succeeded?`"}
    SB_GUIDWDLCHK -- "No" --> SB_GUIDSTALE{"`Stale cache
    exists?`"}
    SB_GUIDSTALE -- "Yes" --> SB_GUIDREAD
    SB_GUIDSTALE -- "No" --> SB_GUIDFAIL["raise SpacebookError
    'Satcat download failed'"]
    SB_GUIDWDLCHK -- "Yes" --> SB_GUIDWRITE["cache_path.write_text(resp.text)
    json.loads(resp.text)"]
    SB_GUIDREAD & SB_GUIDWRITE --> SB_GUIDBUILD["Build _guid_map: noradId(int) → id(UUID)
    Acquire _GUID_LOCK
    _guid_loaded = True
    _guid_last_success = now()
    If stale → spawn _bg_refresh thread"]
    SB_GUIDBUILD --> SB_GUIDLOOKUP
    SB_GUIDLOOKUP["Acquire _GUID_LOCK
    _guid_map.get(int(norad_id))"] --> SB_GUIDHIT{"`GUID
    found?`"}
    SB_GUIDHIT -- "No" --> SB_GUIDNOTFOUND["raise SpacebookLookupError
    'NORAD ID not in Spacebook catalog'
    Suggest refresh_satcat_cache()"]
    SB_GUIDHIT -- "Yes" --> SB_GUIDOK["guid = UUID string"]
    SB_GUIDOK --> SB_COVFETCH["fetch_synthetic_covariance_stk(norad_id)
    url = _SB_SYNTH_COV_URL.format(guid=guid)
    _sb_get(url, timeout=30)
    Cache to 'synth_cov_{norad_id}.stk' (TTL=1h)"]
    SB_COVFETCH --> SB_COVPARSE["parse_stk_ephemeris(text)
    Scan for 'CovarianceTimePosVel' block
    Parse 22-field rows → 6×6 lower-triangular
    Return np.ndarray(6,6) or None"]

%% ═══════════════════════════════════════════════════════════════════
%%  SECTION B — TLE / OMM PARSING
%% ═══════════════════════════════════════════════════════════════════

    CT_LOADTLE & ST_PARSEJTLE & SB_TLEPARSE --> TLE_PARSE_DETAIL["parse_tle(line0, line1, line2)
    Extract: NORAD ID, epoch, bstar, inclination
    n, e, omega, M, RAAN, eccentricity
    validate_tle() — checksum, field range"]
    TLE_PARSE_DETAIL --> TLE_VAL{"`validate_tle()
    Passes?`"}
    TLE_VAL -- "Checksum mismatch / OOB field" --> TLE_INVALID["raise InvalidTLEError
    'Invalid TLE for NORAD {id}'
    log_error()"]
    TLE_VAL -- "OK" --> TLE_OBJ["SatelliteTLE(
      norad_id, name, line1, line2,
      epoch_jd, bstar, ...)"]

    CT_LOADOMM & ST_PARSEOMM --> OMM_PARSE_DETAIL["parse_omm_json(text) → per record
    parse_omm_record(rec: dict)
    validate_omm(rec) — required fields
    xptle_to_satellite_omm() if XP-TLE format"]
    OMM_PARSE_DETAIL --> OMM_VAL{"`validate_omm()
    Passes?`"}
    OMM_VAL -- "Missing required field" --> OMM_INVALID["raise AstraError / log.warning()
    Skip record in relaxed mode
    Raise in strict mode"]
    OMM_VAL -- "OK" --> OMM_OBJ["SatelliteOMM(
      norad_id, epoch_jd, mean_motion,
      eccentricity, inclination, raan, arg_perigee,
      mean_anomaly, bstar, mass_kg, rcs_m2,
      ballistic_coeff, object_type...)"]

%% ═══════════════════════════════════════════════════════════════════
%%  SECTION C — SPACE WEATHER PIPELINE
%% ═══════════════════════════════════════════════════════════════════

    OP_PROP --> SW_GATE["get_space_weather(t_jd)
    PRIORITY HIERARCHY:
    1. Spacebook (COMSPOC)
    2. CelesTrak SW-All.csv
    3. Synthetic fallback (150/150/15)"]
    SW_GATE --> SW_SBFIRST{"`SPACEBOOK_ENABLED?`"}
    SW_SBFIRST -- "True" --> SW_SBTRY["spacebook.get_space_weather_sb(t_jd)
    (see Spacebook SW branch above)"]
    SW_SBTRY --> SW_SBRESULT{"`Spacebook
    succeeded?`"}
    SW_SBRESULT -- "Yes" --> SW_OUT["(f107_obs, f107_adj, ap_daily)
    → atmospheric_density_empirical()"]
    SW_SBRESULT -- "Exception" --> SW_CTFALLBACK["log.warning('Spacebook SW failed, CelesTrak fallback')
    load_space_weather(data_dir)"]
    SW_SBFIRST -- "False" --> SW_CTFALLBACK
    SW_CTFALLBACK --> SW_CTLOADED{"`_sw_loaded?`"}
    SW_CTLOADED -- "No" --> SW_DLCT["_download_space_weather()
    Create requests.Session()
    Retry(total=3, backoff=1s, 429/5xx)
    GET celestrak.org/SpaceData/SW-All.csv
    timeout=30s"]
    SW_CTLOADED -- "Yes" --> SW_STALECT{"`_sw_last_success
    > 48h ago?`"}
    SW_STALECT -- "Yes" --> SW_BGREFRESH2["threading.Thread(_background_sw_fetch, daemon=True)
    Background refresh; serve stale cache"]
    SW_STALECT -- "No" --> SW_CTLOOKUP
    SW_DLCT --> SW_DLCTCHK{"`Download
    OK?`"}
    SW_DLCTCHK -- "Timeout/network error" --> SW_STRICTCT{"`STRICT_MODE?`"}
    SW_STRICTCT -- "True" --> SW_CTFAIL["raise ValueError
    '[ASTRA STRICT] SW fetch failed'"]
    SW_STRICTCT -- "False" --> SW_SYNTH["log.warning()
    return (150.0, 150.0, 15.0)
    'Synthetic defaults'"]
    SW_DLCTCHK -- "Proxy / HTML response (not 'DATE')" --> SW_PROXYCHECK{"STRICT_MODE?"}
    SW_PROXYCHECK -- "True" --> SW_PROXYFAIL["raise ValueError '[ASTRA STRICT] Invalid payload'"]
    SW_PROXYCHECK -- "False" --> SW_SYNTH
    SW_DLCTCHK -- "200 OK & valid CSV" --> SW_WRITESW["local_file.write_text(text)
    _parse_sw_csv(text)"]
    SW_WRITESW --> SW_PARSECT["_parse_sw_csv(text)
    csv.reader → iterate rows
    col[0]=DATE, col[24]=F10.7_OBS
    col[25]=F10.7_ADJ, col[20]=Ap_AVG
    Skip malformed rows (try/except)
    Acquire _SW_LOCK → _sw_cache.update()
    _sw_loaded = True"]
    SW_PARSECT --> SW_CTLOOKUP
    SW_BGREFRESH2 --> SW_CTLOOKUP
    SW_CTLOOKUP["jd_utc_to_datetime(t_jd)
    date_str = 'YYYY-MM-DD'
    Acquire _SW_LOCK
    _sw_cache.get(date_str)"] --> SW_CTHIT{"`Date in
    cache?`"}
    SW_CTHIT -- "Yes" --> SW_OUT
    SW_CTHIT -- "No" --> SW_NOCTSTRICT{"`STRICT_MODE?`"}
    SW_NOCTSTRICT -- "True" --> SW_NOERR["raise SpaceWeatherError
    '[ASTRA STRICT] No SW data for {date}'"]
    SW_NOCTSTRICT -- "False" --> SW_SYNTH

    SW_OUT --> ATMO["atmospheric_density_empirical(
      alt_km, f107_obs, f107_adj, ap_daily)
    NRLMSISE-00 model:
    T_inf = T_c + 3.24*f107_adj + 1.3*(obs-adj) + 28*Ap^0.4
    H_km = kB*T_inf / (m_eff * g_local) / 1000
    rho = rho_ref * exp(-(alt-400)/H)
    Clamp: rho > 1e-18 kg/m³
    if alt > 1500km or < 0 → return 0.0"]

%% ═══════════════════════════════════════════════════════════════════
%%  SECTION D — EPHEMERIS PIPELINE (JPL DE421)
%% ═══════════════════════════════════════════════════════════════════

    OP_PROP --> EPH_INIT["_ensure_skyfield(data_dir)
    Acquire _SKYFIELD_INIT_LOCK (RLock)
    Check _skyfield_ts / _skyfield_eph"]
    EPH_INIT --> EPH_LOADED{"`Already
    initialised?`"}
    EPH_LOADED -- "Yes" --> EPH_SKIP["Return singleton (fast path)"]
    EPH_LOADED -- "No" --> EPH_LOADER["_get_skyfield_loader(data_dir)
    Loader(~/.astra/data if ASTRA_DATA_DIR unset)"]
    EPH_LOADER --> EPH_TS["load.timescale()
    Downloads finals2000A.all (IERS EOP)
    Provides UT1-UTC, polar motion"]
    EPH_TS --> EPH_BSP["load('de421.bsp')
    Download DE421 if absent (~17 MB)
    Covers 1900-2050
    log.info('de421.bsp loaded')"]
    EPH_BSP --> EPH_READY["_skyfield_ts, _skyfield_eph initialized"]

    EPH_READY --> SUNPOS["sun_position_de(t_jd)
    jd_utc_to_datetime(t_jd)
    _skyfield_ts.utc(dt)
    earth.at(t).observe(sun)
    pos_au × 149597870.7 → km (GCRS)"]
    SUNPOS --> SUNTEME["sun_position_teme(t_jd)
    TEME.rotation_at(t).T  (R_teme_from_gcrs)
    return R @ pos_gcrs_km"]

    EPH_READY --> MOONPOS["moon_position_de(t_jd)
    earth.at(t).observe(moon)
    pos_au × 149597870.7 → km (GCRS)"]
    MOONPOS --> MOONTEME["moon_position_teme(t_jd)
    TEME.rotation_at(t).T @ pos_gcrs_km"]

    EPH_LOADED -- "DE421 unavailable" --> EPH_FALLBACK{"`STRICT_MODE?`"}
    EPH_FALLBACK -- "True" --> EPH_FAIL["raise EphemerisError
    '[ASTRA STRICT] DE421 unavailable'"]
    EPH_FALLBACK -- "False" --> EPH_APPROX["_sun_position_approx(t_jd)  [Meeus]
    _moon_position_approx(t_jd)  [Brown]
    log.warning('DE421 unavailable — low-fidelity fallback')"]

%% ═══════════════════════════════════════════════════════════════════
%%  SECTION E — ORBIT PROPAGATION
%% ═══════════════════════════════════════════════════════════════════

    OP_PROP --> PROPTYPE{"`Propagator
    Type?`"}
    PROPTYPE --> PROP_SGP4["SGP4 Analytical
    propagate_orbit()
    propagate_many()
    propagate_trajectory()"]
    PROPTYPE --> PROP_COWELL["Cowell Numerical
    propagate_cowell()
    RK8(7) adaptive step"]

%% ─────────────────────────────────────────────
%%  E1: SGP4 PROPAGATION
%% ─────────────────────────────────────────────
    PROP_SGP4 --> SGP4_INPUT{"`Input Type?`"}
    SGP4_INPUT -- "SatelliteTLE" --> SGP4_DIRECT["Use TLE line1/line2 directly
    sgp4.propagate(t)"]
    SGP4_INPUT -- "SatelliteOMM" --> SGP4_OMM2TLE["Convert OMM fields → SGP4 record
    mean_motion, eccentricity, inclination,
    raan, arg_perigee, mean_anomaly, bstar"]
    SGP4_DIRECT & SGP4_OMM2TLE --> SGP4_RUN["sgp4lib.propagate(t_epoch + dt)
    Returns (pos_km, vel_km_s) in TEME frame"]
    SGP4_RUN --> SGP4_OK{"`SGP4 ok?`"}
    SGP4_OK -- "Error code != 0 (reentry/decay)" --> SGP4_WARN["log.warning('SGP4 error code {e}')
    Insert NaN row
    Continue in relaxed mode
    Raise PropagationError in strict mode"]
    SGP4_OK -- "Success" --> SGP4_STATE["OrbitalState(
      t_jd, position_km, velocity_km_s,
      altitude_km = |r| - Re,
      frame='TEME')"]
    SGP4_STATE --> PROPOUT["list[OrbitalState]
    → downstream pipeline"]

%% ─────────────────────────────────────────────
%%  E2: COWELL NUMERICAL PROPAGATOR
%% ─────────────────────────────────────────────
    PROP_COWELL --> CW_INPUT["propagate_cowell(initial_state, t_span_jd,
      drag_config=DragConfig(...),
      burns=list[FiniteBurn],
      num_steps, use_numba)"]
    CW_INPUT --> CW_CHEB["Build Chebyshev ephemeris spans
    for Sun & Moon across t_span:
    _eval_cheb_3d_njit(t_norm, coeffs)
    Clenshaw recurrence (N=20 nodes)"]
    CW_CHEB --> CW_SWLOOKUP["_atmospheric_density(alt_km, t_jd)
    → get_space_weather(t_jd)
    → atmospheric_density_empirical()
    (NRLMSISE-00, see Section C)"]
    CW_SWLOOKUP --> CW_SEGLOOP["Segment Orchestrator:
    Slice t_span at burn ignition/cutoff boundaries
    for seg in segments:
      if seg.is_coast → _coast_derivative()
      if seg.is_powered → _powered_derivative()"]

    CW_SEGLOOP --> CW_COAST["_coast_derivative(t_sec, y, ...)
    6-DOF: y = [x,y,z,vx,vy,vz]
    dy/dt = [v, acceleration]"]
    CW_SEGLOOP --> CW_POWERED["_powered_derivative(t_sec, y, ...)
    7-DOF: y = [x,y,z,vx,vy,vz,mass]
    Thrust direction rotated from VNB/RTN frame
    _build_vnb_matrix_njit(r,v) or _build_rtn_matrix_njit(r,v)
    dm/dt = -F/(Isp*g0)"]

    CW_COAST & CW_POWERED --> CW_ACCEL["_acceleration(t_jd, r, v, ...)
    or _acceleration_njit() [@njit fastmath cache]"]

    CW_ACCEL --> CW_TWO["Two-body: a = -μ/r³ · r"]
    CW_ACCEL --> CW_J234["J2/J3/J4 Zonal Harmonics (WGS84)
    fJ2 = 1.5·J2·μ·Re²/r⁵
    fJ3 = 0.5·J3·μ·Re³/r⁷
    fJ4 = 0.625·J4·μ·Re⁴/r⁹
    Altitude-aware truncation (Numba path)"]
    CW_ACCEL --> CW_DRAG["Atmospheric Drag
    alt_km = |r| - Re
    rho = rho_ref * exp(-(alt-alt_ref)/H)
    v_rel = v - ω_earth × r
    a_drag = -0.5·ρ·1e3·(Cd·A/m)·|v_rel|·v_rel"]
    CW_ACCEL --> CW_3BODY["3rd Body Gravity
    _eval_cheb_3d_njit(t_norm, sun_coeffs) → sun_pos
    _eval_cheb_3d_njit(t_norm, moon_coeffs) → moon_pos
    a_sun = GM_sun*(d_sun/|d|³ - r_sun/|r_sun|³)
    a_moon similarly"]
    CW_ACCEL --> CW_SRP["Solar Radiation Pressure (Cannonball)
    d_ss = r - sun_pos
    scale = (AU/|d_ss|)²
    a_srp = ν·P0·scale·Cr·(A/m)/1000"]
    CW_SRP --> CW_SHADOW{"`Shadow model?`"}
    CW_SHADOW -- "Cylindrical (legacy)" --> CW_CYLAMBER["srp_cylindrical_illumination_factor_njit(r, r_sun)
    Night side + |⊥| < Re → ν=0 else ν=1"]
    CW_SHADOW -- "Conical (default PHY-D)" --> CW_CONAMBER["_srp_illumination_factor_njit(r, r_sun, Re, R_sun)
    alpha=asin(Re/r), beta=asin(R_sun/d)
    gamma=acos(-r·d_sun / (|r||d|))
    Penumbra: circle-circle intersection area
    ν = 1 - overlap/πβ²  ∈ [0,1]"]

    CW_TWO & CW_J234 & CW_DRAG & CW_3BODY & CW_CYLAMBER & CW_CONAMBER --> CW_ATOTAL["a_total = sum all perturbations
    return np.concatenate([v, a_total])
    or [v, a_total, dm_dt] for powered"]
    CW_ATOTAL --> CW_INTEGRATE["scipy.integrate.solve_ivp(
      method='DOP853' (RK8(7)),
      rtol=1e-9, atol=1e-12,
      dense_output=False)
    Adaptive step-size control"]
    CW_INTEGRATE --> CW_NUMBA{"`use_numba
    and numba installed?`"}
    CW_NUMBA -- "Yes" --> CW_JIT["@njit(fastmath=True, cache=True)
    _acceleration_njit() JIT-compiled path
    ~10-50× speedup vs pure Python"]
    CW_NUMBA -- "No" --> CW_PY["_acceleration() pure Python fallback
    log.warning('Numba unavailable')"]
    CW_JIT & CW_PY --> CW_STATES["NumericalState(
      t_jd, position_km[3],
      velocity_km_s[3], mass_kg)
    Output per timestep"]

%% ═══════════════════════════════════════════════════════════════════
%%  SECTION F — DEBRIS FILTERING
%% ═══════════════════════════════════════════════════════════════════

    OP_FILT --> FILT_INPUT["apply_filters(objects, config=FilterConfig(...))"]
    FILT_INPUT --> FILT_MAKE["make_debris_object(sat: SatelliteState)
    Wrap TLE or OMM → DebrisObject
    (norad_id, name, epoch_jd, source, rcs_m2, radius_m)"]
    FILT_MAKE --> FILT_SEQ["Sequential Filter Chain"]
    FILT_SEQ --> FILT_ALT["filter_altitude(objects, alt_min_km, alt_max_km)
    propagate_orbit(obj, t_now)
    Compute alt = |pos| - Re
    Keep if alt_min ≤ alt ≤ alt_max"]
    FILT_ALT --> FILT_REGION["filter_region(objects, lat_band, lon_band)
    Convert ECI → ECEF (teme_to_ecef)
    get_eop_sb() → (xp, yp, dut1)
    ECEF → geodetic (lat, lon, alt)
    Keep if lat ∈ lat_band AND lon ∈ lon_band"]
    FILT_REGION --> FILT_TIME["filter_time_window(objects, t_start, t_end)
    Check epoch_jd is within [t_start, t_end]
    or object is active at that time"]
    FILT_TIME --> FILT_STATS["catalog_statistics(objects)
    Count: total, LEO/MEO/GEO/HEO,
    active vs debris vs payload
    rcs_hist, alt_distribution"]
    FILT_STATS --> FILT_OUT["list[DebrisObject] filtered
    → Conjunction or Propagation pipeline"]

%% ═══════════════════════════════════════════════════════════════════
%%  SECTION G — SPATIAL INDEX (cKDTree)
%% ═══════════════════════════════════════════════════════════════════

    FILT_OUT --> SPATIAL["SpatialIndex()
    Build KD-tree over object positions"]
    SPATIAL --> SPATIAL_BUILD["idx.rebuild_for_trajectories(trajectories)
    For each object: compute AABB (min/max position envelope)
    scipy.spatial.cKDTree(positions)
    SE-C: use entire propagation window AABB"]
    SPATIAL_BUILD --> SPATIAL_QUERY["idx.query_pairs(threshold_km=coarse_threshold_km)
    tree.query_pairs(r=threshold_km)
    Return set of (A,B) candidate pairs"]

%% ═══════════════════════════════════════════════════════════════════
%%  SECTION H — CONJUNCTION ANALYSIS (3-phase)
%% ═══════════════════════════════════════════════════════════════════

    OP_CONJ --> CONJ_INPUT["find_conjunctions(
      trajectories, times_jd,
      elements_map, threshold_km=5,
      coarse_threshold_km=50,
      cov_map, vel_map)"]
    CONJ_INPUT --> CONJ_NANCHECK["Validate trajectories:
    np.isfinite(traj) for each object
    Collect nan_ids"]
    CONJ_NANCHECK --> CONJ_NANSTRIC{"`STRICT_MODE
    and NaNs exist?`"}
    CONJ_NANSTRIC -- "True" --> CONJ_NANFAIL["raise PropagationError
    '{n} satellites have invalid trajectories'"]
    CONJ_NANSTRIC -- "False" --> CONJ_NANWARN["log.warning('N satellites excluded (NaN)')
    valid_trajectories = filter(isfinite)"]
    CONJ_NANWARN --> CONJ_PHASE1["Phase 1+2: SpatialIndex screening
    SpatialIndex.rebuild_for_trajectories()
    idx.query_pairs(coarse_threshold_km)
    → candidate_pairs_phase2 (set of tuples)"]
    CONJ_PHASE1 --> CONJ_PHASE1CHK{"`candidate_pairs
    > 0?`"}
    CONJ_PHASE1CHK -- "None" --> CONJ_EMPTY["log.info('0 candidate pairs')
    return []"]
    CONJ_PHASE1CHK -- "Yes" --> CONJ_PHASE3["Phase 3: Exact Curvilinear TCA
    ThreadPoolExecutor(max_workers=cpu_count())
    submit evaluate_pair(A, B) for each pair"]
    CONJ_PHASE3 --> CONJ_PAIR["evaluate_pair(A, B):
    1. coarse_dists = distance_3d(traj_A, traj_B)
    2. t_idx = argmin(coarse_dists)
    3. if coarse_min > coarse_threshold → return None"]
    CONJ_PAIR --> CONJ_SPLINE["Build CubicSpline(times_jd, traj, bc='natural')
    spline_A, spline_B
    vel_spline_A, vel_spline_B (from vel_map if available)"]
    CONJ_SPLINE --> CONJ_DENSE["Dense 1-second scan:
    bracket_width = 1 or 2 (edge detection)
    t_dense = linspace(t_lo, t_hi, seconds_in_bracket)
    rA_dense = spline_A(t_dense)
    rB_dense = spline_B(t_dense)
    tca_dense_idx = argmin(distance_3d(rA, rB))"]
    CONJ_DENSE --> CONJ_FILTER{"`min_dist
    > threshold_km?`"}
    CONJ_FILTER -- "Yes" --> CONJ_SKIP["return None (not a conjunction)"]
    CONJ_FILTER -- "No" --> CONJ_TCA["tca_jd = t_dense[tca_dense_idx]
    pos_A, pos_B at TCA
    vel_A = vel_spline(tca) OR spline'(tca)/86400"]
    CONJ_TCA --> CONJ_COVPATH["Covariance Resolution (per object):
    Priority: cov_map → Spacebook → estimate_covariance → None"]
    CONJ_COVPATH --> CONJ_COVMAP{"`cov_map
    supplied?`"}
    CONJ_COVMAP -- "Yes" --> CONJ_USECDM["cov = cov_map[id]
    src = 'CDM'"]
    CONJ_COVMAP -- "No" --> CONJ_SBCOV["load_spacebook_covariance(norad_id)
    fetch_synthetic_covariance_stk()
    parse_stk_ephemeris() → 6×6 matrix"]
    CONJ_SBCOV --> CONJ_SBCOVOK{"`Spacebook
    cov OK?`"}
    CONJ_SBCOVOK -- "Yes" --> CONJ_SBCOVUSE["cov = sb_matrix
    src = 'COMSPOC_SYNTHETIC'"]
    CONJ_SBCOVOK -- "None / fail" --> CONJ_ESTCOV["estimate_covariance(days_since_epoch)
    Heuristic diagonal RTN covariance
    rotate_covariance_rtn_to_eci(cov_rtn, pos, vel)
    RTN → ECI rotation matrix Q
    cov_eci = Q @ cov_rtn @ Q.T
    src = 'SYNTHETIC'"]
    CONJ_USECDM & CONJ_SBCOVUSE & CONJ_ESTCOV --> CONJ_COVSHAPE{"`cov_A.shape
    == cov_B.shape?`"}
    CONJ_COVSHAPE -- "No: 6×6 vs 3×3" --> CONJ_COVSLICE["Slice 6×6 → [:3,:3]
    to match dimensionality"]
    CONJ_COVSHAPE -- "Yes" --> CONJ_PC
    CONJ_COVSLICE --> CONJ_PC["compute_collision_probability(
      miss_vector, rel_vel, cov_A, cov_B,
      radius_a_km, radius_b_km)
    Foster short-encounter 2D B-plane integral"]
    CONJ_PC --> CONJ_RADII["_dynamic_radius_km(obj, rel_vel_hat, pos, vel)
    Priority:
    1. TLE dimensions_m + attitude (TUMBLING/NADIR/INERTIAL)
       projected_area_m2(dims, quat, v_hat) → sqrt(A/π)/1000
    2. OMM rcs_m2 → sqrt(RCS/π)/1000
    3. DebrisObject.radius_m / 1000
    4. Fallback: 5m sphere"]
    CONJ_RADII --> CONJ_RISKCLASS["_classify_risk(P_c):
    P_c > 1e-4 → CRITICAL
    P_c > 1e-5 → HIGH
    P_c > 1e-6 → MEDIUM
    else → LOW, None → UNKNOWN"]
    CONJ_RISKCLASS --> CONJ_COVSRC["Determine covariance_source:
    CDM / COMSPOC_SYNTHETIC / SYNTHETIC / UNAVAILABLE
    MIXED(A+B) if sources differ"]
    CONJ_COVSRC --> CONJ_EVENT["ConjunctionEvent(
      object_a_id, object_b_id,
      tca_jd, miss_distance_km,
      relative_velocity_km_s,
      collision_probability = P_c,
      risk_level, position_a_km, position_b_km,
      covariance_source)"]
    CONJ_EVENT --> CONJ_FUTURES["as_completed(futures) aggregator
    events.append(result)
    if STRICT_MODE and exception → raise AstraError
    else log.warning + skipped++"]
    CONJ_FUTURES --> CONJ_SORT["events.sort(key=lambda x: x.miss_distance_km)
    log.info('N events detected')
    return list[ConjunctionEvent]"]
    CONJ_NANSTRIC & CONJ_PHASE1CHK --> CONJ_SORT

%% ═══════════════════════════════════════════════════════════════════
%%  SECTION I — COVARIANCE & COLLISION PROBABILITY
%% ═══════════════════════════════════════════════════════════════════

    CONJ_PC --> COV_FOSTER["compute_collision_probability(
      miss_vector, rel_vel, cov_A, cov_B, ra, rb)
    Foster (2002) short-encounter integral:
    Combined covariance Σ = cov_A + cov_B
    Project to encounter B-plane (⊥ rel_vel)
    HBR = ra + rb (hard-body radius)
    Integrate bivariate normal over HBR disk
    → P_c ∈ [0, 1]"]
    CONJ_PC --> COV_MC["compute_collision_probability_mc(
      miss_vec, rel_vel, cov_A, cov_B, ra, rb,
      n_samples=100000)
    MC sampling from N(miss_vec, Σ)
    count hits within (ra+rb)
    P_c_mc = hits / n_samples"]
    CONJ_PC --> COV_STM["propagate_covariance_stm(cov0, Phi)
    Φ(t,t0) = state-transition matrix (6×6)
    cov(t) = Φ @ cov0 @ Φ.T
    Propagate uncertainty forward in time"]

%% ═══════════════════════════════════════════════════════════════════
%%  SECTION J — MANEUVER PLANNING
%% ═══════════════════════════════════════════════════════════════════

    OP_MAN --> MAN_VALIDATE["validate_burn_sequence(burns: list[FiniteBurn])
    Check: t_start < t_end for each burn
    No overlapping burns
    thrust_N > 0, isp_s > 0, mass_kg > 0"]
    MAN_VALIDATE --> MAN_VCHK{"`Validation
    passes?`"}
    MAN_VCHK -- "No" --> MAN_FAIL["raise ManeuverError
    'Invalid burn sequence'"]
    MAN_VCHK -- "Yes" --> MAN_FRAME{"`Burn frame?`"}
    MAN_FRAME -- "VNB" --> MAN_VNB["rotation_vnb_to_inertial(r, v)
    _build_vnb_matrix_njit(r, v)
    V = v̂, N = v×r / |v×r|, B = V×N"]
    MAN_FRAME -- "RTN" --> MAN_RTN["rotation_rtn_to_inertial(r, v)
    _build_rtn_matrix_njit(r, v)
    R = r̂, T = R×N̂, N = r×v / |r×v|"]
    MAN_VNB & MAN_RTN --> MAN_ACCEL["frame_to_inertial(direction, r, v, frame)
    thrust_acceleration_inertial(
      burn.direction_vnb, r, v,
      burn.thrust_N, mass_kg)
    a_thrust = R_frame @ dir × (F/m)"]
    MAN_ACCEL --> MAN_COWELL["propagate_cowell(initial_state, t_span,
      drag_config, burns=[...])
    7-DOF integration during burn arcs
    6-DOF during coast arcs
    (See Section E2)"]
    MAN_COWELL --> MAN_OUT["list[NumericalState]
    Position/velocity/mass per timestep
    → Mission Analysis / ΔV budget"]

%% ═══════════════════════════════════════════════════════════════════
%%  SECTION K — VISIBILITY & GROUND PASSES
%% ═══════════════════════════════════════════════════════════════════

    OP_VIS --> VIS_INPUT["passes_over_location(sat, observer, t_start, t_end)
    OR visible_from_location(sats, observer, t_jd)"]
    VIS_INPUT --> VIS_PROP["propagate_orbit(sat, times_jd)
    SGP4 over entire time span
    → list[OrbitalState]"]
    VIS_PROP --> VIS_ECEF["teme_to_ecef(pos_teme, t_jd)
    get_eop_sb(t_jd) → (xp, yp, dut1)
    GAST = greenwich_apparent_sidereal_time()
    R3(-GAST) @ (I + polar_correction) @ pos_teme"]
    VIS_ECEF --> VIS_GEODETIC["ecef_to_geodetic(x, y, z)
    Bowring iterative algorithm
    → (lat_rad, lon_rad, alt_km)"]
    VIS_GEODETIC --> VIS_TOPO["Compute topocentric vector:
    obs_ecef = geodetic_to_ecef(lat, lon, alt)
    delta = sat_ecef - obs_ecef
    Rotate to SEZ (South-East-Zenith)
    elevation = asin(SEZ_z / |delta|)
    azimuth = atan2(SEZ_e, -SEZ_s)"]
    VIS_TOPO --> VIS_ELEVCHK{"`elevation ≥
    min_elevation?`"}
    VIS_ELEVCHK -- "No" --> VIS_BELOW["Not visible at this timestep"]
    VIS_ELEVCHK -- "Yes" --> VIS_COLLECT["Collect pass events:
    PassEvent(
      rise_jd, set_jd, max_elevation_deg,
      max_azimuth_deg, duration_s)"]
    VIS_COLLECT --> VIS_OUT["list[PassEvent] sorted by rise_jd
    → Ground Station Scheduling"]

%% ═══════════════════════════════════════════════════════════════════
%%  SECTION L — COORDINATE FRAME TRANSFORMS
%% ═══════════════════════════════════════════════════════════════════

    CW_STATES & SGP4_STATE --> FRAME_TEME["Output in TEME (True Equator Mean Equinox)
    Epoch-of-date frame used by SGP4/Cowell"]
    FRAME_TEME --> FRAME_WHAT{"`Target frame?`"}
    FRAME_WHAT --> FRAME_ECEF["teme_to_ecef(pos_teme, t_jd)
    Primary: get_eop_sb() for live EOP
    Fallback: get_ut1_utc_correction() via Skyfield IERS
    GAST = GMST + equation_of_equinoxes
    W = polar_wobble_matrix(xp, yp)
    pos_ecef = R3(-GAST) @ W @ pos_teme"]
    FRAME_WHAT --> FRAME_ECI["TEME → ECI (J2000/GCRS)
    Apply precession + nutation correction
    (used for covariance projections)"]
    FRAME_ECEF --> FRAME_GEODETIC["ecef_to_geodetic(x, y, z)
    WGS84 Bowring iteration
    → (lat°, lon°, alt_km)
    Used by: ground_track(), filter_region(), visibility"]

%% ═══════════════════════════════════════════════════════════════════
%%  SECTION M — ERROR HANDLING HIERARCHY
%% ═══════════════════════════════════════════════════════════════════

    CT_RATE & CT_FAIL & CT_SUPFAIL --> ERR_ASTRA["AstraError (base)"]
    ST_CREDFAIL & ST_AUTHFAIL & ST_NETFAIL --> ERR_ASTRA
    ST_QUERYFAIL & ST_EMPTYFAIL --> ERR_ASTRA
    TLE_INVALID --> ERR_TLEVAL["InvalidTLEError(AstraError)"]
    SGP4_WARN --> ERR_PROP["PropagationError(AstraError)"]
    EPH_FAIL --> ERR_EPH["EphemerisError(AstraError)"]
    SW_CTFAIL & SW_NOERR --> ERR_SW["SpaceWeatherError(AstraError)"]
    SB_DISABLEDEX & SB_TOUT & SB_NETERR & SB_HTTPERR --> ERR_SB["SpacebookError(AstraError)"]
    SB_GUIDNOTFOUND --> ERR_SBLKUP["SpacebookLookupError(SpacebookError)"]
    MAN_FAIL --> ERR_MAN["ManeuverError(AstraError)"]
    CONJ_NANFAIL --> ERR_PROP
    FILT_STATS --> ERR_FILT["FilterError(AstraError)"]

    ERR_ASTRA & ERR_TLEVAL & ERR_PROP & ERR_EPH & ERR_SW & ERR_SB & ERR_SBLKUP & ERR_MAN & ERR_FILT --> ERR_LOG["get_logger(__name__).error/warning()
    structlog structured fields:
    timestamp, module, function, norad_id, message
    → stderr / log file"]

%% ═══════════════════════════════════════════════════════════════════
%%  SECTION N — CACHE MANAGEMENT SUMMARY
%% ═══════════════════════════════════════════════════════════════════

    ATMO & SW_OUT --> CACHE_SW["_sw_cache: dict[date_str → (f107_obs, f107_adj, ap)]
    TTL: 24h (CelesTrak), 6h (Spacebook recent)
    _SW_LOCK (RLock), _SW_DOWNLOAD_LOCK (Lock)
    Background refresh thread (daemon=True)"]
    SBEOP_RETURN --> CACHE_EOP["_eop_cache: dict[MJD → (xp, yp, dut1)]
    TTL: 24h (recent), 7 days (full)
    _EOP_LOCK (RLock)"]
    SB_GUIDBUILD --> CACHE_GUID["_guid_map: dict[norad_id(int) → UUID str]
    TTL: 12h, background refresh on stale
    _GUID_LOCK (RLock), _GUID_DOWNLOAD_LOCK (Lock)"]
    EPH_READY --> CACHE_EPH["_skyfield_loader, _skyfield_ts, _skyfield_eph
    Singleton — initialised once per process
    _SKYFIELD_INIT_LOCK (RLock)
    de421.bsp: ~17 MB local cache"]
    ST_CACHESTORE --> CACHE_ST["_SESSION_CACHE: dict[username → requests.Session]
    Cookie-based session reuse
    _SESSION_LOCK (threading.Lock)
    spacetrack_logout() clears cache"]

%% ═══════════════════════════════════════════════════════════════════
%%  SECTION O — OUTPUT SUMMARY
%% ═══════════════════════════════════════════════════════════════════

    CONJ_SORT --> OUTPUTS(["**OUTPUTS**"])
    PROPOUT --> OUTPUTS
    VIS_OUT --> OUTPUTS
    MAN_OUT --> OUTPUTS
    FILT_OUT --> OUTPUTS
    CT_OUT & CT_OUT2 & ST_OUT & SB_TLEOUT --> OUTPUTS
    OUTPUTS --> OUT_API["All results are
    typed Python objects:
    list[SatelliteTLE/OMM]
    list[OrbitalState]
    list[ConjunctionEvent]
    list[DebrisObject]
    list[PassEvent]
    list[NumericalState]
    NumericalState / ConjunctionDataMessage
    All serialisable via dataclasses.asdict()"]

%% ═══════════════════════════════════════════════════════════════════
%%  STYLES
%% ═══════════════════════════════════════════════════════════════════

    classDef decision fill:#1a1a2e,stroke:#e94560,color:#fff,font-size:11px
    classDef func fill:#16213e,stroke:#0f3460,color:#a8dadc,font-size:10px
    classDef source fill:#0f3460,stroke:#533483,color:#e2e2e2,font-size:11px
    classDef output fill:#533483,stroke:#e94560,color:#fff,font-size:11px
    classDef error fill:#e94560,stroke:#c0392b,color:#fff,font-size:10px
    classDef cache fill:#2d6a4f,stroke:#52b788,color:#d8f3dc,font-size:10px
    classDef entry fill:#e94560,stroke:#fff,color:#fff,font-size:13px,font-weight:bold

    class MODECHECK,SB_CHECK,USEROP,DSRC,CT_FMT,CT_RESP,CT_SUPP,CT_SUPPCHK,CT_PARSEFMT,ST_CREDCHK,ST_CACHED,ST_LOGINCHK,ST_QUERYCHK,ST_PARSEFMT,SB_ENABLEDCHK,SB_WHAT,SB_GUIDLOADED,SB_GUIDAGE,SB_GUIDWDLCHK,SB_GUIDSTALE,SB_GUIDHIT,SBSW_LOADED,SBSW_STALE,SBSW_FILESTALE,SBSW_FILESTALE2,SBSW_HIT,SBEOP_LOADED,SBEOP_RECAGE,SBEOP_FULLAGE,SBEOP_HIT,SBEOP_HIT2,SW_SBFIRST,SW_SBRESULT,SW_CTLOADED,SW_STALECT,SW_DLCTCHK,SW_STRICTCT,SW_PROXYCHECK,SW_CTHIT,SW_NOCTSTRICT,EPH_LOADED,EPH_FALLBACK,PROPTYPE,SGP4_INPUT,SGP4_OK,CW_NUMBA,CW_SHADOW,FILT_STATS,CONJ_NANSTRIC,CONJ_PHASE1CHK,CONJ_FILTER,CONJ_COVMAP,CONJ_SBCOVOK,CONJ_COVSHAPE,MAN_VCHK,MAN_FRAME,VIS_ELEVCHK,FRAME_WHAT,STRICT_ON,STRICT_OFF decision
    class START,READY,OUTPUTS entry
    class CT_OUT,CT_OUT2,ST_OUT,SB_TLEOUT,PROPOUT,CW_STATES,VIS_OUT,MAN_OUT,FILT_OUT,OUT_API output
    class CT_RATE,CT_FAIL,CT_SUPFAIL,ST_CREDFAIL,ST_AUTHFAIL,ST_NETFAIL,ST_QUERYFAIL,ST_EMPTYFAIL,TLE_INVALID,SGP4_WARN,EPH_FAIL,SW_CTFAIL,SW_NOERR,SB_DISABLEDEX,SB_TOUT,SB_NETERR,SB_HTTPERR,SB_GUIDNOTFOUND,MAN_FAIL,CONJ_NANFAIL,ERR_ASTRA,ERR_TLEVAL,ERR_PROP,ERR_EPH,ERR_SW,ERR_SB,ERR_SBLKUP,ERR_MAN,ERR_FILT,ERR_LOG error
    class CACHE_SW,CACHE_EOP,CACHE_GUID,CACHE_EPH,CACHE_ST cache
    class DS_CT,DS_ST,DS_SB source
```
