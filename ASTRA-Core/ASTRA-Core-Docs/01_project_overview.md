# 01 — ASTRA Core: Project Overview

---

## 1. Project Identity

| Property | Value |
|---|---|
| **Library Name** | `astra` (importable Python package) |
| **Full Name** | ASTRA Core |
| **Version** | 0.1.0 |
| **Language** | Python 3.10+ |
| **Purpose** | Orbital analysis engine for space debris workflows |
| **Target Domain** | Space situational awareness (SSA), debris tracking, collision risk assessment |

---

## 2. What Is ASTRA Core?

ASTRA Core is a **high-level Python library** that provides structured, optimized computation workflows for analyzing large catalogs of orbital objects (satellites, debris, rocket bodies).

It is **NOT** an implementation of orbital physics. It does **NOT** implement SGP4 equations, gravity models, or numerical integrators. Instead, it:

- **Orchestrates** existing physics libraries (`sgp4`, `skyfield`)
- **Filters** large datasets efficiently before expensive computation
- **Reduces** computational load through staged pipelines
- **Exposes** clean, composable, testable functions

Think of ASTRA Core as the **decision engine** that determines:
- Which objects to compute
- Over which time windows
- In what order
- With what level of precision

---

## 3. Why ASTRA Core Exists

### The Problem

Space debris catalogs routinely contain **10,000–50,000+ objects**. Naive approaches to orbital analysis such as computing full trajectories for every object before any filtering, or performing O(n²) pairwise comparisons across an entire catalog, are computationally infeasible.

### The Solution

ASTRA Core provides a **multi-stage computation reduction framework**. Rather than:

```
catalog (50,000 objects)
  → propagate all objects
  → compare all pairs
  → find interesting events
```

ASTRA Core does:

```
catalog (50,000 objects)
  → altitude filter (→ ~5,000 objects)
  → region filter  (→ ~1,000 objects)
  → time-window filter (→ ~200 objects)
  → propagate only survivors
  → proximity pre-filter (→ ~50 candidate pairs)
  → fine-grained conjunction analysis
```

This is the core value proposition of the library.

---

## 4. Relationship to the Full ASTRA System

ASTRA Core is the **bottom layer** of a larger system:

```
┌─────────────────────────────────┐
│   Frontend UI (React/Three.js)  │  ← Future
├─────────────────────────────────┤
│   Backend API (FastAPI/Flask)   │  ← Future
├─────────────────────────────────┤
│   ASTRA Core Library  ◄ THIS   │
├─────────────────────────────────┤
│ sgp4 │ skyfield │ numpy         │  ← External dependencies
└─────────────────────────────────┘
```

**Strict Rules:**
- ASTRA Core has **zero knowledge** of HTTP, databases, or frontend logic
- ASTRA Core only receives data structures and returns data structures
- ASTRA Core is importable as a standalone Python package with no web dependencies

---

## 5. Primary Use Cases

| Use Case | Description |
|---|---|
| **Debris Catalog Analysis** | Filter and analyze tens of thousands of TLE objects by altitude, region, and time |
| **Conjunction Detection** | Find close-approach events between orbital objects efficiently |
| **Ground Station Visibility** | Compute when satellites pass over observer locations |
| **Orbit Propagation** | Generate position/velocity trajectories for defined time windows |
| **Catalog Statistics** | Compute density, distribution, and population statistics across orbital regimes |

---

## 6. Intended Users

| User | How They Use ASTRA Core |
|---|---|
| **Backend Developer** | Imports `astra` modules to power API endpoints |
| **Research Scientist** | Uses the library directly for orbital analysis scripts |
| **AI Coding Agent** | Implements features guided by this documentation |
| **Test Engineer** | Validates orbital calculations against known reference data |

---

## 7. Scope Boundaries

### Included in ASTRA Core
- TLE parsing and validation
- Orbit propagation (via `sgp4`/`skyfield`)
- Multi-stage debris filtering
- Conjunction detection (optimized)
- Ground station visibility calculation
- Time conversion utilities
- Orbital element computation
- Catalog statistics

### Explicitly Excluded
- HTTP servers, REST APIs, or web frameworks
- Database connections or ORM models
- File upload / file server logic
- Authentication or session management
- Frontend rendering or JavaScript
- Raw socket communication
- Any I/O inside computational functions (no file reads/writes inside logic)

---

## 8. Design Philosophy Summary

| Principle | Implication |
|---|---|
| Pure functions | Every function: input data → output data, no side effects |
| Deterministic | Same inputs always produce same outputs |
| No global state | No module-level mutable variables |
| Strict module separation | Each module owns exactly one responsibility domain |
| Vector-first | All bulk operations use NumPy arrays, not Python loops |
| Lazy computation | Only compute what filtering survivors require |
| Testable | Every function is independently testable with mocked inputs |

---

## 9. Key Metrics and Parameters

These parameters govern all simulation and analysis work in ASTRA Core:

| Parameter | Value |
|---|---|
| **Prediction Window** | 24 hours |
| **Time Resolution** | 5 minutes |
| **Total Simulation Steps** | 288 steps (24h × 60min / 5min) |
| **Collision Risk Threshold** | < 5 km proximity = conjunction candidate |
| **Minimum Catalog Objects** | 10,000 (scale target) |
| **Maximum Catalog Objects** | 50,000 (scale target) |
| **Coordinate Frame (SGP4 output)** | TEME |
| **Coordinate Frame (Analysis)** | ECI (J2000 / GCRS) |
