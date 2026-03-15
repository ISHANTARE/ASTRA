# ASTRA — Orbital Computation Implementation Guidelines

## 1. Purpose of This Document
Orbital mechanics and conjunction analysis require careful and precise implementation. AI-generated code often makes mistakes when:
* implementing orbital propagation manually
* recalculating propagation excessively
* using inefficient nested loops
* ignoring coordinate frame differences

This document defines **strict rules for implementing orbital computation in ASTRA** to ensure scientific accuracy and high computational efficiency.

## 2. Mandatory Orbital Propagation Model
ASTRA must essentially use the **SGP4 orbital propagation model** for computing satellite positions.

**Rules:**
* NEVER manually implement orbital mechanics equations.
* NEVER approximate orbits using simple circular orbit formulas.
* ALWAYS use trusted Python libraries.

**Approved Libraries:**
* `sgp4`
* `skyfield`

**Example Incorrect Approach:**
```python
x = r * cos(theta)
y = r * sin(theta)
```
*Note: This assumes circular orbits and produces fundamentally incorrect satellite positions.*

## 3. Propagation Efficiency Rules
Propagation must be computed **per object, not per pair comparison**. 

**Incorrect Pattern (Inefficient):**
```python
for objectA in objects:
    for objectB in objects:
        for t in time_steps:
            propagate(objectA)
            propagate(objectB)
```

**Correct Pattern:**
1. Precompute trajectories for each object across all simulation time steps.
2. Store results in memory.
3. Use those stored trajectories during distance calculations.

**Example Conceptual Flow:**
`objects` → `propagate trajectories` → `store positions` → `perform pairwise distance checks`

This avoids redundant propagation and drastically reduces runtime computation.

## 4. Vectorized Distance Computation
Distance calculations across time steps should use **NumPy vectorization** instead of Python loops.

**Incorrect Pattern:**
```python
for t in time_steps:
    distance = sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)
```

**Preferred Pattern:**
Use NumPy arrays and compute distances for all time steps simultaneously.

**Example Conceptual Approach:**
```python
distances = numpy.linalg.norm(positionA - positionB, axis=1)
```

This allows efficient calculation of all distances across the 288 simulation steps in bulk.

## 5. Coordinate Frame Awareness
SGP4 outputs satellite positions in the **TEME reference frame**.

Visualization systems typically require **ECI or Earth-centered coordinate frames** (like ECEF).

If coordinate frames are mishandled, satellite positions may appear incorrect or misaligned in the visualization.

**Recommended Approach:**
Use libraries like **skyfield** to explicitly handle coordinate transformations when necessary. 

Developers should ensure coordinate systems are mathematically consistent between:
* propagation
* analysis
* visualization

## 6. ASTRA Simulation Parameters
The implementation must follow the core simulation parameters defined by the project accurately:

* **Prediction Window:** 24 hours
* **Time Resolution:** 5 minutes
* **Total Simulation Steps:** 288

These parameters must remain consistent across:
* propagation
* distance calculations
* conjunction analysis

## 7. Performance Expectations
The system must actively avoid naive O(N²) propagation loops. 

Performance should rely dynamically on:
* multi-stage filtering
* orbital spatial grid indexing
* vectorized distance computation
* trajectory precomputation

These techniques combined allow ASTRA to analyze thousands of objects efficiently even when constrained by free-tier infrastructure.

## 8. Summary for AI Coding Assistants
To properly assist the main developer AI, all coding assistants generating implementation code must follow these rules:

1. Use `sgp4` or `skyfield` for propagation.
2. Never implement orbital mechanics manually.
3. Precompute trajectories before distance calculations.
4. Use NumPy vectorized operations.
5. Maintain consistent coordinate frames (handle TEME properly).
6. Follow ASTRA’s defined simulation parameters.

This ensures that AI-generated code remains scientifically valid and computationally efficient without unnecessary backend refactoring.

## 9. Trajectory Precomputation Requirement

### Problem Description
Satellite propagation is computationally expensive.

A common mistake in orbital simulation systems is performing propagation **inside pairwise comparison loops**, which leads to extremely slow performance.

Example of an incorrect pattern:
```python
for objectA in objects:
    for objectB in objects:
        for time_step in simulation_times:
            propagate(objectA, time_step)
            propagate(objectB, time_step)
            compute_distance(...)
```

In this structure, each object may be propagated **thousands of times**, which wastes computation and severely slows down the analysis.

### Required Architecture
ASTRA must enforce **trajectory precomputation**.

Propagation must occur **once per object across the full simulation window**.

Example conceptual pipeline:
```
objects
↓
SGP4 propagation for each object across all simulation times
↓
store trajectory arrays
↓
generate candidate object pairs
↓
compute distances using precomputed trajectories
```

This approach ensures that:
* each object is propagated only once
* propagation results can be reused
* distance calculations become significantly faster.

### Conceptual Data Structure
Trajectories should be stored in memory using array structures.

Example conceptual representation:
```python
trajectories = {
    object_id_1: [[x1,y1,z1], [x2,y2,z2], ...],
    object_id_2: [[x1,y1,z1], [x2,y2,z2], ...]
}
```

Each trajectory array contains **288 coordinate points** corresponding to the 24-hour simulation window.

### Benefits
Trajectory precomputation provides major performance improvements:
* prevents redundant propagation calls
* allows vectorized distance calculations
* simplifies conjunction detection logic
* reduces overall computational load

This architecture is essential for allowing ASTRA to run efficiently on **student-level infrastructure and free-tier cloud services**.

### Implementation Note for AI Coding Assistants
AI-generated code must follow this rule:

Propagation must occur **outside any pairwise comparison loops**.

Distance calculations must operate only on **precomputed trajectory arrays**.
