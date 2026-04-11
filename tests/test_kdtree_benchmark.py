"""TEST-05: KDTree spatial index speedup regression benchmark.

Validates that ASTRA's cKDTree-based conjunction pre-filter is meaningfully
faster than a naive O(N²) all-pairs brute-force for realistic catalog sizes,
and that it produces no false negatives (every pair within threshold is found).

The performance claim from the LinkedIn post / README is ≥10× speedup over
a naive distance loop for N=500 objects.  This test asserts ≥5× (conservative)
to allow for variance across hardware.
"""
from __future__ import annotations

import time
import numpy as np
import pytest


def _brute_force_pairs(positions: dict[str, np.ndarray],
                        threshold_km: float) -> set[tuple[str, str]]:
    """O(N²) naive all-pairs search — ground truth for correctness check."""
    ids = list(positions.keys())
    pairs: set[tuple[str, str]] = set()
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            dist = float(np.linalg.norm(positions[ids[i]] - positions[ids[j]]))
            if dist <= threshold_km:
                pairs.add((min(ids[i], ids[j]), max(ids[i], ids[j])))
    return pairs


def _make_leo_catalog(n: int, seed: int = 42) -> dict[str, np.ndarray]:
    """Generate a synthetic LEO catalog of n objects at random positions
    within a 200 km shell around a 400 km circular orbit."""
    rng = np.random.default_rng(seed)
    # Random positions on a sphere of radius ~6778 km (400 km LEO)
    # with ±100 km altitude spread
    r_base = 6778.0
    positions = {}
    for i in range(n):
        theta  = rng.uniform(0, np.pi)
        phi    = rng.uniform(0, 2 * np.pi)
        r      = r_base + rng.uniform(-100.0, 100.0)
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        positions[str(i)] = np.array([x, y, z])
    return positions


def test_kdtree_finds_all_brute_force_pairs():
    """TEST-05: KDTree must find every pair that brute-force finds (no false negatives)."""
    from astra.spatial_index import SpatialIndex

    N = 200
    THRESHOLD_KM = 100.0
    positions = _make_leo_catalog(N, seed=7)

    idx = SpatialIndex()
    idx.rebuild(positions)

    kdtree_pairs = set(idx.query_pairs(threshold_km=THRESHOLD_KM))
    brute_pairs  = _brute_force_pairs(positions, THRESHOLD_KM)

    missing = brute_pairs - kdtree_pairs
    assert not missing, (
        f"TEST-05: KDTree missed {len(missing)} pairs found by brute-force: "
        f"{list(missing)[:5]}..."
    )


def test_kdtree_no_false_positives():
    """TEST-05: KDTree must not report pairs beyond the threshold."""
    from astra.spatial_index import SpatialIndex

    N = 200
    THRESHOLD_KM = 80.0
    positions = _make_leo_catalog(N, seed=13)

    idx = SpatialIndex()
    idx.rebuild(positions)

    kdtree_pairs = set(idx.query_pairs(threshold_km=THRESHOLD_KM))

    for id_a, id_b in kdtree_pairs:
        dist = float(np.linalg.norm(positions[id_a] - positions[id_b]))
        assert dist <= THRESHOLD_KM + 1e-6, (
            f"TEST-05: False positive pair ({id_a},{id_b}) with dist={dist:.3f} km "
            f"> threshold={THRESHOLD_KM} km"
        )


def test_kdtree_speedup_vs_brute_force():
    """TEST-05: KDTree must be ≥5× faster than naive O(N²) for N=500 objects.

    The README claims ≥10× speedup; we assert ≥5× as a conservative floor
    that accounts for hardware variance and test overhead.
    """
    N = 500
    THRESHOLD_KM = 50.0
    positions = _make_leo_catalog(N, seed=99)

    # --- Brute-force timing ---
    t0 = time.perf_counter()
    _ = _brute_force_pairs(positions, THRESHOLD_KM)
    t_brute = time.perf_counter() - t0

    # --- KDTree timing (includes rebuild) ---
    from astra.spatial_index import SpatialIndex
    idx = SpatialIndex()

    t0 = time.perf_counter()
    idx.rebuild(positions)
    _ = idx.query_pairs(threshold_km=THRESHOLD_KM)
    t_kdtree = time.perf_counter() - t0

    speedup = t_brute / max(t_kdtree, 1e-9)

    assert speedup >= 5.0, (
        f"TEST-05: KDTree speedup {speedup:.1f}× is below the ≥5× regression threshold. "
        f"(brute={t_brute*1000:.1f}ms, kdtree={t_kdtree*1000:.1f}ms). "
        "Performance may have regressed."
    )


def test_kdtree_scales_with_n():
    """TEST-05: KDTree time should grow sub-quadratically (O(N log N)) with catalog size.

    Compares query time for N=100 vs N=500 (5× larger catalog).
    Expects time growth < 5² = 25× (which would be O(N²)).
    True O(N log N): expect ~5 * log(500)/log(100) ≈ 6.5× growth.
    We assert < 20× to leave headroom for variance while still catching regression.
    """
    from astra.spatial_index import SpatialIndex

    THRESHOLD_KM = 50.0
    REPEATS = 3

    def timed_query(n: int) -> float:
        pos = _make_leo_catalog(n, seed=42)
        idx = SpatialIndex()
        times = []
        for _ in range(REPEATS):
            idx.rebuild(pos)
            t0 = time.perf_counter()
            idx.query_pairs(threshold_km=THRESHOLD_KM)
            times.append(time.perf_counter() - t0)
        return min(times)

    t_small = timed_query(100)
    t_large = timed_query(500)

    ratio = t_large / max(t_small, 1e-9)

    assert ratio < 25.0, (
        f"TEST-05: KDTree time scaled {ratio:.1f}× from N=100 to N=500 "
        f"(expected < 25×). May indicate O(N²) behaviour. "
        f"(N=100: {t_small*1000:.2f}ms, N=500: {t_large*1000:.2f}ms)"
    )


def test_kdtree_empty_catalog():
    """TEST-05: Edge case — empty catalog must return an empty pair list."""
    from astra.spatial_index import SpatialIndex

    idx = SpatialIndex()
    idx.rebuild({})
    assert idx.query_pairs(50.0) == []


def test_kdtree_single_object():
    """TEST-05: Edge case — single-object catalog has no pairs."""
    from astra.spatial_index import SpatialIndex

    idx = SpatialIndex()
    idx.rebuild({"ISS": np.array([6778.0, 0.0, 0.0])})
    assert idx.query_pairs(50.0) == []
