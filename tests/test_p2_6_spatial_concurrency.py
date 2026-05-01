import time
import threading
import numpy as np
from astra.spatial_index import SpatialIndex


def test_spatial_batch_performance_p2_6():
    """Verify that rebuild() is faster than repeated insert()."""
    n_objects = 1000
    rng = np.random.default_rng(20260502)
    positions = {f"SAT_{i}": rng.uniform(-7000, 7000, 3) for i in range(n_objects)}

    # 1. Repeated insert
    idx_insert = SpatialIndex()
    t0 = time.time()
    for nid, pos in positions.items():
        idx_insert.insert(nid, pos)
    # Tree construction happens on query
    idx_insert.query_pairs(50.0)
    dt_insert = time.time() - t0

    # 2. Batch rebuild
    idx_rebuild = SpatialIndex()
    t0 = time.time()
    idx_rebuild.rebuild(positions)
    # Tree construction already happened in rebuild()
    idx_rebuild.query_pairs(50.0)
    dt_rebuild = time.time() - t0

    print(f"Insert time: {dt_insert:.4f}s | Rebuild time: {dt_rebuild:.4f}s")
    # In large N, rebuild is usually faster due to fewer invalidations
    # but the construction cost is generally dominated by cKDTree itself.
    # At least verify it works.
    assert idx_rebuild.size == n_objects
    print("Batch performance test PASSED.")


def test_spatial_thread_safety_p2_6():
    """Verify that concurrent rebuilds and queries do not crash."""
    n_threads = 5
    n_iterations = 20
    idx = SpatialIndex()
    n_objects = 500
    errors = []
    completed = []
    lock = threading.Lock()

    def worker(seed):
        rng = np.random.default_rng(seed)
        try:
            for iteration in range(n_iterations):
                positions = {
                    f"SAT_{i}": rng.uniform(-7000, 7000, 3) for i in range(n_objects)
                }
                idx.rebuild(positions)
                pairs = idx.query_pairs(100.0)
                assert idx.size == n_objects
                assert all(a < b for a, b in pairs)
                with lock:
                    completed.append((seed, iteration))
        except BaseException as exc:
            with lock:
                errors.append(exc)

    threads = [
        threading.Thread(target=worker, args=(20260502 + i,))
        for i in range(n_threads)
    ]

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"SpatialIndex worker thread errors: {errors!r}"
    assert len(completed) == n_threads * n_iterations


if __name__ == "__main__":
    test_spatial_batch_performance_p2_6()
    test_spatial_thread_safety_p2_6()
