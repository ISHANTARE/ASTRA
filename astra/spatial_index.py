# astra/spatial_index.py
"""ASTRA Core Persistent Spatial Index — KDTree.

Implements a 3D KDTree spatial partitioning structure for O(N log N)
conjunction candidate pair generation, replacing the naive O(N²)
all-pairs search.

Wraps the ultra-fast C++ scipy.spatial.cKDTree.
"""
from __future__ import annotations

import threading
import numpy as np
from scipy.spatial import cKDTree

class SpatialIndex:
    """High-level persistent spatial index for conjunction screening.

    Wraps scipy.spatial.cKDTree for robust and extremely fast spatial queries.

    Usage:
        idx = SpatialIndex()
        idx.insert("25544", np.array([6771.0, 0.0, 0.0]))
        pairs = idx.query_pairs(threshold_km=50.0)

    Args:
        half_size_km: Nominal maximum separation (km) for screening workflows.
            Pass the same value to :meth:`query_pairs` when you want a consistent
            radius; it is not applied automatically.
        max_objects_per_node: ``leafsize`` for SciPy's ``cKDTree`` (bucket size).
    """

    def __init__(self, half_size_km: float = 50000.0, max_objects_per_node: int = 16):
        self.half_size_km = float(half_size_km)
        self.max_objects_per_node = int(max_objects_per_node)
        self._tree: cKDTree | None = None
        self._excursions: dict[str, float] = {}  # Max distance from center in window
        self._max_excursion: float = 0.0
        self._lock = threading.RLock()
        self._positions: dict[str, np.ndarray] = {}
        self._ids: list[str] = []

    def insert(self, obj_id: str, position: np.ndarray) -> None:
        """Insert or update an object's position. Thread-safe."""
        with self._lock:
            self._positions[obj_id] = position.copy()
            self._tree = None

    def query_radius(self, point: np.ndarray, radius_km: float) -> list[tuple[str, np.ndarray]]:
        """Find all objects within radius of a point. Thread-safe."""
        with self._lock:
            self._ensure_tree()
            if self._tree is None:
                return []
            
            indices = self._tree.query_ball_point(point, r=radius_km)
            return [(self._ids[i], self._positions[self._ids[i]]) for i in indices]

    def query_pairs(self, threshold_km: float = 50.0) -> list[tuple[str, str]]:
        """Find all pairs of objects within threshold distance. Thread-safe."""
        with self._lock:
            self._ensure_tree()
            if self._tree is None or len(self._ids) < 2:
                return []
            
            # If this is a trajectory index, we must account for excursions
            if self._max_excursion > 0:
                # Conservative search: dist(centers) <= threshold + 2 * max_excursion
                # Then we refine based on individual excursions.
                combined_r = threshold_km + (2.0 * self._max_excursion)
                index_pairs = self._tree.query_pairs(r=combined_r, output_type='set')
                
                results = []
                for i, j in index_pairs:
                    id_a, id_b = self._ids[i], self._ids[j]
                    # Refinement: can they actually collide?
                    # dist(centers) <= threshold + R_a + R_b
                    dist_centers = np.linalg.norm(self._positions[id_a] - self._positions[id_b])
                    if dist_centers <= threshold_km + self._excursions[id_a] + self._excursions[id_b]:
                        results.append((min(id_a, id_b), max(id_a, id_b)))
                return results
            else:
                index_pairs = self._tree.query_pairs(r=threshold_km, output_type='set')
                results = []
                for i, j in index_pairs:
                    id_a, id_b = self._ids[i], self._ids[j]
                    results.append((min(id_a, id_b), max(id_a, id_b)))
                return results

    def rebuild_for_trajectories(self, trajectories: TrajectoryMap) -> None:
        """Build a unified spatial index for entire trajectories (high-performance).

        Uses the mean position of each trajectory as the center and tracks the
        maximum excursion from that center. Enables one-shot conjunction
        screening for the entire propagation window.
        """
        with self._lock:
            new_positions = {}
            new_excursions = {}
            for nid, traj in trajectories.items():
                if np.any(~np.isfinite(traj)):
                    continue
                # Center = average of start and end might be enough, but mean is safer
                center = np.mean(traj, axis=0)
                # Max excursion = max distance from center
                excursions = np.linalg.norm(traj - center, axis=1)
                max_exc = float(np.max(excursions))
                
                new_positions[nid] = center
                new_excursions[nid] = max_exc
            
            self._positions = new_positions
            self._excursions = new_excursions
            self._max_excursion = max(new_excursions.values()) if new_excursions else 0.0
            self._ensure_tree(force=True)

    @property
    def size(self) -> int:
        """Number of objects indexed."""
        return len(self._positions)

    def rebuild(self, positions: dict[str, np.ndarray]) -> None:
        """Rebuild the entire index from a fresh position dictionary. Thread-safe.
        
        Silently drops any objects whose position contains NaN or Inf.
        """
        with self._lock:
            self._positions = {
                k: v.copy() for k, v in positions.items() 
                if np.all(np.isfinite(v))
            }
            self._ensure_tree(force=True)
        
    def _ensure_tree(self, force: bool = False) -> None:
        """Internal tree reconstruction logic. MUST BE CALLED WITHIN _lock."""
        if (self._tree is None or force) and self._positions:
            self._ids = list(self._positions.keys())
            points = np.array([self._positions[nid] for nid in self._ids])
            if len(points) > 0:
                self._tree = cKDTree(
                    points, leafsize=max(1, self.max_objects_per_node)
                )
            else:
                self._tree = None
