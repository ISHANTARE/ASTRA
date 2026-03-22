# astra/spatial_index.py
"""ASTRA Core Persistent Spatial Index — KDTree.

Implements a 3D KDTree spatial partitioning structure for O(N log N)
conjunction candidate pair generation, replacing the naive O(N²)
all-pairs search.

Wraps the ultra-fast C++ scipy.spatial.cKDTree.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

class SpatialIndex:
    """High-level persistent spatial index for conjunction screening.

    Wraps scipy.spatial.cKDTree for robust and extremely fast spatial queries.

    Usage:
        idx = SpatialIndex()
        idx.insert("25544", np.array([6771.0, 0.0, 0.0]))
        pairs = idx.query_pairs(threshold_km=50.0)
    """

    def __init__(self, half_size_km: float = 50000.0, max_objects_per_node: int = 16):
        self._positions: dict[str, np.ndarray] = {}
        self._ids: list[str] = []
        self._tree: cKDTree | None = None

    def insert(self, obj_id: str, position: np.ndarray) -> None:
        """Insert or update an object's position."""
        self._positions[obj_id] = position.copy()
        self._tree = None

    def query_radius(self, point: np.ndarray, radius_km: float) -> list[tuple[str, np.ndarray]]:
        """Find all objects within radius of a point."""
        self._ensure_tree()
        if self._tree is None:
            return []
            
        indices = self._tree.query_ball_point(point, r=radius_km)
        return [(self._ids[i], self._positions[self._ids[i]]) for i in indices]

    def query_pairs(self, threshold_km: float = 50.0) -> list[tuple[str, str]]:
        """Find all pairs of objects within threshold distance.

        This is the primary entry point for conjunction candidate generation.
        Returns:
            List of (id_a, id_b) pairs with id_a < id_b lexicographically.
        """
        self._ensure_tree()
        if self._tree is None or len(self._ids) < 2:
            return []
            
        index_pairs = self._tree.query_pairs(r=threshold_km, output_type='set')
        
        results = []
        for i, j in index_pairs:
            id_a, id_b = self._ids[i], self._ids[j]
            results.append((min(id_a, id_b), max(id_a, id_b)))
            
        return results

    @property
    def size(self) -> int:
        """Number of objects indexed."""
        return len(self._positions)

    def rebuild(self, positions: dict[str, np.ndarray]) -> None:
        """Rebuild the entire index from a fresh position dictionary."""
        self._positions = {k: v.copy() for k, v in positions.items()}
        self._ensure_tree(force=True)
        
    def _ensure_tree(self, force: bool = False) -> None:
        if (self._tree is None or force) and self._positions:
            self._ids = list(self._positions.keys())
            points = np.array([self._positions[nid] for nid in self._ids])
            if len(points) > 0:
                self._tree = cKDTree(points)
            else:
                self._tree = None
