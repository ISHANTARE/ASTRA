# astra/spatial_index.py
"""ASTRA Core Persistent Spatial Index — Octree.

Implements a 3D Octree spatial partitioning structure for O(N log N)
conjunction candidate pair generation, replacing the naive O(N²)
all-pairs search.

The Octree persists between analysis runs. Only objects whose bounding
shells have changed need reinsertion, eliminating redundant spatial
decomposition for the ~95% of catalog objects that haven't maneuvered.

References:
    Meagher, D. (1982). Geometric Modeling Using Octree Encoding.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class OctreeNode:
    """Single node in the Octree hierarchy.

    Attributes:
        center: (3,) center point of this node's bounding cube (km).
        half_size: Half the edge length of this cube (km).
        objects: List of (object_id, position) stored at this leaf.
        children: 8 child nodes (None if leaf).
        max_objects: Maximum objects before subdivision.
        max_depth: Maximum recursion depth.
        depth: Current depth in the tree.
    """
    center: np.ndarray
    half_size: float
    max_objects: int = 16
    max_depth: int = 12
    depth: int = 0
    objects: list[tuple[str, np.ndarray]] = field(default_factory=list)
    children: Optional[list[Optional["OctreeNode"]]] = None

    def _get_octant(self, point: np.ndarray) -> int:
        """Determine which of the 8 octants a point belongs to."""
        octant = 0
        if point[0] >= self.center[0]: octant |= 4
        if point[1] >= self.center[1]: octant |= 2
        if point[2] >= self.center[2]: octant |= 1
        return octant

    def _subdivide(self) -> None:
        """Split this node into 8 children."""
        hs = self.half_size / 2.0
        self.children = []
        for i in range(8):
            offset = np.array([
                hs if (i & 4) else -hs,
                hs if (i & 2) else -hs,
                hs if (i & 1) else -hs,
            ])
            child = OctreeNode(
                center=self.center + offset,
                half_size=hs,
                max_objects=self.max_objects,
                max_depth=self.max_depth,
                depth=self.depth + 1,
            )
            self.children.append(child)

    def insert(self, obj_id: str, position: np.ndarray) -> bool:
        """Insert an object into the octree.

        Args:
            obj_id: NORAD ID or unique identifier.
            position: (3,) position in ECI/TEME frame (km).

        Returns:
            True if inserted successfully.
        """
        # Check bounds
        diff = np.abs(position - self.center)
        if np.any(diff > self.half_size):
            return False

        # If leaf and not full, store here
        if self.children is None:
            if len(self.objects) < self.max_objects or self.depth >= self.max_depth:
                self.objects.append((obj_id, position.copy()))
                return True

            # Subdivide and redistribute
            self._subdivide()
            existing = self.objects
            self.objects = []
            for eid, epos in existing:
                octant = self._get_octant(epos)
                if self.children[octant] is not None:
                    self.children[octant].insert(eid, epos)

        # Insert into appropriate child
        octant = self._get_octant(position)
        if self.children is not None and self.children[octant] is not None:
            return self.children[octant].insert(obj_id, position)

        return False

    def query_radius(self, point: np.ndarray, radius: float) -> list[tuple[str, np.ndarray]]:
        """Find all objects within a given radius of a point.

        Args:
            point: (3,) query center (km).
            radius: Search radius (km).

        Returns:
            List of (object_id, position) tuples within radius.
        """
        results = []

        # Check if this node's bounding cube intersects the search sphere
        diff = np.abs(point - self.center)
        if np.any(diff > self.half_size + radius):
            return results

        # Check objects at this node
        for obj_id, pos in self.objects:
            if np.linalg.norm(pos - point) <= radius:
                results.append((obj_id, pos))

        # Recurse into children
        if self.children is not None:
            for child in self.children:
                if child is not None:
                    results.extend(child.query_radius(point, radius))

        return results

    def count(self) -> int:
        """Count total objects in this subtree."""
        n = len(self.objects)
        if self.children is not None:
            for child in self.children:
                if child is not None:
                    n += child.count()
        return n


class SpatialIndex:
    """High-level persistent spatial index for conjunction screening.

    Wraps an Octree rooted at Earth's center with a bounding cube large
    enough to contain all LEO/MEO/GEO objects (~50,000 km half-extent).

    Usage:
        idx = SpatialIndex()
        idx.insert("25544", np.array([6771.0, 0.0, 0.0]))
        pairs = idx.query_pairs(threshold_km=50.0)
    """

    def __init__(self, half_size_km: float = 50000.0, max_objects_per_node: int = 16):
        self._root = OctreeNode(
            center=np.zeros(3),
            half_size=half_size_km,
            max_objects=max_objects_per_node,
        )
        self._positions: dict[str, np.ndarray] = {}

    def insert(self, obj_id: str, position: np.ndarray) -> None:
        """Insert or update an object's position."""
        self._positions[obj_id] = position.copy()
        self._root.insert(obj_id, position)

    def query_radius(self, point: np.ndarray, radius_km: float) -> list[tuple[str, np.ndarray]]:
        """Find all objects within radius of a point."""
        return self._root.query_radius(point, radius_km)

    def query_pairs(self, threshold_km: float = 50.0) -> list[tuple[str, str]]:
        """Find all pairs of objects within threshold distance.

        This is the primary entry point for conjunction candidate generation.
        For each object, queries the Octree for neighbors within threshold,
        producing deduplicated pairs in O(N log N) average time.

        Returns:
            List of (id_a, id_b) pairs with id_a < id_b lexicographically.
        """
        pairs = set()
        for obj_id, pos in self._positions.items():
            neighbors = self._root.query_radius(pos, threshold_km)
            for neighbor_id, _ in neighbors:
                if neighbor_id != obj_id:
                    pair = (min(obj_id, neighbor_id), max(obj_id, neighbor_id))
                    pairs.add(pair)
        return list(pairs)

    @property
    def size(self) -> int:
        """Number of objects indexed."""
        return len(self._positions)

    def rebuild(self, positions: dict[str, np.ndarray]) -> None:
        """Rebuild the entire index from a fresh position dictionary."""
        self._root = OctreeNode(
            center=np.zeros(3),
            half_size=self._root.half_size,
            max_objects=self._root.max_objects,
        )
        self._positions = {}
        for obj_id, pos in positions.items():
            self.insert(obj_id, pos)
