"""
Spawn position management for Station Concordia simulations.

This module is responsible for:
- Generating spawn positions within geometry polygons
- Validating spawn positions against walkable areas
- Distributing agents across spawn areas
"""

import random
from typing import List, Tuple

from shapely.geometry import Point

from scenarios.common.logger import get_logger

logger = get_logger(__name__)


class SpawnManager:
    """Handles generation of spawn positions for agents."""

    @staticmethod
    def generate_spawn_positions(
        jps_sim,
        num_agents: int,
        seed: int = 42,
    ) -> List[Tuple[float, float]]:
        """
        Generate spawn positions for agents within the geometry.

        Prefers spawning in 'entrance' area (near exits) for shorter evacuation.
        Falls back to platform areas or walkable areas if entrance not available.

        Args:
            jps_sim: JuPedSim simulation instance with geometry
            num_agents: Number of spawn positions to generate
            seed: Random seed for reproducibility

        Returns:
            List of (x, y) coordinate tuples for spawn positions

        Raises:
            RuntimeError: If no valid spawn polygons are found
        """
        random.seed(seed)

        # Get geometry from JuPedSim simulation
        walkable_areas = jps_sim.walkable_areas
        platform_areas = jps_sim.platform_areas

        # Determine spawn polygons (prefer entrance area)
        spawn_polygons = SpawnManager._select_spawn_polygons(walkable_areas, platform_areas)

        if not spawn_polygons:
            logger.error("No valid spawn polygons found in geometry")
            raise RuntimeError("Cannot spawn agents without geometry")

        # Get walkable polygons (with obstacles removed)
        walkable_polygons = SpawnManager._get_walkable_polygons(jps_sim, walkable_areas)

        # Generate spawn positions
        spawn_positions = SpawnManager._sample_positions(
            spawn_polygons, walkable_polygons, num_agents
        )

        logger.info(
            f"Generated {len(spawn_positions)} spawn positions from "
            f"{'platform' if platform_areas else 'walkable'} areas"
        )

        return spawn_positions

    @staticmethod
    def _select_spawn_polygons(walkable_areas: dict, platform_areas: dict) -> list:
        """
        Select which polygons to use for spawning agents.

        Args:
            walkable_areas: Dictionary of walkable area polygons
            platform_areas: Dictionary of platform area polygons

        Returns:
            List of polygons to spawn agents in
        """
        # TEMPORARY: Only spawn in 'entrance' area (near exits) for shorter evacuation
        spawn_polygons = []
        if "entrance" in walkable_areas:
            spawn_polygons = [walkable_areas["entrance"]]
            logger.info("🎯 Using 'entrance' area only for spawning (near exits)")
        else:
            logger.warning(f"⚠️ 'entrance' area not found! Available: {list(walkable_areas.keys())}")
            # Fallback to platform areas or all walkable areas
            spawn_polygons = list(platform_areas.values()) if platform_areas else []
            if not spawn_polygons and walkable_areas:
                spawn_polygons = list(walkable_areas.values())

        return spawn_polygons

    @staticmethod
    def _get_walkable_polygons(jps_sim, walkable_areas: dict) -> list:
        """
        Get walkable polygons (with obstacles removed if available).

        Args:
            jps_sim: JuPedSim simulation instance
            walkable_areas: Dictionary of walkable area polygons

        Returns:
            List of walkable polygons
        """
        walkable_polygons = list(getattr(jps_sim, "walkable_areas_with_obstacles", {}).values())
        if not walkable_polygons:
            walkable_polygons = list(walkable_areas.values())
        return walkable_polygons

    @staticmethod
    def _sample_positions(
        spawn_polygons: list,
        walkable_polygons: list,
        num_agents: int,
    ) -> List[Tuple[float, float]]:
        """
        Sample spawn positions from polygons.

        Uses weighted random selection based on polygon area for better distribution.

        Args:
            spawn_polygons: Polygons to spawn agents in
            walkable_polygons: Polygons that are walkable (for validation)
            num_agents: Number of positions to generate

        Returns:
            List of (x, y) coordinate tuples
        """
        # Weighted choice by polygon area for better distribution
        areas = [poly.area for poly in spawn_polygons]
        spawn_positions = []

        for _ in range(num_agents):
            chosen = random.choices(spawn_polygons, weights=areas, k=1)[0]
            candidate = SpawnManager._sample_point_in_polygon(chosen, random)

            # Ensure the spawn point is inside walkable geometry
            # Try up to 50 times to find a valid point within the chosen spawn polygon
            if walkable_polygons and not SpawnManager._point_in_any_polygon(
                candidate, walkable_polygons
            ):
                for _ in range(50):
                    candidate = SpawnManager._sample_point_in_polygon(chosen, random)
                    if SpawnManager._point_in_any_polygon(candidate, walkable_polygons):
                        break

            spawn_positions.append(candidate)

        return spawn_positions

    @staticmethod
    def _sample_point_in_polygon(polygon, rng, max_attempts: int = 200) -> Tuple[float, float]:
        """
        Sample a random point inside a polygon.

        Args:
            polygon: Shapely polygon to sample from
            rng: Random number generator
            max_attempts: Maximum attempts to find a point

        Returns:
            (x, y) coordinate tuple
        """
        min_x, min_y, max_x, max_y = polygon.bounds
        for _ in range(max_attempts):
            x = rng.uniform(min_x, max_x)
            y = rng.uniform(min_y, max_y)
            if polygon.contains(Point(x, y)):
                return (x, y)

        # Fallback to representative point
        rep = polygon.representative_point()
        return (rep.x, rep.y)

    @staticmethod
    def _point_in_any_polygon(point: Tuple[float, float], polygons: list) -> bool:
        """
        Check if a point is inside any of the given polygons.

        Args:
            point: (x, y) coordinate tuple
            polygons: List of Shapely polygons

        Returns:
            True if point is in any polygon, False otherwise
        """
        pt = Point(point)
        return any(poly.contains(pt) for poly in polygons)
