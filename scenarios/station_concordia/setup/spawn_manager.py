"""
Spawn position management for Station Concordia simulations.

This module is responsible for:
- Generating spawn positions within geometry polygons
- Validating spawn positions against walkable areas
- Distributing agents across spawn areas
"""

import random

import jupedsim as jps

from scenarios.common.logger import get_logger

logger = get_logger(__name__)


class SpawnManager:
    """Handles generation of spawn positions for agents."""

    @staticmethod
    def generate_spawn_positions(
        jps_sim,
        num_agents: int,
        seed: int = 42,
    ) -> list[tuple[float, float]]:
        """
        Generate spawn positions for agents within the geometry.

        Uses JuPedSim's distribute_by_number on the actual simulation geometry
        to ensure positions respect boundary constraints and obstacles.

        Args:
            jps_sim: JuPedSim simulation instance with geometry
            num_agents: Number of spawn positions to generate
            seed: Random seed for reproducibility

        Returns:
            List of (x, y) coordinate tuples for spawn positions

        Raises:
            RuntimeError: If unable to generate spawn positions
        """
        random.seed(seed)

        # Get geometry from JuPedSim simulation
        walkable_areas_with_obstacles = jps_sim.walkable_areas_with_obstacles

        if not walkable_areas_with_obstacles:
            logger.error("No walkable areas with obstacles found in geometry")
            raise RuntimeError("Cannot spawn agents without geometry")

        # Generate spawn positions using JuPedSim's distribution on actual geometry polygons
        spawn_positions = SpawnManager._distribute_agents_across_areas(
            walkable_areas_with_obstacles, num_agents, seed
        )

        logger.info(
            f"Generated {len(spawn_positions)} spawn positions across "
            f"{len(walkable_areas_with_obstacles)} walkable areas"
        )

        return spawn_positions

    @staticmethod
    def _distribute_agents_across_areas(
        walkable_areas: dict, num_agents: int, seed: int
    ) -> list[tuple[float, float]]:
        """
        Distribute agents across all walkable areas using JuPedSim's distribution.

        Args:
            walkable_areas: Dictionary of area name -> polygon (with obstacles removed)
            num_agents: Total number of agents to spawn
            seed: Random seed

        Returns:
            List of (x, y) spawn positions
        """
        spawn_positions = []

        # Calculate total area
        area_list = list(walkable_areas.items())
        total_area = sum(poly.area for _, poly in area_list)

        logger.info(f"Distributing {num_agents} agents across {len(area_list)} walkable areas")

        for idx, (area_name, poly) in enumerate(area_list):
            # Proportional allocation based on polygon area
            poly_agents = int(num_agents * (poly.area / total_area))

            # Last polygon gets remainder to ensure exact count
            if idx == len(area_list) - 1:
                poly_agents = num_agents - len(spawn_positions)

            if poly_agents > 0:
                try:
                    # Use JuPedSim's distribution with safe boundary distances
                    positions = jps.distribute_by_number(
                        polygon=poly,
                        number_of_agents=poly_agents,
                        distance_to_agents=0.5,  # Min 0.5m between agents
                        distance_to_polygon=0.4,  # Min 0.4m from boundaries (>0.2 required)
                        seed=seed + idx,
                    )
                    spawn_positions.extend(positions)
                    logger.info(f"  {area_name}: {len(positions)} agents")
                except Exception as e:
                    logger.warning(f"  {area_name}: Failed to place {poly_agents} agents - {e}")
                    # Try with fewer agents if density is too high
                    if poly_agents > 1:
                        try:
                            reduced = max(1, poly_agents // 2)
                            positions = jps.distribute_by_number(
                                polygon=poly,
                                number_of_agents=reduced,
                                distance_to_agents=0.5,
                                distance_to_polygon=0.4,
                                seed=seed + idx,
                            )
                            spawn_positions.extend(positions)
                            logger.info(f"  {area_name}: {len(positions)} agents (reduced)")
                        except Exception as e2:
                            logger.error(f"  {area_name}: Could not place any agents - {e2}")

        if len(spawn_positions) < num_agents:
            logger.warning(
                f"Only generated {len(spawn_positions)} of {num_agents} requested positions"
            )

        return spawn_positions
