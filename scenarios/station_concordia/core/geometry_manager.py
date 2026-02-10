"""
Geometry management for JuPedSim simulation.

Handles loading, processing, and validation of station geometry including
walkable areas, entrance areas, platforms, and obstacles.
"""

from pathlib import Path
from typing import Any

import jupedsim as jps

from scenarios.common.logger import get_logger
from scenarios.station_jupedsim.geometry import (
    GeometryProcessor,
    load_entrance_areas,
    load_obstacles,
    load_platform_areas,
    load_walkable_areas,
)

logger = get_logger(__name__)


class GeometryManager:
    """
    Manages station geometry loading and JuPedSim simulation creation.

    Handles:
    - Loading geometry from SUMO network files
    - Integrating obstacles into walkable areas
    - Creating JuPedSim simulation with combined geometry
    - Providing access to geometry data for visualization
    """

    def __init__(self, network_path: Path, dt: float = 0.05):
        """
        Initialize geometry manager and load station geometry.

        Args:
            network_path: Path to network directory containing walking_areas.add.xml
            dt: Timestep in seconds (matches JuPedSim convention)

        Raises:
            FileNotFoundError: If geometry file not found
            ValueError: If no walkable areas found in geometry
        """
        self.network_path = network_path
        self.dt = dt

        # Load geometry
        logger.info("Loading station geometry from network files...")
        (
            self.walkable_areas,
            self.walkable_areas_with_obstacles,
            self.entrance_areas,
            self.platform_areas,
            self.obstacles,
        ) = self._load_geometry()

        # Create JuPedSim simulation
        logger.info("Initializing JuPedSim simulation...")
        self.simulation = self._create_simulation()

        logger.info(
            f"Geometry loaded: "
            f"{len(self.walkable_areas)} walkable areas, "
            f"{len(self.entrance_areas)} entrances, "
            f"{len(self.platform_areas)} platforms, "
            f"{len(self.obstacles)} obstacles"
        )

    def _load_geometry(self) -> tuple[dict, dict, dict, dict, list]:
        """
        Load station geometry from SUMO network files.

        Returns:
            Tuple of (walkable_areas, walkable_areas_with_obstacles,
                     entrance_areas, platform_areas, obstacles)

        Raises:
            FileNotFoundError: If walking_areas.add.xml not found
        """
        walking_areas_file = self.network_path / "walking_areas.add.xml"

        if not walking_areas_file.exists():
            raise FileNotFoundError(f"Geometry file not found: {walking_areas_file}")

        walkable_areas = load_walkable_areas(str(walking_areas_file))
        entrance_areas = load_entrance_areas(str(walking_areas_file))
        platform_areas = load_platform_areas(str(walking_areas_file))
        obstacles = load_obstacles(str(walking_areas_file))

        # Integrate obstacles into walkable areas as polygon holes
        walkable_areas_with_obstacles, fixed_obstacles = GeometryProcessor.integrate_obstacles(
            walkable_areas, obstacles
        )

        logger.info(f"  Loaded {len(walkable_areas)} walkable areas")
        logger.info(f"  Loaded {len(entrance_areas)} entrance areas")
        logger.info(f"  Loaded {len(platform_areas)} platform areas")
        logger.info(f"  Loaded {len(obstacles)} obstacles")
        logger.info(f"  Integrated {len(fixed_obstacles)} obstacles into walkable areas")

        return (
            walkable_areas,
            walkable_areas_with_obstacles,
            entrance_areas,
            platform_areas,
            fixed_obstacles,
        )

    def _create_simulation(self) -> jps.Simulation:
        """
        Create JuPedSim simulation with loaded geometry.

        Returns:
            Configured JuPedSim simulation instance

        Raises:
            ValueError: If no walkable areas found
        """
        # Merge all walkable areas (with obstacles removed) into one geometry
        all_areas = list(self.walkable_areas_with_obstacles.values())

        if not all_areas:
            raise ValueError("No walkable areas found in geometry")

        # Combine into a single geometry
        main_area = GeometryProcessor.combine_geometry(all_areas)

        # Create JuPedSim simulation
        simulation = jps.Simulation(
            model=jps.CollisionFreeSpeedModel(),
            geometry=main_area,
            dt=self.dt,
        )

        area = main_area.area if hasattr(main_area, "area") else 0.0
        logger.info(f"  Created simulation with area: {area:.1f} m²")

        return simulation

    def get_geometry_data(self) -> dict[str, Any]:
        """
        Get geometry information for visualization or analysis.

        Returns:
            Dictionary with all geometry components
        """
        return {
            "walkable_areas": self.walkable_areas,
            "walkable_areas_with_obstacles": self.walkable_areas_with_obstacles,
            "entrance_areas": self.entrance_areas,
            "platform_areas": self.platform_areas,
            "obstacles": self.obstacles,
        }
