"""
Evacuation exit management for JuPedSim simulation.

Handles creation and management of evacuation exits and journey routing
to exits in the station environment.
"""

from typing import Any

from scenarios.common.logger import get_logger
from scenarios.station_jupedsim.core.stage_manager import StageManager
from scenarios.station_jupedsim.geometry import GeometryProcessor

logger = get_logger(__name__)


class ExitManager:
    """
    Manages evacuation exits and journey routing.

    Handles:
    - Creation of exit stages from entrance polygons
    - Journey routing to exits
    - Exit ID and journey ID tracking
    - Exit placement validation
    """

    def __init__(
        self,
        stage_manager: StageManager,
        entrance_areas: dict[str, Any],
        walkable_areas_with_obstacles: dict[str, Any],
    ):
        """
        Initialize exit manager and create evacuation exits.

        Args:
            stage_manager: StageManager instance for creating exits and journeys
            entrance_areas: Dictionary of entrance area polygons
            walkable_areas_with_obstacles: Dictionary of walkable area polygons

        Raises:
            RuntimeError: If no valid exits can be created from entrance areas
        """
        self.stage_manager = stage_manager
        self.entrance_areas = entrance_areas

        # Setup evacuation exits and routes
        logger.info("Setting up evacuation exits...")

        walkable_geometry = GeometryProcessor.combine_geometry(
            list(walkable_areas_with_obstacles.values())
        )

        self.evacuation_exits, self.evacuation_journeys = self._setup_evacuation_exits(
            walkable_geometry
        )

        logger.info(
            f"Created {len(self.evacuation_exits)} exits: {list(self.evacuation_exits.keys())}"
        )

    def _setup_evacuation_exits(
        self, walkable_geometry: Any
    ) -> tuple[dict[str, int], dict[str, int]]:
        """
        Create evacuation exit stages at entrance locations.

        Args:
            walkable_geometry: Combined walkable geometry for validation

        Returns:
            Tuple of (evacuation_exits dict, evacuation_journeys dict)

        Raises:
            RuntimeError: If no valid exits can be created from entrance areas
        """
        evacuation_exits = {}
        evacuation_journeys = {}

        if not self.entrance_areas:
            raise RuntimeError(
                "No entrance areas found in geometry. "
                "Check that walking_areas.add.xml contains entrance area definitions."
            )

        failed_exits = []
        for entrance_name, entrance_polygon in self.entrance_areas.items():
            exit_id = self._create_convex_exit_from_polygon(
                entrance_name, entrance_polygon, walkable_geometry
            )

            if exit_id is None:
                failed_exits.append(entrance_name)
                logger.warning(f"Failed to create exit at '{entrance_name}'")
                continue

            evacuation_exits[entrance_name] = exit_id

            # Create journey to this exit
            journey_id = self.stage_manager.create_simple_exit_journey(
                journey_name=f"journey_to_{entrance_name}", exit_id=exit_id
            )
            evacuation_journeys[entrance_name] = journey_id

            logger.info(
                f"Created evacuation exit '{entrance_name}' "
                f"(exit={exit_id}, journey={journey_id})"
            )

        if not evacuation_exits:
            raise RuntimeError(
                f"Failed to create any evacuation exits. "
                f"Attempted exits: {list(self.entrance_areas.keys())}. "
                f"All exits failed. Check geometry configuration and ensure "
                f"entrance areas overlap with walkable areas."
            )

        if failed_exits:
            logger.warning(
                f"Successfully created {len(evacuation_exits)} exits, "
                f"but {len(failed_exits)} failed: {failed_exits}"
            )

        return evacuation_exits, evacuation_journeys

    def _create_convex_exit_from_polygon(
        self,
        exit_name: str,
        polygon: Any,
        walkable_geometry: Any,
    ) -> int | None:
        """
        Create a convex rectangular exit centered on polygon centroid.

        Args:
            exit_name: Name for the exit
            polygon: Entrance polygon to create exit from
            walkable_geometry: Combined walkable geometry for validation

        Returns:
            Exit ID if successful, None if exit couldn't be placed
        """
        # Find a valid centroid within walkable geometry
        centroid = polygon.centroid
        if not walkable_geometry.contains(centroid):
            intersection = polygon.intersection(walkable_geometry)
            if not intersection.is_empty:
                centroid = intersection.representative_point()
            else:
                logger.error(
                    f"Exit '{exit_name}': polygon doesn't intersect walkable geometry. "
                    f"Polygon bounds: {polygon.bounds}"
                )
                return None

        # Create exit with standard size
        exit_size = 4.0  # 4x4 meter exit (standard)
        exit_coords = [
            (centroid.x - exit_size / 2, centroid.y - exit_size / 2),
            (centroid.x + exit_size / 2, centroid.y - exit_size / 2),
            (centroid.x + exit_size / 2, centroid.y + exit_size / 2),
            (centroid.x - exit_size / 2, centroid.y + exit_size / 2),
        ]

        exit_id = self.stage_manager.create_exit_at_coordinates(exit_name, exit_coords)
        logger.info(
            f"Created {exit_size}m x {exit_size}m exit '{exit_name}' " f"at {centroid.coords[0]}"
        )
        return exit_id

    def get_default_exit(self) -> tuple[str, int, int]:
        """
        Get default exit for agents.

        Returns:
            Tuple of (exit_name, exit_id, journey_id)

        Raises:
            RuntimeError: If no evacuation exits are available
        """
        if not self.evacuation_exits:
            raise RuntimeError(
                "Cannot get default exit: no evacuation exits available. "
                "Check geometry configuration."
            )

        exit_name = list(self.evacuation_exits.keys())[0]
        exit_id = self.evacuation_exits[exit_name]
        journey_id = self.evacuation_journeys[exit_name]

        return exit_name, exit_id, journey_id

    def get_exit_ids(self, exit_name: str) -> tuple[int, int]:
        """
        Get stage and journey IDs for a specific exit.

        Args:
            exit_name: Name of the evacuation exit

        Returns:
            Tuple of (stage_id, journey_id)

        Raises:
            KeyError: If exit name not found
        """
        if exit_name not in self.evacuation_journeys:
            raise KeyError(
                f"Unknown exit: {exit_name}. "
                f"Available exits: {list(self.evacuation_journeys.keys())}"
            )

        stage_id = self.evacuation_exits[exit_name]
        journey_id = self.evacuation_journeys[exit_name]

        return stage_id, journey_id
