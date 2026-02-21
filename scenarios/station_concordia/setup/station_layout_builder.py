"""
Station layout builder for Station Concordia simulations.

This module is responsible for:
- Building station layout dictionary from simulation geometry
- Processing entrance and platform areas
- Creating zone definitions
- Handling walkable areas and obstacles
"""

from typing import Any

from scenarios.common.logger import get_logger
from scenarios.station_concordia.jps_integration.simulation_interface import PedestrianSimulation

logger = get_logger(__name__)


class StationLayoutBuilder:
    """Handles creation of station layout from simulation geometry."""

    @staticmethod
    def build_layout(jps_sim: PedestrianSimulation, config: dict) -> dict[str, Any]:
        """
        Build station layout dictionary from pedestrian simulation geometry.

        Args:
            jps_sim: Pedestrian simulation instance (implements PedestrianSimulation)
            config: Configuration dictionary

        Returns:
            Dictionary containing station layout information including:
            - exits: Dictionary of exit names to (x, y) coordinates
            - exits_polygons: Dictionary of exit area polygons
            - exits: Dictionary of exit names to (x, y) coordinates
            - exits_polygons: Dictionary of exit area polygons
            - walkable_areas: Dictionary of walkable area polygons
            - zones: Dictionary of zone boundaries
            - zones_polygons: Dictionary of zone polygons
            - obstacles: List of obstacle polygons
        """
        # For multi-level simulations, consolidate exits from all levels
        if hasattr(jps_sim, "simulations"):
            # Multi-level: Consolidate exits and zones from ALL levels
            all_exits = {}
            all_exit_polygons = {}
            all_zones = {}
            all_zone_polygons = {}

            # Collect exits from each level
            for level_id in sorted(jps_sim.simulations.keys()):
                level_sim = jps_sim.simulations[level_id]
                gm = level_sim.geometry_manager

                # Add street exits (from entrance areas)
                for name, poly in gm.entrance_areas.items():
                    all_exits[name] = (poly.centroid.x, poly.centroid.y)
                    all_exit_polygons[name] = poly

                # Add zones from this level
                for zone_name, zone_poly in gm.platform_areas.items():
                    zone_key = (
                        f"{zone_name}_L{level_id}" if len(jps_sim.simulations) > 1 else zone_name
                    )
                    all_zones[zone_key] = StationLayoutBuilder._polygon_bounds(zone_poly)
                    all_zone_polygons[zone_key] = zone_poly

            # Add escalator exits from all levels
            for level_sim in jps_sim.simulations.values():
                for exit_name in level_sim.exit_manager.evacuation_exits:
                    if exit_name.startswith("escalator_"):
                        all_exits[exit_name] = level_sim.exit_manager.exit_coordinates.get(
                            exit_name, (0, 0)
                        )
        else:
            # Single-level: Use geometry manager and exit manager
            gm = jps_sim.geometry_manager
            all_exits = {
                name: (poly.centroid.x, poly.centroid.y) for name, poly in gm.entrance_areas.items()
            }
            all_exit_polygons = gm.entrance_areas
            all_zones = StationLayoutBuilder._build_zones(jps_sim, gm.platform_areas)
            all_zone_polygons = StationLayoutBuilder._build_zone_polygons(
                jps_sim, gm.platform_areas
            )

        station_layout = {
            **config.get("station", {}),
            "exits": all_exits,
            "exits_polygons": all_exit_polygons,
            "walkable_areas": jps_sim.geometry_manager.walkable_areas_with_obstacles,
            "zones": all_zones,
            "zones_polygons": all_zone_polygons,
            "obstacles": jps_sim.geometry_manager.obstacles,
        }

        logger.info(
            f"Built station layout with {len(all_exits)} exits (street + escalators) and {len(all_zones)} zones"
        )
        return station_layout

    @staticmethod
    def _build_zones(
        jps_sim: PedestrianSimulation, platform_areas: dict
    ) -> dict[str, dict[str, float]]:
        """
        Build zone boundary definitions.

        Args:
            jps_sim: Pedestrian simulation instance (implements PedestrianSimulation)
            platform_areas: Dictionary of platform area polygons

        Returns:
            Dictionary mapping zone names to boundary dictionaries
        """
        if platform_areas:
            return {
                name: StationLayoutBuilder._polygon_bounds(poly)
                for name, poly in platform_areas.items()
            }
        else:
            # Fallback to main walkable area
            main_area = list(jps_sim.geometry_manager.walkable_areas.values())[0]
            return {"main_area": StationLayoutBuilder._polygon_bounds(main_area)}

    @staticmethod
    def _build_zone_polygons(jps_sim, platform_areas: dict) -> dict[str, Any]:
        """
        Build zone polygon definitions.

        Args:
            jps_sim: JuPedSim simulation instance
            platform_areas: Dictionary of platform area polygons

        Returns:
            Dictionary mapping zone names to polygons
        """
        if platform_areas:
            return platform_areas
        else:
            # Fallback to main walkable area
            main_area = list(jps_sim.geometry_manager.walkable_areas.values())[0]
            return {"main_area": main_area}

    @staticmethod
    def _polygon_bounds(polygon) -> dict[str, float]:
        """
        Extract bounding box from a polygon.

        Args:
            polygon: Shapely polygon

        Returns:
            Dictionary with x_min, x_max, y_min, y_max keys
        """
        min_x, min_y, max_x, max_y = polygon.bounds
        return {"x_min": min_x, "x_max": max_x, "y_min": min_y, "y_max": max_y}
