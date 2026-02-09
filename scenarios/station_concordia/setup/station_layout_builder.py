"""
Station layout builder for Station Concordia simulations.

This module is responsible for:
- Building station layout dictionary from JuPedSim geometry
- Processing entrance and platform areas
- Creating zone definitions
- Handling walkable areas and obstacles
"""

from typing import Dict, Any

from scenarios.common.logger import get_logger

logger = get_logger(__name__)


class StationLayoutBuilder:
    """Handles creation of station layout from JuPedSim geometry."""

    @staticmethod
    def build_layout(jps_sim, config: dict) -> Dict[str, Any]:
        """
        Build station layout dictionary from JuPedSim simulation geometry.

        Args:
            jps_sim: JuPedSim simulation instance with geometry loaded
            config: Configuration dictionary

        Returns:
            Dictionary containing station layout information including:
            - exits: Dictionary of exit names to (x, y) coordinates
            - exits_polygons: Dictionary of exit area polygons
            - walkable_areas: Dictionary of walkable area polygons
            - zones: Dictionary of zone boundaries
            - zones_polygons: Dictionary of zone polygons
            - obstacles: List of obstacle polygons
        """
        entrance_areas = jps_sim.entrance_areas
        platform_areas = jps_sim.platform_areas

        station_layout = {
            **config.get("station", {}),
            "exits": {
                name: (poly.centroid.x, poly.centroid.y) for name, poly in entrance_areas.items()
            },
            "exits_polygons": entrance_areas,
            "walkable_areas": getattr(jps_sim, "walkable_areas_with_obstacles", None)
            or jps_sim.walkable_areas,
            "zones": StationLayoutBuilder._build_zones(jps_sim, platform_areas),
            "zones_polygons": StationLayoutBuilder._build_zone_polygons(jps_sim, platform_areas),
            "obstacles": jps_sim.obstacles,
        }

        logger.info(f"Built station layout with {len(entrance_areas)} exits")
        return station_layout

    @staticmethod
    def _build_zones(jps_sim, platform_areas: dict) -> Dict[str, Dict[str, float]]:
        """
        Build zone boundary definitions.

        Args:
            jps_sim: JuPedSim simulation instance
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
            main_area = list(jps_sim.walkable_areas.values())[0]
            return {"main_area": StationLayoutBuilder._polygon_bounds(main_area)}

    @staticmethod
    def _build_zone_polygons(jps_sim, platform_areas: dict) -> Dict[str, Any]:
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
            main_area = list(jps_sim.walkable_areas.values())[0]
            return {"main_area": main_area}

    @staticmethod
    def _polygon_bounds(polygon) -> Dict[str, float]:
        """
        Extract bounding box from a polygon.

        Args:
            polygon: Shapely polygon

        Returns:
            Dictionary with x_min, x_max, y_min, y_max keys
        """
        min_x, min_y, max_x, max_y = polygon.bounds
        return {"x_min": min_x, "x_max": max_x, "y_min": min_y, "y_max": max_y}
