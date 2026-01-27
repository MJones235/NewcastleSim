"""
Geometry processing utilities for JuPedSim simulations.

Handles obstacle integration, topology fixes, and polygon operations.
"""

from typing import List, Dict, Tuple
import shapely
from shapely.geometry import Polygon


class GeometryProcessor:
    """Processes geometry for JuPedSim simulations, handling obstacles and topology."""
    
    @staticmethod
    def fix_topology(polygon: Polygon) -> Polygon:
        """
        Fix invalid polygon topology using buffer(0) operation.
        
        Args:
            polygon: Input polygon that may have topology issues
            
        Returns:
            Topology-corrected polygon
        """
        return polygon.buffer(0)
    
    @staticmethod
    def integrate_obstacles(
        zones: Dict[str, Polygon],
        obstacles: List[Polygon]
    ) -> Tuple[Dict[str, Polygon], List[Polygon]]:
        """
        Integrate obstacles into walkable zones by subtracting them.
        
        In JuPedSim, obstacles must be represented as holes in polygons,
        not as separate geometry. This method creates zones with obstacles
        removed using shapely difference operations.
        
        Args:
            zones: Dictionary mapping zone names to polygons
            obstacles: List of obstacle polygons to subtract
            
        Returns:
            Tuple of:
                - Dictionary mapping zone names to polygons with obstacles removed
                - List of topology-fixed obstacles that were successfully processed
        """
        # Fix any topology issues in obstacles
        fixed_obstacles = []
        for obs in obstacles:
            fixed_obs = GeometryProcessor.fix_topology(obs)
            if not fixed_obs.is_empty and fixed_obs.is_valid:
                fixed_obstacles.append(fixed_obs)
        
        # Subtract obstacles from zones where they intersect
        zones_with_obstacles = {}
        
        for zone_name, zone_polygon in zones.items():
            # Fix zone polygon topology
            zone_with_holes = GeometryProcessor.fix_topology(zone_polygon)
            
            # Check which obstacles intersect this zone
            for obstacle in fixed_obstacles:
                if zone_with_holes.intersects(obstacle):
                    # Subtract obstacle from zone
                    try:
                        zone_with_holes = zone_with_holes.difference(obstacle)
                    except Exception as e:
                        print(f"Warning: Could not subtract obstacle from {zone_name}: {e}")
            
            if not zone_with_holes.is_empty:
                zones_with_obstacles[zone_name] = zone_with_holes
        
        return zones_with_obstacles, fixed_obstacles
    
    @staticmethod
    def combine_geometry(polygons: List[Polygon]) -> shapely.GeometryCollection:
        """
        Combine multiple polygons into a single geometry for JuPedSim.
        
        Args:
            polygons: List of polygons to combine
            
        Returns:
            GeometryCollection or single Polygon if only one input
        """
        if len(polygons) == 1:
            return polygons[0]
        else:
            return shapely.GeometryCollection(polygons)
