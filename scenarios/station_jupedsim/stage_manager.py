"""
Stage management for JuPedSim simulations.

Handles creation of exits, waypoints, and journeys.
"""

import jupedsim as jps
from typing import List, Tuple
from shapely.geometry import Polygon


class StageManager:
    """Manages stages (exits, waypoints) and journeys in JuPedSim simulations."""
    
    def __init__(self, simulation: jps.Simulation):
        """
        Initialize stage manager.
        
        Args:
            simulation: JuPedSim simulation object
        """
        self.simulation = simulation
        self.exits = {}  # Map exit name -> stage ID
        self.waypoints = {}  # Map waypoint name -> stage ID
        self.journeys = {}  # Map journey name -> journey ID
    
    def create_exit_at_zone_centroid(
        self, 
        zone_name: str,
        zone_polygon: Polygon,
        width: float = 30.0,
        height: float = 30.0
    ) -> int:
        """
        Create an exit stage at the centroid of a zone.
        
        JuPedSim requires convex polygons for exits. This creates a
        rectangular exit centered on the zone's centroid.
        
        Args:
            zone_name: Name of the zone (for tracking)
            zone_polygon: Polygon defining the zone
            width: Width of exit rectangle in meters
            height: Height of exit rectangle in meters
            
        Returns:
            Stage ID of created exit
        """
        centroid = zone_polygon.centroid
        
        # Create rectangular exit
        exit_coords = [
            (centroid.x - width/2, centroid.y - height/2),
            (centroid.x + width/2, centroid.y - height/2),
            (centroid.x + width/2, centroid.y + height/2),
            (centroid.x - width/2, centroid.y + height/2)
        ]
        
        stage_id = self.simulation.add_exit_stage(exit_coords)
        self.exits[zone_name] = stage_id
        
        print(f"Created exit stage: {zone_name} (id={stage_id})")
        print(f"  Exit location: ({centroid.x:.1f}, {centroid.y:.1f})")
        print(f"  Exit size: {width}m x {height}m")
        
        return stage_id
    
    def create_exit_at_coordinates(
        self,
        exit_name: str,
        coords: List[Tuple[float, float]]
    ) -> int:
        """
        Create an exit stage at specific coordinates.
        
        Args:
            exit_name: Name for the exit (for tracking)
            coords: List of (x, y) coordinate tuples defining exit polygon
            
        Returns:
            Stage ID of created exit
        """
        stage_id = self.simulation.add_exit_stage(coords)
        self.exits[exit_name] = stage_id
        
        print(f"Created exit stage: {exit_name} (id={stage_id})")
        
        return stage_id
    
    def create_waypoint(
        self,
        waypoint_name: str,
        coords: List[Tuple[float, float]],
        distance: float = 1.0
    ) -> int:
        """
        Create a waypoint stage.
        
        Args:
            waypoint_name: Name for the waypoint (for tracking)
            coords: List of (x, y) coordinate tuples defining waypoint polygon
            distance: Distance threshold for considering waypoint reached
            
        Returns:
            Stage ID of created waypoint
        """
        stage_id = self.simulation.add_waypoint_stage(coords, distance)
        self.waypoints[waypoint_name] = stage_id
        
        print(f"Created waypoint stage: {waypoint_name} (id={stage_id})")
        
        return stage_id
    
    def create_journey(
        self,
        journey_name: str,
        stage_ids: List[int]
    ) -> int:
        """
        Create a journey through a sequence of stages.
        
        Args:
            journey_name: Name for the journey (for tracking)
            stage_ids: List of stage IDs in order
            
        Returns:
            Journey ID
        """
        journey = jps.JourneyDescription(stage_ids)
        journey_id = self.simulation.add_journey(journey)
        self.journeys[journey_name] = journey_id
        
        return journey_id
    
    def create_simple_exit_journey(
        self,
        journey_name: str,
        exit_id: int
    ) -> int:
        """
        Create a simple journey that goes directly to an exit.
        
        Args:
            journey_name: Name for the journey (for tracking)
            exit_id: Stage ID of the exit
            
        Returns:
            Journey ID
        """
        return self.create_journey(journey_name, [exit_id])
    
    def get_exit_id(self, exit_name: str) -> int:
        """Get stage ID for a named exit."""
        return self.exits.get(exit_name)
    
    def get_waypoint_id(self, waypoint_name: str) -> int:
        """Get stage ID for a named waypoint."""
        return self.waypoints.get(waypoint_name)
    
    def get_journey_id(self, journey_name: str) -> int:
        """Get journey ID for a named journey."""
        return self.journeys.get(journey_name)
