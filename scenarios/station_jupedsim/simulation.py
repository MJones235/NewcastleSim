"""
Basic JuPedSim simulation setup for station scenario.
"""

import jupedsim as jps
from pathlib import Path
from typing import List, Tuple, Dict
from .geometry_loader import load_walkable_areas, combine_walkable_geometry


class StationSimulation:
    """Manages a standalone JuPedSim station simulation."""
    
    def __init__(self, network_path: str, dt: float = 0.05, output_file: str = None):
        """
        Initialize the simulation.
        
        Args:
            network_path: Path to directory containing walking_areas.add.xml
            dt: Simulation time step in seconds (default 0.05s = 20 fps)
            output_file: Optional path to save trajectory data (sqlite)
        """
        self.dt = dt
        self.network_path = Path(network_path)
        self.output_file = output_file
        
        # Load geometry
        walking_areas_file = self.network_path / "walking_areas.add.xml"
        self.walkable_areas = load_walkable_areas(str(walking_areas_file))
        
        # Track zones by name (original polygons for agent spawning)
        self.zones = self.walkable_areas  # Map zone name -> Polygon (original, no holes)
        self.zones_with_obstacles = {}  # Map zone name -> Polygon (with obstacles cut out)
        
        # Load obstacles
        from .geometry_loader import load_obstacles
        self.obstacles = load_obstacles(str(walking_areas_file))
        
        # In JuPedSim, obstacles must be defined as holes in polygons
        # Fix any topology issues by buffering obstacles slightly
        import shapely
        
        fixed_obstacles = []
        for obs in self.obstacles:
            # Buffer by 0 to fix invalid geometries
            fixed_obs = obs.buffer(0)
            if not fixed_obs.is_empty and fixed_obs.is_valid:
                fixed_obstacles.append(fixed_obs)
        
        # For now, we'll subtract obstacles from all walkable areas
        # This creates proper polygons with holes that JuPedSim understands
        processed_areas = []
        for zone_name, zone_polygon in self.walkable_areas.items():
            # Fix zone polygon topology
            zone_with_holes = zone_polygon.buffer(0)
            
            # Check which obstacles intersect this zone
            for obstacle in fixed_obstacles:
                if zone_with_holes.intersects(obstacle):
                    # Subtract obstacle from zone
                    try:
                        zone_with_holes = zone_with_holes.difference(obstacle)
                    except Exception as e:
                        print(f"Warning: Could not subtract obstacle from {zone_name}: {e}")
            
            if not zone_with_holes.is_empty:
                processed_areas.append(zone_with_holes)
                self.zones_with_obstacles[zone_name] = zone_with_holes
        
        # Combine all processed areas
        if len(processed_areas) == 1:
            geometry = processed_areas[0]
        else:
            geometry = shapely.GeometryCollection(processed_areas)
        
        print(f"Loaded geometry: {len(self.walkable_areas)} walkable areas, {len(fixed_obstacles)} obstacles")
        print(f"Obstacles integrated as polygon holes")
        
        # Create JuPedSim simulation
        # Using CollisionFreeSpeedModel (similar to what SUMO uses)
        if output_file:
            # Create simulation with trajectory writer
            writer = jps.SqliteTrajectoryWriter(
                output_file=output_file
            )
            self.simulation = jps.Simulation(
                model=jps.CollisionFreeSpeedModel(),
                geometry=geometry,
                dt=dt,
                trajectory_writer=writer
            )
        else:
            self.simulation = jps.Simulation(
                model=jps.CollisionFreeSpeedModel(),
                geometry=geometry,
                dt=dt
            )
        
        # Stage IDs
        self.entrance_exits: Dict[str, int] = {}
        self.zone_waypoints: Dict[str, int] = {}
        
        # Iteration counter
        self.iteration = 0
        
    def setup_stages(self):
        """Define stages for the simulation (exits, waypoints)."""
        # Create exits at entrance zone (agents leave through entrance)
        # JuPedSim requires CONVEX polygons for exits
        
        # Get centroid of entrance zone for better exit placement
        entrance_polygon = self.zones['entrance']
        centroid = entrance_polygon.centroid
        
        # Create a larger exit rectangle near entrance centroid
        exit_width = 30
        exit_height = 30
        exit_coords = [
            (centroid.x - exit_width/2, centroid.y - exit_height/2),
            (centroid.x + exit_width/2, centroid.y - exit_height/2),
            (centroid.x + exit_width/2, centroid.y + exit_height/2),
            (centroid.x - exit_width/2, centroid.y + exit_height/2)
        ]
        self.entrance_exits['main_exit'] = self.simulation.add_exit_stage(exit_coords)
        
        print(f"Created exit stage: main_exit (id={self.entrance_exits['main_exit']})")
        print(f"  Exit location: ({centroid.x:.1f}, {centroid.y:.1f})")
        print(f"  Exit size: {exit_width}m x {exit_height}m")
        
    def create_simple_journey(self, start_zone: str, exit_id: int) -> int:
        """
        Create a simple journey from start zone to exit.
        
        Args:
            start_zone: Name of starting zone
            exit_id: Stage ID of exit
            
        Returns:
            Journey ID
        """
        # For now, simple journey: just go to exit
        journey = jps.JourneyDescription([exit_id])
        journey_id = self.simulation.add_journey(journey)
        
        return journey_id
    
    def get_zone_for_position(self, x: float, y: float) -> str:
        """
        Determine which zone contains the given position.
        
        Args:
            x, y: Coordinates
            
        Returns:
            Zone name, or 'unknown' if not in any zone
        """
        from shapely.geometry import Point
        point = Point(x, y)
        
        for zone_name, polygon in self.zones.items():
            if polygon.contains(point):
                return zone_name
                
        return 'unknown'
    
    def step(self) -> bool:
        """
        Advance simulation by one time step.
        
        Returns:
            True if simulation should continue, False if done
        """
        if self.simulation.agent_count() == 0:
            return False
            
        self.simulation.iterate()
        self.iteration += 1
        
        return True
    
    def get_simulation_time(self) -> float:
        """Get current simulation time in seconds."""
        return self.iteration * self.dt


if __name__ == "__main__":
    # Test basic simulation setup
    import os
    
    network_path = os.path.join(
        os.path.dirname(__file__),
        "..", "station_sim", "network"
    )
    
    print("Initializing simulation...")
    sim = StationSimulation(network_path)
    
    print(f"\nLoaded {len(sim.zones)} zones:")
    for zone_name, polygon in sim.zones.items():
        print(f"  {zone_name}: area={polygon.area:.2f}")
    
    print("\nSetting up stages...")
    sim.setup_stages()
    
    print(f"\nSimulation ready!")
    print(f"  Time step: {sim.dt}s")
    print(f"  Agent count: {sim.simulation.agent_count()}")
