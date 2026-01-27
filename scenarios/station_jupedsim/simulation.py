"""
Basic JuPedSim simulation setup for station scenario.

Orchestrates geometry processing, stage management, and simulation execution.
"""

import jupedsim as jps
from pathlib import Path
from typing import Dict
from shapely.geometry import Point

try:
    from .geometry import load_walkable_areas, load_obstacles, GeometryProcessor
    from .stage_manager import StageManager
except ImportError:
    from geometry import load_walkable_areas, load_obstacles, GeometryProcessor
    from stage_manager import StageManager


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
        self.obstacles = load_obstacles(str(walking_areas_file))
        
        # Track zones by name (original polygons for reference)
        self.zones = self.walkable_areas  # Map zone name -> Polygon (original)
        
        # Process geometry: integrate obstacles into zones
        self.zones_with_obstacles, fixed_obstacles = GeometryProcessor.integrate_obstacles(
            self.zones, 
            self.obstacles
        )
        
        # Combine processed areas for JuPedSim
        processed_areas = list(self.zones_with_obstacles.values())
        geometry = GeometryProcessor.combine_geometry(processed_areas)
        
        print(f"Loaded geometry: {len(self.walkable_areas)} walkable areas, {len(fixed_obstacles)} obstacles")
        print(f"Obstacles integrated as polygon holes")
        
        # Create JuPedSim simulation with CollisionFreeSpeedModel
        if output_file:
            writer = jps.SqliteTrajectoryWriter(output_file=output_file)
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
        
        # Initialize stage manager
        self.stage_manager = StageManager(self.simulation)
        
        # Iteration counter
        self.iteration = 0
        
    def setup_stages(self):
        """Define stages for the simulation (exits, waypoints)."""
        # Create exit at entrance zone centroid
        entrance_polygon = self.zones['entrance']
        exit_id = self.stage_manager.create_exit_at_zone_centroid(
            zone_name='main_exit',
            zone_polygon=entrance_polygon,
            width=30,
            height=30
        )
        
        return exit_id
    
    def create_simple_journey(self, start_zone: str, exit_id: int) -> int:
        """
        Create a simple journey from start zone to exit.
        
        Args:
            start_zone: Name of starting zone
            exit_id: Stage ID of exit
            
        Returns:
            Journey ID
        """
        journey_id = self.stage_manager.create_simple_exit_journey(
            journey_name=f"{start_zone}_to_exit",
            exit_id=exit_id
        )
        return journey_id
    
    def get_zone_for_position(self, x: float, y: float) -> str:
        """
        Determine which zone contains the given position.
        
        Args:
            x, y: Coordinates
            
        Returns:
            Zone name, or 'unknown' if not in any zone
        """
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
