"""
Run the JuPedSim station simulation.

This is the main entry point for the standalone JuPedSim implementation.
"""

import os
import sys
import time
from pathlib import Path
from typing import List
import jupedsim as jps

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from simulation import StationSimulation
from population_loader import create_agents_from_entrances
from movement_jupedsim import JuPedSimMovementProvider
from geometry_loader import load_entrance_areas, load_platform_areas

# Import common agent
sys.path.append(str(Path(__file__).parent.parent))
from common.station_agent import StationAgent


def main():
    """Main simulation loop."""
    
    # Setup paths
    scenario_dir = Path(__file__).parent
    network_path = scenario_dir / ".." / "station_sim" / "network"
    output_dir = scenario_dir / "output"
    output_dir.mkdir(exist_ok=True)
    trajectory_file = output_dir / "trajectory.db"
    
    print("=" * 60)
    print("JuPedSim Station Simulation")
    print("=" * 60)
    
    # Initialize simulation
    print("\n[1/4] Initializing simulation...")
    sim = StationSimulation(str(network_path), dt=0.05, output_file=str(trajectory_file))
    print(f"  Trajectory output: {trajectory_file}")
    
    # Load entrance and platform areas
    walking_areas_file = network_path / "walking_areas.add.xml"
    entrance_areas = load_entrance_areas(str(walking_areas_file))
    platform_areas = load_platform_areas(str(walking_areas_file))
    print(f"Loaded {len(entrance_areas)} entrances, {len(platform_areas)} platforms")
    
    # Setup platform stages (waiting areas at each platform)
    print("\n[2/4] Setting up platform stages...")
    platform_stages = {}
    platform_journeys = {}
    
    for platform_name, platform_polygon in platform_areas.items():
        # Get representative point (guaranteed to be inside polygon, unlike centroid)
        point = platform_polygon.representative_point()
        position = (point.x, point.y)
        
        # Try to create waiting stage, skip if position is outside walkable area
        try:
            stage_id = sim.stage_manager.create_waiting_stage(
                name=platform_name,
                position=position
            )
            platform_stages[platform_name] = stage_id
            
            # Create journey for this platform (single-stage journey to the waypoint)
            journey = jps.JourneyDescription([stage_id])
            journey_id = sim.simulation.add_journey(journey)
            platform_journeys[platform_name] = journey_id
            
            print(f"  Created waiting stage for platform '{platform_name}' (stage_id={stage_id}, journey_id={journey_id})")
        except RuntimeError as e:
            print(f"  Warning: Skipped platform '{platform_name}' - position outside walkable area")
    
    if not platform_stages:
        print("Error: No valid platform stages created!")
        return
    
    # Create agents
    print("\n[3/4] Creating agent population...")
    agents: List[StationAgent] = []
    
    # Create movement provider for JuPedSim
    movement_provider = JuPedSimMovementProvider(sim.simulation, sim.zones)
    
    # Create agents at entrances with random platform destinations
    num_agents = 60
    create_agents_from_entrances(
        simulation=sim.simulation,
        movement_provider=movement_provider,
        entrance_areas=entrance_areas,
        platform_stages=platform_stages,
        platform_journeys=platform_journeys,
        num_agents=num_agents,
        agent_list=agents
    )
    
    print(f"\nTotal agents created: {len(agents)}")
    print(f"JuPedSim agent count: {sim.simulation.agent_count()}")
    
    # Run simulation
    print("\n[4/4] Running simulation...")
    print("Press Ctrl+C to stop\n")
    
    max_iterations = 2000  # ~100 seconds at 0.05s timestep
    
    # Start timer for real execution time
    start_time = time.time()
    
    try:
        while sim.step() and sim.iteration < max_iterations:
            # Print progress every 100 steps (5 seconds)
            if sim.iteration % 100 == 0:
                agent_count = sim.simulation.agent_count()
                sim_time = sim.get_simulation_time()
                print(f"t={sim_time:6.2f}s  agents={agent_count:3d}")
                
    except KeyboardInterrupt:
        print("\n\nSimulation interrupted by user")
    
    # Stop timer
    end_time = time.time()
    real_time_elapsed = end_time - start_time
    simulated_time = sim.get_simulation_time()
    
    # Summary
    print("\n" + "=" * 60)
    print("Simulation Complete")
    print("=" * 60)
    print(f"Total iterations: {sim.iteration}")
    print(f"Simulation time: {simulated_time:.2f}s")
    print(f"Real execution time: {real_time_elapsed:.2f}s")
    print(f"Speed factor: {simulated_time / real_time_elapsed:.2f}x realtime" if real_time_elapsed > 0 else "Speed factor: N/A")
    print(f"Remaining agents: {sim.simulation.agent_count()}")
    print(f"Agents who exited: {len(agents) - sim.simulation.agent_count()}")
    
    # Trajectory info
    print(f"\nTrajectory saved to: {trajectory_file}")
    
    # Run visualization automatically
    print("\n" + "=" * 60)
    print("Launching Visualization")
    print("=" * 60)
    
    try:
        import visualize
        visualize.visualize_simulation(str(trajectory_file), str(network_path))
    except Exception as e:
        print(f"Could not launch visualization: {e}")
        print("\nTo visualize manually, run:")
        print(f"  .venv/bin/python scenarios/station_jupedsim/visualize.py {trajectory_file}")


if __name__ == "__main__":
    main()
