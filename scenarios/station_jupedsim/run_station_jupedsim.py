"""
Run the JuPedSim station simulation.

This is the main entry point for the standalone JuPedSim implementation.
"""

import os
import sys
import time
from pathlib import Path
from typing import List

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from simulation import StationSimulation
from population_loader import create_agents_in_zone
from movement_jupedsim import JuPedSimMovementProvider

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
    
    # Setup stages
    print("[2/4] Setting up stages...")
    exit_id = sim.setup_stages()
    
    # Create simple journey (all zones -> exit)
    journey_id = sim.create_simple_journey('all_zones', exit_id)
    print(f"Created journey {journey_id}: all zones -> exit")
    
    # Create agents
    print("\n[3/4] Creating agent population...")
    agents: List[StationAgent] = []
    
    # Create movement provider for JuPedSim
    movement_provider = JuPedSimMovementProvider(sim.simulation, sim.zones)
    
    # Distribute agents across all platforms
    agent_distribution = {
        'platform_5_to_7': 15,
        'platform_3_to_4': 15,
        'foot_bridge': 10,
        'entrance': 20
    }
    
    for zone_name, num_agents in agent_distribution.items():
        if zone_name in sim.zones:
            zone_polygon = sim.zones[zone_name]
            create_agents_in_zone(
                simulation=sim.simulation,
                movement_provider=movement_provider,
                zone_name=zone_name,
                zone_polygon=zone_polygon,
                num_agents=num_agents,
                journey_id=journey_id,
                stage_id=exit_id,
                agent_list=agents,
                zones_with_obstacles=sim.zones_with_obstacles,
                destination="main_exit"
            )
        else:
            print(f"Warning: Zone '{zone_name}' not found")
    
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
