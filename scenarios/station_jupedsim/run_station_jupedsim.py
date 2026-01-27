"""
Run the JuPedSim station simulation.

This is the main entry point for the standalone JuPedSim implementation.
"""

import os
import sys
import time
import argparse
import random
from pathlib import Path
from typing import List, Optional
import jupedsim as jps

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from simulation import StationSimulation
from population_loader import create_agents_from_entrances
from movement_jupedsim import JuPedSimMovementProvider
from geometry_loader import load_entrance_areas, load_platform_areas
from live_viewer import LiveViewer

# Import common agent
sys.path.append(str(Path(__file__).parent.parent))
from common.station_agent import StationAgent


def main(enable_gui: bool = False, gui_update_interval: float = 1.0):
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
    
    # Create agents at entrances with random platform destinations (but don't spawn yet)
    num_agents = 60
    create_agents_from_entrances(
        simulation=sim.simulation,
        movement_provider=movement_provider,
        entrance_areas=entrance_areas,
        platform_stages=platform_stages,
        platform_journeys=platform_journeys,
        platform_areas=platform_areas,
        num_agents=num_agents,
        agent_list=agents,
        spawn_immediately=False  # Don't spawn agents yet
    )
    
    print(f"\nTotal agents created: {len(agents)}")
    print(f"Agents queued for gradual spawning")
    
    # Queue agents for spawning - randomize order so entrances are mixed
    agents_to_spawn = list(agents)
    random.shuffle(agents_to_spawn)
    print(f"Agent spawn order randomized across {len(entrance_areas)} entrances")
    
    spawn_interval = 2.0  # Spawn one agent every 2 seconds
    last_spawn_time = -spawn_interval  # Allow first spawn immediately
    
    # Initialize live viewer if requested
    viewer: Optional[LiveViewer] = None
    if enable_gui:
        print("\n[GUI] Initializing live viewer...")
        viewer = LiveViewer(
            walkable_areas=sim.walkable_areas,
            obstacles=sim.obstacles,
            platform_areas=platform_areas,
            update_interval=gui_update_interval
        )
        print(f"[GUI] Live viewer ready (updating every {gui_update_interval}s)")
    
    # Run simulation
    print("\n[4/4] Running simulation...")
    print("Press Ctrl+C to stop\n")
    
    max_iterations = 3600
    
    # Start timer for real execution time
    start_time = time.time()
    last_gui_update = 0.0
    
    try:
        while sim.iteration < max_iterations:
            # Spawn one agent if interval has passed and any are waiting
            if agents_to_spawn and (sim.get_simulation_time() - last_spawn_time >= spawn_interval):
                agent = agents_to_spawn.pop(0)
                last_spawn_time = sim.get_simulation_time()
                try:
                    agent.spawn()
                except Exception as e:
                    print(f"Failed to spawn {agent.id}: {e}")
            
            # Step simulation (even if no agents yet)
            if not sim.step():
                # Simulation ended - check if we're done
                if not agents_to_spawn and sim.simulation.agent_count() == 0:
                    break  # All agents spawned and completed
            
            sim_time = sim.get_simulation_time()
            agent_count = sim.simulation.agent_count()
            
            # Update all spawned agents
            for agent in agents:
                if agent.is_spawned:
                    agent.update(sim_time)
            
            # Print progress every 100 steps (5 seconds)
            if sim.iteration % 100 == 0:
                spawned_count = sum(1 for a in agents if a.is_spawned)
                print(f"t={sim_time:6.2f}s  agents={agent_count:3d}  spawned={spawned_count:3d}/{len(agents)}")
            
            # Update GUI at specified interval
            if viewer and (sim_time - last_gui_update) >= gui_update_interval:
                # Get current agent positions directly from JuPedSim
                agent_positions = []
                for agent in sim.simulation.agents():
                    pos = agent.position
                    agent_positions.append((pos[0], pos[1]))
                
                viewer.update(agent_positions, sim_time, agent_count)
                last_gui_update = sim_time
                
    except KeyboardInterrupt:
        print("\n\nSimulation interrupted by user")
    finally:
        # Close GUI if open
        if viewer:
            viewer.close()
    
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
    parser = argparse.ArgumentParser(description='Run JuPedSim station simulation')
    parser.add_argument('--gui', action='store_true', help='Enable real-time GUI visualization')
    parser.add_argument('--gui-interval', type=float, default=1.0, 
                        help='GUI update interval in seconds (default: 1.0)')
    
    args = parser.parse_args()
    
    main(enable_gui=args.gui, gui_update_interval=args.gui_interval)
