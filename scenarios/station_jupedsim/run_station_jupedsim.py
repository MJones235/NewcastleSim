"""
Run the JuPedSim station simulation.

This is the main entry point for the standalone JuPedSim implementation.
"""

import sys
import argparse
from pathlib import Path
from typing import List

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from simulation import StationSimulation
from simulation_setup import setup_evacuation_exits, setup_platform_stages, load_geometry
from simulation_runner import SimulationRunner
from population_loader import create_agents_from_entrances
from movement_jupedsim import JuPedSimMovementProvider
from event_system import EventManager
from visualization import LiveViewer

# Import common agent
sys.path.append(str(Path(__file__).parent.parent))
from common.station_agent import StationAgent


def main(enable_gui: bool = False, gui_update_interval: float = 1.0, events_file: str = None):
    """Main simulation setup and execution."""
    
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
    print("\n[1/5] Initializing simulation...")
    sim = StationSimulation(str(network_path), dt=0.05, output_file=str(trajectory_file))
    print(f"  Trajectory output: {trajectory_file}")
    
    # Load entrance and platform geometry
    print("\n[2/5] Loading geometry...")
    entrance_areas, platform_areas = load_geometry(network_path)
    
    # Setup evacuation exits at entrances
    print("\n[3/5] Setting up evacuation exits...")
    evacuation_exits, evacuation_journeys = setup_evacuation_exits(sim, entrance_areas)
    
    # Setup platform stages
    platform_stages, platform_journeys = setup_platform_stages(sim, platform_areas)
    
    # Create agents
    print("\n[4/5] Creating agent population...")
    agents: List[StationAgent] = []
    
    # Create movement provider for JuPedSim
    movement_provider = JuPedSimMovementProvider(sim.simulation, sim.zones)
    
    # Store evacuation exits in movement provider for agent access
    movement_provider.evacuation_journeys = evacuation_journeys
    movement_provider.evacuation_exits = evacuation_exits
    
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
    
    # Initialize event manager
    event_manager = EventManager(events_file)
    if event_manager.events:
        print(f"\n[EVENTS] Event system active with {len(event_manager.events)} scheduled events:")
        for event in event_manager.events:
            print(f"  t={event.time:6.1f}s: {event.action} - '{event.value}'")
    
    # Initialize live viewer if requested
    viewer = None
    if enable_gui:
        print("\n[GUI] Initializing live viewer...")
        viewer = LiveViewer(
            walkable_areas=sim.walkable_areas,
            obstacles=sim.obstacles,
            platform_areas=platform_areas,
            update_interval=gui_update_interval
        )
        print(f"[GUI] Live viewer ready (updating every {gui_update_interval}s)")
    
    # Create simulation runner
    print("\n[5/5] Running simulation...")
    runner = SimulationRunner(
        sim=sim,
        agents=agents,
        event_manager=event_manager,
        max_iterations=3600,
        spawn_interval=2.0
    )
    
    # Run simulation
    stats = runner.run(
        enable_gui=enable_gui,
        gui_update_interval=gui_update_interval,
        viewer=viewer
    )
    
    # Save events
    runner.save_events(output_dir)
    
    # Print summary
    print_summary(stats, trajectory_file)
    
    # Run visualization automatically
    launch_visualization(trajectory_file, network_path)


def print_summary(stats: dict, trajectory_file: Path):
    """Print simulation summary statistics."""
    print("\n" + "=" * 60)
    print("Simulation Complete")
    print("=" * 60)
    print(f"Total iterations: {stats['iterations']}")
    print(f"Simulation time: {stats['simulated_time']:.2f}s")
    print(f"Real execution time: {stats['real_time']:.2f}s")
    if stats['real_time'] > 0:
        print(f"Speed factor: {stats['simulated_time'] / stats['real_time']:.2f}x realtime")
    print(f"Remaining agents: {stats['remaining_agents']}")
    print(f"Agents who exited: {stats['total_agents'] - stats['remaining_agents']}")
    print(f"\nTrajectory saved to: {trajectory_file}")


def launch_visualization(trajectory_file: Path, network_path: Path):
    """Launch post-run visualization."""
    print("\n" + "=" * 60)
    print("Launching Visualization")
    print("=" * 60)
    
    try:
        from visualization import visualize
        visualize.visualize_simulation(str(trajectory_file), str(network_path))
    except Exception as e:
        print(f"Could not launch visualization: {e}")
        print("\nTo visualize manually, run:")
        print(f"  .venv/bin/python scenarios/station_jupedsim/visualization/visualize.py {trajectory_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run JuPedSim station simulation')
    parser.add_argument('--gui', action='store_true', help='Enable real-time GUI visualization')
    parser.add_argument('--gui-interval', type=float, default=1.0, 
                        help='GUI update interval in seconds (default: 1.0)')
    
    # Default to events.csv if it exists
    scenario_dir = Path(__file__).parent
    default_events_file = scenario_dir / "events.csv"
    default_events = str(default_events_file) if default_events_file.exists() else None
    
    parser.add_argument('--events', type=str, default=default_events,
                        help=f'Path to events CSV file for mid-simulation injections (default: {default_events})')
    
    args = parser.parse_args()
    
    main(enable_gui=args.gui, gui_update_interval=args.gui_interval, events_file=args.events)
