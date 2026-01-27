"""
Run the JuPedSim station simulation.

This is the main entry point for the standalone JuPedSim implementation.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from scenarios.station_jupedsim.simulation import StationSimulation
from scenarios.station_jupedsim.simulation_setup import setup_evacuation_exits, setup_platform_stages, load_geometry
from scenarios.station_jupedsim.simulation_runner import SimulationRunner
from scenarios.station_jupedsim.population_loader import create_agents_from_entrances
from scenarios.station_jupedsim.movement_jupedsim import JuPedSimMovementProvider
from scenarios.station_jupedsim.event_system import EventManager
from scenarios.station_jupedsim.visualization.live_viewer import LiveViewer
from scenarios.common.station_agent import StationAgent


class SimulationError(Exception):
    """Custom exception for simulation errors."""
    pass


def main(enable_gui: bool = False, gui_update_interval: float = 1.0, events_file: Optional[str] = None) -> int:
    """Main simulation setup and execution.
    
    Returns:
        0 on success, 1 on error
    """
    
    try:
        # Setup paths
        scenario_dir = Path(__file__).parent
        network_path = scenario_dir / ".." / "station_sim" / "network"
        output_dir = scenario_dir / "output"
        
        # Validate network path exists
        if not network_path.exists():
            raise SimulationError(f"Network directory not found: {network_path}")
        
        walking_areas_file = network_path / "walking_areas.add.xml"
        if not walking_areas_file.exists():
            raise SimulationError(f"Required file not found: {walking_areas_file}")
        
        # Create output directory
        try:
            output_dir.mkdir(exist_ok=True)
        except PermissionError:
            raise SimulationError(f"Cannot create output directory (permission denied): {output_dir}")
        except Exception as e:
            raise SimulationError(f"Cannot create output directory: {e}")
        
        trajectory_file = output_dir / "trajectory.db"
        
        print("=" * 60)
        print("JuPedSim Station Simulation")
        print("=" * 60)
        
        # Initialize simulation
        print("\n[1/5] Initializing simulation...")
        try:
            sim = StationSimulation(str(network_path), dt=0.05, output_file=str(trajectory_file))
        except FileNotFoundError as e:
            raise SimulationError(f"Failed to load network files: {e}")
        except Exception as e:
            raise SimulationError(f"Failed to initialize simulation: {e}")
        
        print(f"  Trajectory output: {trajectory_file}")
    
        # Load entrance and platform geometry
        print("\n[2/5] Loading geometry...")
        try:
            entrance_areas, platform_areas = load_geometry(network_path)
            if not entrance_areas:
                raise SimulationError("No entrance areas found in network files")
            if not platform_areas:
                raise SimulationError("No platform areas found in network files")
        except FileNotFoundError as e:
            raise SimulationError(f"Geometry file not found: {e}")
        except Exception as e:
            raise SimulationError(f"Failed to load geometry: {e}")
        
        # Setup evacuation exits at entrances
        print("\n[3/5] Setting up evacuation exits...")
        try:
            evacuation_exits, evacuation_journeys = setup_evacuation_exits(sim, entrance_areas)
            if not evacuation_exits:
                raise SimulationError("Failed to create any evacuation exits")
        except Exception as e:
            raise SimulationError(f"Failed to setup evacuation exits: {e}")
        
        # Setup platform stages
        try:
            platform_stages, platform_journeys = setup_platform_stages(sim, platform_areas)
            if not platform_stages:
                raise SimulationError("Failed to create any platform stages")
        except Exception as e:
            raise SimulationError(f"Failed to setup platform stages: {e}")
    
        # Create agents
        print("\n[4/5] Creating agent population...")
        agents: List[StationAgent] = []
        
        # Create movement provider for JuPedSim
        try:
            movement_provider = JuPedSimMovementProvider(sim.simulation, sim.zones)
        except Exception as e:
            raise SimulationError(f"Failed to create movement provider: {e}")
        
        # Store evacuation exits in movement provider for agent access
        movement_provider.evacuation_journeys = evacuation_journeys
        movement_provider.evacuation_exits = evacuation_exits
        
        # Create agents at entrances with random platform destinations (but don't spawn yet)
        num_agents = 60
        try:
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
            
            if not agents:
                raise SimulationError("Failed to create any agents")
        except Exception as e:
            raise SimulationError(f"Failed to create agents: {e}")
        
        print(f"\nTotal agents created: {len(agents)}")
        print(f"Agents queued for gradual spawning")
        
        # Initialize event manager
        try:
            event_manager = EventManager(events_file)
        except FileNotFoundError as e:
            raise SimulationError(f"Events file not found: {e}")
        except Exception as e:
            raise SimulationError(f"Failed to load events: {e}")
        if event_manager.events:
            print(f"\n[EVENTS] Event system active with {len(event_manager.events)} scheduled events:")
            for event in event_manager.events:
                print(f"  t={event.time:6.1f}s: {event.action} - '{event.value}'")
        
        # Initialize live viewer if requested
        viewer = None
        if enable_gui:
            print("\n[GUI] Initializing live viewer...")
            try:
                viewer = LiveViewer(
                    walkable_areas=sim.walkable_areas,
                    obstacles=sim.obstacles,
                    platform_areas=platform_areas,
                    update_interval=gui_update_interval
                )
                print(f"[GUI] Live viewer ready (updating every {gui_update_interval}s)")
            except Exception as e:
                print(f"WARNING: Failed to initialize GUI: {e}")
                print("Continuing without live visualization...")
                viewer = None
        
        # Create simulation runner
        print("\n[5/5] Running simulation...")
        try:
            runner = SimulationRunner(
                sim=sim,
                agents=agents,
                event_manager=event_manager,
                max_iterations=3600,
                spawn_interval=2.0
            )
        except Exception as e:
            raise SimulationError(f"Failed to create simulation runner: {e}")
        
        # Run simulation
        try:
            stats = runner.run(
                enable_gui=enable_gui,
                gui_update_interval=gui_update_interval,
                viewer=viewer
            )
        except KeyboardInterrupt:
            print("\n\nSimulation interrupted by user")
            return 1
        except Exception as e:
            raise SimulationError(f"Simulation execution failed: {e}")
        
        # Save events
        try:
            runner.save_events(output_dir)
        except Exception as e:
            print(f"WARNING: Failed to save events: {e}")
        
        # Print summary
        print_summary(stats, trajectory_file)
        
        # Run visualization automatically
        launch_visualization(trajectory_file, network_path)
        
        return 0
        
    except SimulationError as e:
        print(f"\n\n❌ SIMULATION ERROR: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"\n\n❌ UNEXPECTED ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


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
        from scenarios.station_jupedsim.visualization import visualize
        visualize.visualize_simulation(str(trajectory_file), str(network_path))
    except Exception as e:
        print(f"Could not launch visualization: {e}")
        print("\nTo visualize manually, run:")
        print(f"  .venv/bin/python scenarios/station_jupedsim/visualization/visualize.py {trajectory_file}")


if __name__ == "__main__":
    # This module can be imported, but for standalone execution,
    # use the run_jupedsim_station.py script in the project root instead.
    # That script properly handles Python path setup.
    print("ERROR: Please run from project root using:")
    print("  python run_jupedsim_station.py [--gui] [--events events.csv]")
    import sys
    sys.exit(1)
