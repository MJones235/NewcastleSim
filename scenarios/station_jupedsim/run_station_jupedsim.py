"""
Run the JuPedSim station simulation.

This is the main orchestration script for the standalone JuPedSim implementation.
Handles the complete simulation lifecycle from setup through execution to visualization.

Workflow:
    1. Load configuration (from YAML or defaults)
    2. Initialize JuPedSim simulation with network geometry
    3. Setup evacuation exits and platform stages
    4. Create agent population with decision-making capabilities
    5. Run simulation with gradual spawning and event system
    6. Generate visualization of results

Usage:
    Called from run_jupedsim_station.py in project root:
        python run_jupedsim_station.py --gui --events events.csv

Note:
    This module should not be run directly. Use run_jupedsim_station.py
    which properly handles Python path setup.
"""

import sys
from pathlib import Path

from scenarios.common.logger import get_logger, setup_logger
from scenarios.common.station_agent import StationAgent
from scenarios.station_jupedsim.config import Config
from scenarios.station_jupedsim.core import (
    ConsoleObserver,
    EventManager,
    GUIObserver,
    JuPedSimMovementProvider,
    SimulationObserver,
    SimulationRunner,
    StationSimulation,
    create_agents_from_entrances,
    load_geometry,
    setup_evacuation_exits,
    setup_platform_stages,
)
from scenarios.station_jupedsim.visualization.live_viewer import LiveViewer


class SimulationError(Exception):
    """Custom exception for simulation errors."""

    pass


def main(config: Config | None = None) -> int:
    """Main simulation setup and execution.

    Orchestrates the complete simulation workflow:
    1. Initialize JuPedSim simulation with geometry
    2. Load entrance and platform geometry
    3. Setup evacuation exits and platform stages
    4. Create agent population with movement provider
    5. Run simulation with event system and optional GUI

    Args:
        config: Configuration object. If None, uses defaults.

    Returns:
        0 on success, 1 on error

    Raises:
        SimulationError: For any simulation setup or execution errors
    """

    # Use provided config or load defaults
    if config is None:
        config = Config()

    # Setup logging
    output_dir = Path(config.paths.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    log_file = output_dir / "simulation.log"
    setup_logger(name="scenarios.station_jupedsim", log_file=log_file)
    logger = get_logger("scenarios.station_jupedsim")

    logger.info("=" * 60)
    logger.info("JuPedSim Station Simulation")
    logger.info("=" * 60)
    logger.debug(f"Log file: {log_file}")

    try:
        # Setup paths
        network_path = Path(config.paths.network_dir)
        output_dir = Path(config.paths.output_dir)

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
            raise SimulationError(
                f"Cannot create output directory (permission denied): {output_dir}"
            )
        except Exception as e:
            raise SimulationError(f"Cannot create output directory: {e}")

        trajectory_file = output_dir / "trajectory.db"

        print("=" * 60)
        print("JuPedSim Station Simulation")
        print("=" * 60)

        # Initialize LLM if enabled
        if config.llm.enabled:
            print("\n[LLM] Initializing Azure AI model...")
            try:
                from scenarios.common.decision_makers.llm_decision_maker import LLMDecisionMaker

                LLMDecisionMaker.initialize_llm(
                    endpoint=config.llm.endpoint,
                    api_key=config.llm.api_key,
                    model=config.llm.model,
                )
                model_name = config.llm.model or "serverless endpoint"
                print(f"[LLM] ✓ Connected to model: {model_name}")
                logger.info("LLM provider initialized successfully")
            except Exception as e:
                raise SimulationError(f"Failed to initialize LLM provider: {e}")

        # Initialize simulation
        print("\n[1/5] Initializing simulation...")
        try:
            sim = StationSimulation(
                str(network_path), dt=config.simulation.dt, output_file=str(trajectory_file)
            )
        except FileNotFoundError as e:
            raise SimulationError(f"Failed to load network files: {e}")
        except Exception as e:
            raise SimulationError(f"Failed to initialize simulation: {e}")

        print(f"  Trajectory output: {trajectory_file}")

        # Load entrance and platform geometry
        print("\n[2/5] Loading geometry...")
        try:
            entrance_areas, platform_areas, walkable_areas = load_geometry(network_path)
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
            evacuation_exits, evacuation_journeys = setup_evacuation_exits(
                sim, entrance_areas, exit_radius=config.simulation.exit_radius
            )
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
        agents: list[StationAgent] = []

        # Create movement provider for JuPedSim
        try:
            movement_provider = JuPedSimMovementProvider(sim.simulation, sim.zones)
        except Exception as e:
            raise SimulationError(f"Failed to create movement provider: {e}")

        # Store evacuation exits in movement provider for agent access
        movement_provider.evacuation_journeys = evacuation_journeys
        movement_provider.evacuation_exits = evacuation_exits

        # Create agents based on spawn mode
        try:
            if config.simulation.spawn_mode == "random":
                # Spawn all agents immediately at random positions throughout the station
                from scenarios.station_jupedsim.core import create_agents_in_walkable_areas

                create_agents_in_walkable_areas(
                    simulation=sim.simulation,
                    movement_provider=movement_provider,
                    walkable_areas=walkable_areas,
                    platform_stages=platform_stages,
                    platform_journeys=platform_journeys,
                    platform_areas=platform_areas,
                    num_agents=config.simulation.num_agents,
                    agent_list=agents,
                    use_llm=config.llm.enabled,
                )
                print(f"Spawned {len(agents)} agents randomly throughout station")
            else:
                # Gradual spawning from entrances (original behavior)
                create_agents_from_entrances(
                    simulation=sim.simulation,
                    movement_provider=movement_provider,
                    entrance_areas=entrance_areas,
                    platform_stages=platform_stages,
                    platform_journeys=platform_journeys,
                    platform_areas=platform_areas,
                    num_agents=config.simulation.num_agents,
                    agent_list=agents,
                    spawn_immediately=False,  # Don't spawn agents yet
                    use_llm=config.llm.enabled,  # Use LLM decision maker if enabled
                )
                print("Agents queued for gradual spawning from entrances")

            if not agents:
                raise SimulationError("Failed to create any agents")
        except Exception as e:
            raise SimulationError(f"Failed to create agents: {e}")

        print(f"\nTotal agents created: {len(agents)}")

        # Initialize event manager
        try:
            event_manager = EventManager(config.paths.events_file)
        except FileNotFoundError as e:
            raise SimulationError(f"Events file not found: {e}")
        except Exception as e:
            raise SimulationError(f"Failed to load events: {e}")
        if event_manager.events:
            print(
                f"\n[EVENTS] Event system active with {len(event_manager.events)} scheduled events:"
            )
            for event in event_manager.events:
                print(f"  t={event.time:6.1f}s: {event.action} - '{event.value}'")

        # Setup observers (GUI and console)
        observers: list[SimulationObserver] = []

        # Add console observer for progress updates
        observers.append(ConsoleObserver(update_interval=100))

        # Add GUI observer if requested
        if config.visualization.enable_gui:
            print("\n[GUI] Initializing live viewer...")
            try:
                viewer = LiveViewer(
                    walkable_areas=sim.zones,
                    obstacles=sim.obstacles,
                    platform_areas=platform_areas,
                    update_interval=config.visualization.gui_update_interval,
                )
                gui_observer = GUIObserver(viewer, config.visualization.gui_update_interval)
                observers.append(gui_observer)
                print(
                    f"[GUI] Live viewer ready (updating every {config.visualization.gui_update_interval}s)"
                )
            except Exception as e:
                print(f"WARNING: Failed to initialize GUI: {e}")
                print("Continuing without live visualization...")

        # Create simulation runner with observers
        print("\n[5/5] Running simulation...")
        try:
            runner = SimulationRunner(
                sim=sim,
                agents=agents,
                event_manager=event_manager,
                max_iterations=config.simulation.max_iterations,
                spawn_interval=config.simulation.spawn_interval,
                observers=observers,
            )

            # Enable LLM processing if configured
            if config.llm.enabled:
                runner.enable_llm()
                print("[LLM] ✓ LLM decision-making enabled for agents")

        except Exception as e:
            raise SimulationError(f"Failed to create simulation runner: {e}")

        # Run simulation
        try:
            stats = runner.run()
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

        # Run visualization automatically if enabled
        if config.visualization.enable_post_run_viz:
            launch_visualization(trajectory_file, network_path)
        else:
            print("\n[Visualization disabled - skipping post-run animation]")

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
    """Print simulation summary statistics.

    Displays key metrics about the completed simulation including:
    - Total iterations and simulation time
    - Real execution time and speed factor
    - Agent statistics (remaining, exited)
    - Output file location

    Args:
        stats: Dictionary containing simulation statistics from runner
        trajectory_file: Path to the trajectory output database
    """
    print("\n" + "=" * 60)
    print("Simulation Complete")
    print("=" * 60)
    print(f"Total iterations: {stats['iterations']}")
    print(f"Simulation time: {stats['simulated_time']:.2f}s")
    print(f"Real execution time: {stats['real_time']:.2f}s")
    if stats["real_time"] > 0:
        print(f"Speed factor: {stats['simulated_time'] / stats['real_time']:.2f}x realtime")
    print(f"Remaining agents: {stats['remaining_agents']}")
    print(f"Agents who exited: {stats['total_agents'] - stats['remaining_agents']}")
    print(f"\nTrajectory saved to: {trajectory_file}")


def launch_visualization(trajectory_file: Path, network_path: Path):
    """Launch post-run visualization of simulation results.

    Attempts to automatically start the trajectory visualization tool
    to replay the completed simulation. Falls back to printing manual
    instructions if automatic launch fails.

    Args:
        trajectory_file: Path to the trajectory database file
        network_path: Path to the network directory with geometry files
    """
    print("\n" + "=" * 60)
    print("Launching Visualization")
    print("=" * 60)

    try:
        from scenarios.station_jupedsim.visualization import visualize

        visualize.visualize_simulation(str(trajectory_file), str(network_path))
    except Exception as e:
        print(f"Could not launch visualization: {e}")
        print("\nTo visualize manually, run:")
        print(
            f"  .venv/bin/python scenarios/station_jupedsim/visualization/visualize.py {trajectory_file}"
        )


if __name__ == "__main__":
    # This module can be imported, but for standalone execution,
    # use the run_jupedsim_station.py script in the project root instead.
    # That script properly handles Python path setup.
    print("ERROR: Please run from project root using:")
    print("  python run_jupedsim_station.py [--gui] [--events events.csv]")
    import sys

    sys.exit(1)
