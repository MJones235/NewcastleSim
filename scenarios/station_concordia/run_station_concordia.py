"""
Main entry point for running Station Concordia simulations.

Usage:
    python -m scenarios.station_concordia.run_station_concordia

Or:
    python scenarios/station_concordia/run_station_concordia.py
"""

import argparse
import signal
import sys
from pathlib import Path

# Setup path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scenarios.common.logger import get_logger  # noqa: E402
from scenarios.station_concordia.config.config_loader import ConfigLoader  # noqa: E402
from scenarios.station_concordia.reporting.results_writer import ResultsWriter  # noqa: E402
from scenarios.station_concordia.setup.agent_manager import AgentManager  # noqa: E402
from scenarios.station_concordia.setup.jupedsim_setup import JuPedSimSetup  # noqa: E402
from scenarios.station_concordia.setup.llm_setup import LLMSetup  # noqa: E402
from scenarios.station_concordia.setup.output_manager import OutputManager  # noqa: E402
from scenarios.station_concordia.setup.simulation_runner_factory import (  # noqa: E402
    SimulationRunnerFactory,
)
from scenarios.station_concordia.setup.station_layout_builder import (  # noqa: E402
    StationLayoutBuilder,
)
from scenarios.station_concordia.visualization.video_generation_helper import (  # noqa: E402
    VideoGenerationHelper,
)
from scenarios.station_concordia.visualization.viewer_launcher import ViewerLauncher  # noqa: E402

logger = get_logger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run Station Concordia evacuation simulation")
    parser.add_argument(
        "--config",
        type=str,
        default="scenarios/station_concordia/config/config_monument.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--agents",
        type=int,
        default=None,
        help="Number of agents (overrides config)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum simulation steps (overrides config)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (overrides config)",
    )
    parser.add_argument(
        "--no-viewer",
        action="store_true",
        help="Don't launch the GUI viewer",
    )
    parser.add_argument(
        "--spatial-viewer",
        action="store_true",
        default=True,
        help="Launch spatial matplotlib viewer (shows agent positions on map)",
    )
    parser.add_argument(
        "--no-spatial-viewer",
        action="store_true",
        help="Don't launch spatial matplotlib viewer",
    )
    parser.add_argument(
        "--generate-video",
        action="store_true",
        help="Generate MP4 video after simulation (requires ffmpeg)",
    )
    parser.add_argument(
        "--video-fps",
        type=int,
        default=20,
        help="Video frames per second (default: 20)",
    )
    parser.add_argument(
        "--video-speedup",
        type=float,
        default=1.0,
        help="Video speed multiplier (default: 1.0 = real-time)",
    )
    return parser.parse_args()


def run_simulation(
    config: dict, model, embedder, launch_viewer: bool = True, launch_spatial: bool = True
):
    """
    Run the hybrid Concordia + pedestrian simulation.

    This function orchestrates the entire simulation workflow:
    1. Setup pedestrian simulation with geometry
    2. Build station layout
    3. Create and populate agents
    4. Setup output directory
    5. Launch viewers
    6. Create and configure simulation runner
    7. Run simulation
    8. Save results

    Args:
        config: Configuration dictionary
        model: Language model
        embedder: Sentence embedder function
        launch_viewer: Whether to launch the GUI viewer before simulation starts
        launch_spatial: Whether to launch the spatial matplotlib viewer

    Returns:
        Tuple of (results dict, run_id string, decisions_file Path)
    """
    logger.info("Initializing simulation...")

    # Store runner for signal handler access
    runner = None

    # Step 1: Setup pedestrian simulation
    jps_sim = JuPedSimSetup.create_simulation(config)

    # Step 2: Build station layout from geometry
    station_layout = StationLayoutBuilder.build_layout(jps_sim, config)

    # Step 3: Create and populate agents (handles spawn positions, configs, and adding to simulation)
    agents_config = AgentManager.create_and_populate_agents(jps_sim, config)

    # Step 4: Setup output directory and files
    run_id, output_dir, decisions_file = OutputManager.setup_output_directory(config)

    # Step 5: Launch viewers BEFORE simulation starts (if enabled)
    network_path = jps_sim.network_path
    _viewer_process, _spatial_viewer_process = ViewerLauncher.launch_viewers(
        decisions_file=decisions_file,
        run_id=run_id,
        network_path=network_path,
        launch_gui=launch_viewer,
        launch_spatial=launch_spatial,
    )

    # Step 6: Create and configure simulation runner
    runner = SimulationRunnerFactory.create_runner(
        jps_sim=jps_sim,
        agents_config=agents_config,
        station_layout=station_layout,
        model=model,
        embedder=embedder,
        decisions_file=decisions_file,
        config=config,
    )

    # Setup signal handler for graceful shutdown on Ctrl+C
    def signal_handler(signum, frame):
        logger.warning("\n⚠️  Simulation interrupted! Saving partial results...")
        if runner:
            runner.cleanup()
        logger.info("Partial results saved. Exiting.")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # Step 7: Run simulation
    logger.info("Starting simulation run...")
    try:
        results = runner.run()
    except KeyboardInterrupt:
        # Redundant catch in case signal handler doesn't work
        logger.warning("\n⚠️  Simulation interrupted! Saving partial results...")
        runner.cleanup()
        logger.info("Partial results saved. Exiting.")
        sys.exit(0)

    # Step 8: Save final results
    # Get agent levels for multi-level simulations
    agent_levels = None
    if hasattr(runner.jps_sim, "agent_levels"):
        agent_levels = runner.jps_sim.agent_levels

    # Log cache optimization summary
    if hasattr(runner, "decision_processor"):
        runner.decision_processor.log_cache_summary()

    ResultsWriter.save_final_results(
        decisions_file,
        runner.agent_decisions,
        runner.jps_sim.get_all_agent_positions(),
        runner.current_sim_time,
        runner.event_manager.event_history,
        runner.event_manager.blocked_exits,
        runner.message_system.message_history,
        runner.wait_events,
        runner.decision_interval,
        runner.max_steps,
        len(runner.concordia_agents),
        runner.perf_timer.report(),
        runner.llm_provider,
        agent_levels,
    )
    logger.info(f"Simulation complete. Results saved to {output_dir}")

    return results, run_id, decisions_file


def main():
    """Main entry point."""
    import time

    script_start = time.time()

    args = parse_args()

    logger.info("=" * 60)
    logger.info("Station Concordia - Evacuation Simulation")
    logger.info("=" * 60)

    try:
        # Load and validate configuration (with CLI overrides)
        config = ConfigLoader.load_and_validate(
            config_path=args.config,
            agents=args.agents,
            max_steps=args.max_steps,
            output_dir=args.output_dir,
        )

        # Setup language model
        model, embedder = LLMSetup.setup_language_model(config)

        # Run simulation (with viewers if enabled)
        results, run_id, decisions_file = run_simulation(
            config,
            model,
            embedder,
            launch_viewer=not args.no_viewer,
            launch_spatial=args.spatial_viewer and not args.no_spatial_viewer,
        )

        # Generate video if requested
        if args.generate_video:
            network_path = Path(
                config.get("simulation", {}).get("network_path", "scenarios/station_sim/network")
            )
            VideoGenerationHelper.generate_simulation_video(
                decisions_file=decisions_file,
                run_id=run_id,
                network_path=network_path,
                fps=args.video_fps,
                speedup=args.video_speedup,
            )

        # Log results
        logger.info("=" * 60)
        logger.info("Simulation Results:")
        logger.info("=" * 60)
        for key, value in results.items():
            logger.info(f"  {key}: {value}")

        logger.info("=" * 60)
        logger.info("Simulation complete!")
        logger.info("=" * 60)

        total_time = time.time() - script_start
        logger.info(f"Total execution time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Simulation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
