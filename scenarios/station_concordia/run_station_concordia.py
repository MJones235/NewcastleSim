"""
Main entry point for running Station Concordia simulations.

Usage:
    python -m scenarios.station_concordia.run_station_concordia

Or:
    python scenarios/station_concordia/run_station_concordia.py
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import yaml

# Setup path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scenarios.common.logger import get_logger  # noqa: E402

logger = get_logger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run Station Concordia evacuation simulation")
    parser.add_argument(
        "--config",
        type=str,
        default="scenarios/station_concordia/config/config.yaml",
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
        "--no-llm",
        action="store_true",
        help="Disable LLM (for testing)",
    )
    parser.add_argument(
        "--no-viewer",
        action="store_true",
        help="Don't launch the GUI viewer",
    )
    parser.add_argument(
        "--spatial-viewer",
        action="store_true",
        help="Launch spatial matplotlib viewer (shows agent positions on map)",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_file) as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded configuration from {config_path}")
    return config


def setup_language_model(config: dict, disable_llm: bool = False):
    """Setup the language model and embedder."""
    # Mock mode - return early before checking API keys
    if disable_llm:
        logger.warning("LLM disabled - using mock model")

        # Mock model for testing
        class MockModel:
            def sample_text(self, prompt: str, **kwargs) -> str:
                return "I will evacuate via the nearest exit."

        def mock_embedder(x):
            return [0.0] * 384

        model = MockModel()
        embedder = mock_embedder
        return model, embedder

    import os

    from dotenv import load_dotenv

    # Load .env file
    load_dotenv()

    azure_endpoint = os.getenv("AZURE_LLM_ENDPOINT")
    azure_key = os.getenv("AZURE_LLM_API_KEY")
    azure_model = os.getenv("AZURE_LLM_MODEL")

    if azure_endpoint and azure_key:
        logger.info(f"Using Azure LLM: {azure_model or 'serverless'}")

        try:
            import sentence_transformers

            from scenarios.station_concordia.core.azure_llm_concordia import AzureLLMConcordia

            # Create Azure LLM client designed for Concordia
            # Uses synchronous REST API calls to avoid async/sync conflicts
            llm_config = config.get("llm", {})
            model = AzureLLMConcordia(
                endpoint=azure_endpoint,
                api_key=azure_key,
                model=azure_model,
                temperature=llm_config.get("temperature", 0.7),
                max_retries=llm_config.get("max_retries", 3),
            )

            # Setup embedder (force CPU to avoid GPU compatibility issues)
            embedder_name = llm_config.get("embedder", "sentence-transformers/all-mpnet-base-v2")
            logger.info(f"Loading embedder: {embedder_name}...")
            st_model = sentence_transformers.SentenceTransformer(embedder_name, device="cpu")

            def embedder(x):
                return st_model.encode(x, show_progress_bar=False, device="cpu")

            logger.info("Embedder loaded successfully")
            logger.info("Azure LLM for Concordia initialized successfully")
            return model, embedder

        except ImportError as e:
            logger.error(f"Failed to import Azure provider: {e}")
            raise

    # No Azure configured - show error
    raise ValueError(
        "No LLM configured. Either:\n"
        "  1. Set Azure credentials in .env (AZURE_LLM_ENDPOINT, AZURE_LLM_API_KEY)\n"
        "  2. Use --no-llm flag for testing without LLM"
    )


def run_simulation(
    config: dict, model, embedder, launch_viewer: bool = True, launch_spatial: bool = False
):
    """
    Run the hybrid Concordia + JuPedSim simulation.

    Args:
        config: Configuration dictionary
        model: Language model
        embedder: Sentence embedder function
        launch_viewer: Whether to launch the GUI viewer before simulation starts
        launch_spatial: Whether to launch the spatial matplotlib viewer

    Returns:
        Tuple of (results dict, run_id string, decisions_file Path)
    """
    import subprocess
    import sys
    from pathlib import Path

    from scenarios.station_concordia.core.hybrid_simulation import HybridSimulationRunner
    from scenarios.station_concordia.core.jupedsim_integration import ConcordiaJuPedSimulation

    logger.info("Initializing simulation...")

    # Step 1: Setup JuPedSim simulation
    sim_config = config.get("simulation", {})
    dt = sim_config.get("dt", 0.05)
    use_simple_geometry = sim_config.get(
        "use_simple_geometry", True
    )  # Default to simple for testing

    if use_simple_geometry:
        logger.info("Using simple rectangular room geometry (100m x 100m)")
        jps_sim = ConcordiaJuPedSimulation(
            dt=dt,
            exit_radius=5.0,
            use_simple_geometry=True,
        )
    else:
        # Load station geometry from network files
        network_path = Path("scenarios/station_sim/network")
        logger.info(f"Loading real station geometry from {network_path}...")
        jps_sim = ConcordiaJuPedSimulation(
            network_path=network_path,
            dt=dt,
            exit_radius=10.0,
            use_simple_geometry=False,
        )

    # Step 2: Create agent configurations
    agent_config = config.get("agents", {})
    num_agents = agent_config.get("count", 1)

    agents_config = []
    station_layout = config.get("station", {})

    # Get spawn positions from walkable areas (not entrances, which may be outside)
    walkable_areas = jps_sim.walkable_areas
    if walkable_areas:
        # Use main walkable area for spawning
        main_area = list(walkable_areas.values())[0]
        logger.info(f"Spawning agents in main walkable area (area: {main_area.area:.1f} m²)")

        # Use a more restricted spawn area in the center of the room
        # to ensure agents are far from exits and have to walk there
        import random

        random.seed(42)  # Reproducible

        spawn_positions = []
        for _ in range(num_agents):
            # Spawn in central area: 30-70 for x and y (center 40x40 square)
            x = random.uniform(30.0, 70.0)
            y = random.uniform(30.0, 70.0)
            spawn_positions.append((x, y))

        logger.info(f"Generated {len(spawn_positions)} spawn positions in central area")
    else:
        logger.error("No walkable areas found!")
        raise RuntimeError("Cannot spawn agents without walkable areas")

    for i in range(num_agents):
        agent_id = f"agent_{i}"

        # Create agent config
        agent_cfg = {
            "id": agent_id,
            "name": f"Agent {i}",
            "personality_type": "ISTJ",  # Start with one personality
            "age": 35,
            "gender": "neutral",
            "risk_tolerance": "moderate",
            "initial_zone": "platform",
            "destination": "exit",
        }
        agents_config.append(agent_cfg)

        # Add agent to JuPedSim at spawn position
        start_pos = spawn_positions[i]
        jps_sim.add_agent(agent_id, start_pos)

    logger.info(f"Created {num_agents} agent configuration(s)")

    # Step 3: Initialize HybridSimulationRunner
    max_steps = sim_config.get("max_iterations", 200)  # Shorter for MVP
    decision_interval = sim_config.get("decision_interval", 5.0)

    # Setup output file path with unique run ID
    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    output_config = config.get("output", {})
    base_output_dir = Path(output_config.get("directory", "scenarios/station_concordia/output"))
    output_dir = base_output_dir / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    decisions_file = output_dir / "agent_decisions.json"

    logger.info(f"Run ID: {run_id}")
    logger.info(f"Output directory: {output_dir}")

    # Launch GUI viewer BEFORE simulation starts (if enabled)
    _viewer_process = None
    if launch_viewer:
        logger.info("Launching GUI viewer for live monitoring...")
        try:
            viewer_path = Path(__file__).parent.parent.parent / "tools" / "view_concordia_gui.py"
            _viewer_process = subprocess.Popen(
                [
                    sys.executable,
                    str(viewer_path),
                    "--output-file",
                    str(decisions_file.absolute()),
                    "--run-id",
                    run_id,
                ]
            )
            logger.info("GUI viewer launched - it will update as simulation runs")
        except Exception as e:
            logger.warning(f"Failed to launch GUI viewer: {e}")

    # Launch spatial viewer if requested
    _spatial_viewer_process = None
    if launch_spatial:
        logger.info("Launching spatial matplotlib viewer...")
        try:
            spatial_viewer_path = (
                Path(__file__).parent.parent.parent / "tools" / "view_concordia_spatial.py"
            )
            _spatial_viewer_process = subprocess.Popen(
                [
                    sys.executable,
                    str(spatial_viewer_path),
                    "--output-file",
                    str(decisions_file.absolute()),
                ]
            )
            logger.info("Spatial viewer launched - shows agent positions on map")
        except Exception as e:
            logger.warning(f"Failed to launch spatial viewer: {e}")

    logger.info("Creating HybridSimulationRunner...")

    try:
        runner = HybridSimulationRunner(
            jupedsim_simulation=jps_sim,
            agents_config=agents_config,
            station_layout=station_layout,
            language_model=model,
            embedder=embedder,
            decision_interval=decision_interval,
            max_steps=max_steps,
            output_file=decisions_file,
        )
        logger.info("HybridSimulationRunner initialized")
    except Exception as e:
        logger.error(f"FATAL ERROR during HybridSimulationRunner initialization: {e}")
        import traceback

        traceback.print_exc()
        raise

    # Load events from configuration
    events_config = config.get("events", [])
    for event in events_config:
        runner.event_history.append(
            {
                "time": event.get("time", 0.0),
                "message": event.get("message", ""),
            }
        )

    if events_config:
        logger.info(f"Loaded {len(events_config)} events from configuration")
    else:
        logger.warning("No events defined in configuration")

    # Step 4: Run simulation
    logger.info("Starting simulation run...")
    results = runner.run()

    # Step 5: Save final results
    runner.save_results(decisions_file)

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
        # Load configuration
        config = load_config(args.config)

        # Apply command-line overrides
        if args.agents:
            config["agents"]["count"] = args.agents
        if args.max_steps:
            config["simulation"]["max_iterations"] = args.max_steps
        if args.output_dir:
            config["output"]["directory"] = args.output_dir

        # Setup language model
        model, embedder = setup_language_model(config, disable_llm=args.no_llm)

        # Run simulation (with viewers if enabled)
        # Run simulation (with viewers if enabled)
        results, run_id, decisions_file = run_simulation(
            config,
            model,
            embedder,
            launch_viewer=not args.no_viewer,
            launch_spatial=args.spatial_viewer,
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
