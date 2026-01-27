#!/usr/bin/env python3
"""
Entry point script for JuPedSim station simulation.

This wrapper ensures the project root is in the Python path,
allowing clean absolute imports throughout the codebase.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Now import and run the main function
from scenarios.station_jupedsim.run_station_jupedsim import main
from scenarios.station_jupedsim.config import Config, load_config

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run JuPedSim station simulation')
    
    # Configuration file
    parser.add_argument('--config', type=str, 
                        help='Path to YAML configuration file (default: uses built-in defaults)')
    
    # Override options
    parser.add_argument('--gui', action='store_true', 
                        help='Enable real-time GUI visualization (overrides config)')
    parser.add_argument('--no-gui', action='store_true',
                        help='Disable real-time GUI visualization (overrides config)')
    parser.add_argument('--num-agents', type=int,
                        help='Number of agents to create (overrides config)')
    parser.add_argument('--events', type=str,
                        help='Path to events CSV file (overrides config)')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        config = load_config(args.config)
    else:
        # Try to load default config.yaml if it exists
        default_config = project_root / "scenarios" / "station_jupedsim" / "config" / "config.yaml"
        config = load_config(str(default_config) if default_config.exists() else None)
    
    # Apply command-line overrides
    if args.gui:
        config.visualization.enable_gui = True
    if args.no_gui:
        config.visualization.enable_gui = False
    if args.num_agents:
        config.simulation.num_agents = args.num_agents
    if args.events:
        config.paths.events_file = args.events
    
    # Validate configuration
    try:
        config.validate()
    except Exception as e:
        print(f"Configuration error: {e}")
        sys.exit(1)
    
    # Run simulation
    sys.exit(main(config))
