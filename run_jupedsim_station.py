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

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run JuPedSim station simulation')
    parser.add_argument('--gui', action='store_true', help='Enable real-time GUI visualization')
    parser.add_argument('--gui-interval', type=float, default=1.0, 
                        help='GUI update interval in seconds (default: 1.0)')
    
    # Default to events.csv if it exists
    scenario_dir = project_root / "scenarios" / "station_jupedsim"
    default_events_file = scenario_dir / "events.csv"
    default_events = str(default_events_file) if default_events_file.exists() else None
    
    parser.add_argument('--events', type=str, default=default_events,
                        help=f'Path to events CSV file for mid-simulation injections (default: {default_events})')
    
    args = parser.parse_args()
    
    main(enable_gui=args.gui, gui_update_interval=args.gui_interval, events_file=args.events)
