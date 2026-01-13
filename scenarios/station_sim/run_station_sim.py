#!/usr/bin/env python3
"""
Run the Newcastle Central Station simulation.
Simulates trains arriving/departing with passengers using SUMO + JuPedSim.
"""

import traci
import sumolib
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from scenarios.station_sim.passenger_generator import PassengerGenerator


def setup_scenario(regenerate_passengers=True):
    """Set up the scenario files."""
    print("=" * 60)
    print("Setting up Newcastle Station simulation")
    print("=" * 60)
    
    if regenerate_passengers:
        print("\nGenerating passenger trips...")
        generator = PassengerGenerator(
            network_file='scenarios/station_sim/network/net.net.xml',
            stops_file='scenarios/station_sim/network/osm_stops.add.xml',
            pt_routes_file='scenarios/station_sim/network/osm_pt.rou.xml'
        )
        
        # Generate passengers with reduced numbers to avoid overwhelming JuPedSim
        generator.generate_arriving_passengers(
            passengers_per_train=50,
            output_file='scenarios/station_sim/network/passengers_arriving.rou.xml'
        )
        generator.generate_departing_passengers(
            passengers_per_train=50,
            output_file='scenarios/station_sim/network/passengers_departing.rou.xml'
        )
    
    print("\nScenario setup complete!")


def run_simulation(use_gui=True, max_time=3 * 3600):
    """
    Run the station simulation.
    
    Args:
        use_gui: Whether to use SUMO GUI or headless
        max_time: Maximum simulation time in seconds (default 3 hours)
    """
    sumo_binary = sumolib.checkBinary('sumo-gui' if use_gui else 'sumo')
    
    print("\n" + "=" * 60)
    print("Starting SUMO simulation...")
    print("=" * 60)
    print(f"Mode: {'GUI' if use_gui else 'Headless'}")
    print(f"Duration: {max_time/3600:.1f} hours")
    
    # Build SUMO command
    sumo_cmd = [
        sumo_binary,
        '--net-file', 'scenarios/station_sim/network/net.net.xml',
        '--additional-files', ','.join([
            'scenarios/station_sim/network/osm_stops.add.xml',  # Use original stops with lines attribute
            'scenarios/station_sim/network/osm.add.xml',  # JuPedSim walkable areas
            'scenarios/station_sim/network/osm_pt.rou.xml',
            'scenarios/station_sim/network/passengers_arriving.rou.xml',
            'scenarios/station_sim/network/passengers_departing.rou.xml'
        ]),
        '--pedestrian.model', 'jupedsim',  # Use JuPedSim for pedestrians
        '--begin', '25200',  # Start at 7:00 AM
        '--end', str(25200 + max_time),
        '--step-length', '0.5',  # Smaller step for better pedestrian sim
        '--error-log', 'station_sim_errors.log',
        '--log', 'station_sim.log',
        '--no-step-log',
        '--no-warnings'
    ]
    
    # GUI-specific options
    if use_gui:
        sumo_cmd.extend([
            '--delay', '100',  # Slow down for viewing
            '--start',
            '--quit-on-end'
        ])
    
    # Start TraCI
    traci.start(sumo_cmd)
    
    try:
        step = 0
        last_report = 0
        
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            step += 1
            
            # Progress report every 5 minutes
            sim_time = traci.simulation.getTime()
            if sim_time - last_report >= 300:
                num_persons = traci.person.getIDCount()
                num_vehicles = traci.vehicle.getIDCount()
                
                # Convert to HH:MM
                hours = int(sim_time // 3600)
                minutes = int((sim_time % 3600) // 60)
                
                print(f"Time: {hours:02d}:{minutes:02d} | "
                      f"Passengers: {num_persons} | "
                      f"Trains: {num_vehicles}")
                
                last_report = sim_time
            
            # Safety check
            if sim_time > 25200 + max_time:
                print("\nReached maximum simulation time")
                break
                
    except KeyboardInterrupt:
        print("\n\nSimulation interrupted by user")
    except Exception as e:
        print(f"\nError during simulation: {e}")
        import traceback
        traceback.print_exc()
    finally:
        traci.close()
        print("\nSimulation complete!")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Newcastle Station simulation')
    parser.add_argument('--no-gui', action='store_true', 
                       help='Run without GUI (headless mode)')
    parser.add_argument('--no-regen', action='store_true',
                       help='Skip regenerating passenger files')
    parser.add_argument('--duration', type=float, default=2.0,
                       help='Simulation duration in hours (default: 2.0)')
    
    args = parser.parse_args()
    
    # Setup scenario
    setup_scenario(regenerate_passengers=not args.no_regen)
    
    # Run simulation
    run_simulation(
        use_gui=not args.no_gui,
        max_time=int(args.duration * 3600)
    )


if __name__ == "__main__":
    main()
