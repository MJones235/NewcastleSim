import traci
import sumolib
from simulation_manager import SimulationManager

sumo_binary = sumolib.checkBinary('sumo-gui')


def main():
    # Initialize simulation manager
    sim_manager = SimulationManager('network/osm.net.xml')
    sim_manager.load_network()
    
    # Load population
    # Set use_test=True to use simple test population
    # Set use_test=False to load from actual CSV file
    population_file = '/home/michael/NewcastlePopulation/data/outputs/02_generation/synthetic_population_with_locations_20251209_181047.csv'
    sim_manager.load_population(population_file, use_test=False)
    
    # Start SUMO
    print("\nStarting SUMO simulation...")
    traci.start([
        sumo_binary,
        '--net-file', 'network/osm.net.xml',
        '--additional-files', 'network/osm_stops.add.xml,network/osm_pt.rou.xml',
        '--gui-settings-file', 'network/viewsettings.xml',
        '--delay', '0',
        # '--mesosim',  # Disabled - mesosim has strict insertion limits
        '--start',
        '--error-log', 'sumo_errors.log',
        '--log', 'sumo.log',
        '--ignore-route-errors',  # Don't quit on PT route errors
        '--verbose'
    ])
    
    # Run simulation
    # Start at 7:30 AM (27000 seconds from midnight)
    start_time = 7 * 3600 + 30 * 60  # 7:30 AM
    max_time = 24 * 3600  # End at midnight
    
    sim_manager.current_time = start_time  # Set starting time
    
    print(f"\nRunning simulation from 07:30 to 24:00...")
    
    try:
        while sim_manager.current_time < max_time:
            sim_manager.step()
    except Exception as e:
        print(f"\nError during simulation: {e}")
        print("\nChecking SUMO error log...")
        try:
            with open('sumo_errors.log', 'r') as f:
                errors = f.read()
                if errors:
                    print("=== SUMO Errors ===")
                    print(errors[-2000:])  # Last 2000 chars
        except:
            pass
        raise
    
    # Print final statistics
    print("\n=== Simulation Complete ===")
    stats = sim_manager.get_statistics()
    print(f"Total agents: {stats['total_agents']}")
    print(f"Simulation time: {stats['simulation_time']}s ({stats['simulation_time']/3600:.1f} hours)")
    
    traci.close()


if __name__ == "__main__":
    main()