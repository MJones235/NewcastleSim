import sumolib
import traci
from simulation_manager import SimulationManager

# Performance mode: use headless SUMO for speed (10-20x faster)
use_gui = True
sumo_binary = sumolib.checkBinary("sumo-gui" if use_gui else "sumo")


def main():
    # Initialize simulation manager
    sim_manager = SimulationManager("scenarios/traffic_sim/network/net.net.xml")
    sim_manager.load_network()

    # Load population
    trip_file = "/home/michael/NewcastlePopulation/data/outputs/04_daily_routine/formatted_travel_diaries_20251211_164752.csv"
    sim_manager.load_population(trip_file, use_trips=True)

    # Start SUMO
    print("\nStarting SUMO simulation...")

    max_time = 24 * 3600  # End at midnight

    sumo_cmd = [
        sumo_binary,
        "--net-file",
        "scenarios/traffic_sim/network/net.net.xml",
        "--additional-files",
        "scenarios/traffic_sim/network/osm_stops.add.xml,scenarios/traffic_sim/network/osm_pt.rou.xml",
        "--mesosim",
        "--error-log",
        "sumo_errors.log",
        "--log",
        "sumo.log",
        "--ignore-route-errors",  # Don't quit on PT route errors
        "--time-to-teleport",
        "-1",  # Disable teleporting
        "--no-step-log",  # Disable verbose step logging for speed
        "--no-warnings",  # Suppress routing warnings (logged to file anyway)
    ]

    # GUI-specific options
    if use_gui:
        sumo_cmd.extend(
            [
                "--gui-settings-file",
                "scenarios/traffic_sim/network/viewsettings.xml",
                "--delay",
                "0",
                "--start",
            ]
        )

    traci.start(sumo_cmd)

    try:
        while sim_manager.current_time < max_time:
            sim_manager.step()
    except Exception as e:
        print(f"\nError during simulation: {e}")
        print("\nChecking SUMO error log...")
        try:
            with open("sumo_errors.log") as f:
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
    print(
        f"Simulation time: {stats['simulation_time']}s ({stats['simulation_time']/3600:.1f} hours)"
    )

    traci.close()


if __name__ == "__main__":
    main()
