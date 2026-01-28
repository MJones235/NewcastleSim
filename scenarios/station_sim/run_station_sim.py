import sys
from pathlib import Path

import sumolib
import traci

# Add project root to Python path (two levels up from this file)
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scenarios.common.decision_makers import configs as decision_maker_configs
from scenarios.station_sim.simulation_manager import StationSimulationManager

use_gui = True
sumo_binary = sumolib.checkBinary("sumo-gui" if use_gui else "sumo")


def main():
    # Initialize simulation manager
    sim_manager = StationSimulationManager(
        network_file="scenarios/station_sim/network/station_network.net.xml",
        walking_areas_file="scenarios/station_sim/network/walking_areas.add.xml",
        stops_file="scenarios/station_sim/network/osm_stops.add.xml",
    )
    sim_manager.load_network()

    # Load population with decision makers
    # Configure evacuation probability: RULE_BASED_DEFAULT, RULE_BASED_HIGH_COMPLIANCE, or RULE_BASED_LOW_COMPLIANCE
    sim_manager.load_population(
        num_agents=100, decision_maker_config=decision_maker_configs.RULE_BASED_DEFAULT
    )

    # Configure SUMO
    sumo_cmd = [
        sumo_binary,
        "--net-file",
        "scenarios/station_sim/network/station_network.net.xml",
        "--additional-files",
        "scenarios/station_sim/network/osm_stops.add.xml,scenarios/station_sim/network/walking_areas.add.xml",
        "--route-files",
        "scenarios/station_sim/network/osm_pt.rou.xml",
        "--pedestrian.model",
        "jupedsim",
        "--gui-settings-file",
        "scenarios/station_sim/network/viewsettings.xml",
        "--error-log",
        "sumo_errors.log",
        "--log",
        "sumo.log",
        "--ignore-route-errors",
        "--time-to-teleport",
        "-1",
        "--no-step-log",
        "--step-length",
        "0.1",  # Fine-grained pedestrian simulation
        "--begin",
        "0",
        "--end",
        "3600",  # 1 hour simulation
        "--delay",
        "500",
    ]

    # Start SUMO
    traci.start(sumo_cmd)

    # Queue agents for spawning
    sim_manager.spawn_agents()

    # Main simulation loop
    step = 0
    last_stats_time = 0

    print("\nStarting simulation...")

    try:
        while step < 36000:  # 3600 seconds / 0.1 step-length
            traci.simulationStep()  # Advance SUMO simulation
            sim_time = traci.simulation.getTime()
            sim_manager.step(sim_time)
            step += 1

            # Print statistics every 10 seconds
            if sim_time - last_stats_time >= 10:
                stats = sim_manager.get_simulation_statistics()
                print(
                    f"t={sim_time:.1f}s: spawned={stats['spawned_agents']}, active={stats['active_agents']}, completed={stats['completed_agents']}"
                )
                last_stats_time = sim_time

            # Check if all agents completed their journeys
            stats = sim_manager.get_simulation_statistics()
            if stats["active_agents"] == 0 and stats["completed_agents"] > 0:
                print("\nAll agents completed their journeys!")
                break

    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
    finally:
        print("\nSimulation ended")
        traci.close()


if __name__ == "__main__":
    main()
