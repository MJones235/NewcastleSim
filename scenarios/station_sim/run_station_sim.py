import traci
import sumolib
from simulation_manager import StationSimulationManager
import decision_maker_configs

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
        num_agents=500,
        decision_maker_config=decision_maker_configs.RULE_BASED_DEFAULT
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

    print("\nStarting simulation...")

    try:
        while step < 36000:  # 3600 seconds / 0.1 step-length
            traci.simulationStep()  # Advance SUMO simulation
            sim_time = traci.simulation.getTime()
            sim_manager.step(sim_time)
            step += 1

            # Check if all agents completed
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
