"""
Population loader for JuPedSim station simulation.
Provides functions to create and spawn StationAgent instances in JuPedSim simulations.
Supports both immediate spawning (agents created at t=0) and gradual spawning
(agents created over time as simulation runs).

Key Functions:
    - create_agents_from_entrances: Create agents at entrance locations with
"""

import jupedsim as jps
import numpy as np
from shapely.geometry import Polygon

from scenarios.common.decision_makers.rule_based import RuleBasedDecisionMaker
from scenarios.common.station_agent import StationAgent
from scenarios.common.walking_speed import sample_walking_speed
from scenarios.station_jupedsim.core.movement_jupedsim import JuPedSimMovementProvider


def create_agents_from_entrances(
    simulation: jps.Simulation,
    movement_provider: JuPedSimMovementProvider,
    entrance_areas: dict[str, Polygon],
    platform_stages: dict[str, int],
    platform_journeys: dict[str, int],
    platform_areas: dict[str, Polygon],
    num_agents: int,
    agent_list: list[StationAgent],
    spawn_immediately: bool = True,
) -> None:
    """
    Create agents at entrance locations with random platform destinations.

    Args:
        simulation: JuPedSim simulation object
        movement_provider: JuPedSim movement provider
        entrance_areas: Dictionary of entrance name -> entrance polygon
        platform_stages: Dictionary of platform name -> stage_id for that platform
        platform_journeys: Dictionary of platform name -> journey_id for that platform
        platform_areas: Dictionary of platform name -> platform polygon
        num_agents: Total number of agents to create
        agent_list: List to append created StationAgent objects to
        spawn_immediately: If True, spawn agents immediately; if False, just create them
    """
    if not entrance_areas:
        print("Warning: No entrance areas defined, cannot spawn agents")
        return

    if not platform_stages:
        print("Warning: No platform stages defined, cannot create agents")
        return

    # Get list of platform names and stage IDs
    platform_names = list(platform_stages.keys())

    # Distribute agents across entrances
    entrance_names = list(entrance_areas.keys())
    agents_per_entrance = num_agents // len(entrance_names)
    remaining = num_agents % len(entrance_names)

    for idx, entrance_name in enumerate(entrance_names):
        entrance_polygon = entrance_areas[entrance_name]

        # Distribute count - give remaining agents to first entrances
        count = agents_per_entrance + (1 if idx < remaining else 0)

        if count == 0:
            continue

        if spawn_immediately:
            # Distribute agent positions within entrance area
            positions = jps.distribute_by_number(
                polygon=entrance_polygon,
                number_of_agents=count,
                distance_to_agents=0.5,
                distance_to_polygon=0.3,  # Smaller distance for entrance areas
                seed=42 + idx,
            )
        else:
            # For gradual spawning, create agents without positions yet
            positions = [None] * count

        # Create each agent
        for pos in positions:
            # Assign random platform destination
            platform_name = np.random.choice(platform_names)

            # Sample walking speed
            walking_speed = sample_walking_speed()

            # Create decision maker
            evac_prob = np.random.uniform(0.3, 0.7)
            config = {"evacuation_probability": evac_prob}
            decision_maker = RuleBasedDecisionMaker(config=config)

            # Create unified agent
            agent_id = f"agent_{len(agent_list)}"

            # Store platform info - stage/journey will be created per agent later
            # For gradual spawning, generate position at spawn time from entrance polygon
            if pos is None:
                # Store entrance polygon and platform info for later
                spawn_params = {
                    "entrance_polygon": entrance_polygon,
                    "platform_name": platform_name,
                    "platform_polygon": platform_areas[platform_name],
                    "platform_stages": platform_stages,
                    "platform_journeys": platform_journeys,
                }
            else:
                spawn_params = {
                    "position": pos,
                    "platform_name": platform_name,
                    "platform_polygon": platform_areas[platform_name],
                    "platform_stages": platform_stages,
                    "platform_journeys": platform_journeys,
                }

            agent = StationAgent(
                agent_id=agent_id,
                walking_speed=walking_speed,
                decision_maker=decision_maker,
                movement_provider=movement_provider,
                initial_zone=entrance_name,
                destination=platform_name,
                spawn_params=spawn_params,
            )

            # Spawn immediately or add to list for later spawning
            if spawn_immediately:
                if agent.spawn():
                    agent_list.append(agent)
            else:
                agent_list.append(agent)

        print(f"Created {count} agents at entrance '{entrance_name}'")


if __name__ == "__main__":
    print("Population loader module ready")
