"""
Load pedestrian population and place them within station walking areas.
"""

import sys
from pathlib import Path
from typing import TYPE_CHECKING

# Import from common
sys.path.append(str(Path(__file__).parent.parent))
from common.decision_makers.rule_based import RuleBasedDecisionMaker
from common.station_agent import StationAgent
from common.walking_speed import sample_walking_speed

# Local imports
from movement_sumo import SUMOMovementProvider

if TYPE_CHECKING:
    from station_network import StationNetwork


class PopulationLoader:

    def __init__(self, decision_maker_config: dict = None, network=None):
        """
        Initialize population loader with decision maker configuration.

        Args:
            decision_maker_config: Configuration for rule-based decision makers
            network: SUMO network object (optional, for movement provider)
        """
        self.decision_maker_config = decision_maker_config or {}
        self.movement_provider = SUMOMovementProvider(network)

    def create_agents(
        self, num_agents: int, station_network: "StationNetwork"
    ) -> list[StationAgent]:
        """
        Create agents at random entrance edges with random platform destinations.
        Uses unified StationAgent with SUMOMovementProvider.

        Args:
            num_agents: Number of agents to create
            station_network: StationNetwork instance for querying entrances and platforms

        Returns:
            List of StationAgent objects
        """
        agents = []

        for i in range(num_agents):
            # Select random entrance and platform
            entrance_edge = station_network.get_random_entrance_edge()
            platform_id = station_network.get_random_platform()
            spawn_position = station_network.get_entrance_spawn_position(entrance_edge)

            # Compute full route from entrance to platform
            route = station_network.get_route(entrance_edge, platform_id)

            # Sample preferred walking speed
            walking_speed = sample_walking_speed()

            # Create decision maker for this agent
            decision_maker = RuleBasedDecisionMaker(self.decision_maker_config)

            # Prepare spawn parameters for SUMO
            spawn_params = {
                "entrance_edge": entrance_edge,
                "route": route,
                "destination": platform_id,
                "spawn_position": spawn_position,
                "add_train_stage": True,  # Add train riding stage after walking
            }

            # Create unified agent
            agent = StationAgent(
                agent_id=f"agent_{i}",
                walking_speed=walking_speed,
                decision_maker=decision_maker,
                movement_provider=self.movement_provider,
                initial_zone=station_network.get_location_side(entrance_edge),
                destination=platform_id,
                spawn_params=spawn_params,
            )

            agents.append(agent)
            print(
                f"Created agent_{i}: {entrance_edge} → {platform_id} (zone {station_network.get_location_side(entrance_edge)}→{station_network.get_location_side(platform_id)})"
            )

        return agents
