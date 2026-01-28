"""
JuPedSim-specific movement provider.

Handles spawning, positioning, and routing for JuPedSim simulations.
Implements the MovementProvider interface for JuPedSim pedestrian dynamics.

This provider manages:
    - Agent spawning with position and journey assignment
    - Position queries and updates from JuPedSim simulation
    - Journey/stage switching for agent rerouting
    - Agent lifecycle (active checking, removal)
    - Zone-based location queries
"""

from typing import Any

import jupedsim as jps
from shapely.geometry import Polygon

from scenarios.base.movement_provider import MovementProvider


class JuPedSimMovementProvider(MovementProvider):
    """
    Movement provider for JuPedSim pedestrian simulations.
    Handles JuPedSim agent API calls and stage-based routing.
    """

    def __init__(self, simulation: jps.Simulation, zones: dict[str, Polygon]) -> None:
        """
        Initialize JuPedSim movement provider.

        Args:
            simulation: JuPedSim simulation object
            zones: Dictionary mapping zone names to polygons
        """
        self.simulation: jps.Simulation = simulation
        self.zones: dict[str, Polygon] = zones
        self.evacuation_journeys: dict[str, int] = {}  # Will be set by main script
        self.evacuation_exits: dict[str, int] = {}  # Will be set by main script

    def spawn_agent(self, agent: Any, spawn_params: dict[str, Any]) -> bool:
        """
        Spawn an agent in JuPedSim simulation.

        Args:
            agent: StationAgent to spawn
            spawn_params: Must contain:
                - position: (x, y) tuple OR entrance_polygon: Polygon for spawn location
                - platform_polygon: Polygon for platform destination
                - journey_id and stage_id OR create unique waypoint

        Returns:
            True if spawn successful
        """
        try:
            # Get or generate spawn position
            if "position" in spawn_params:
                position = spawn_params["position"]
            elif "entrance_polygon" in spawn_params:
                # Generate a single position from the entrance polygon
                entrance_polygon = spawn_params["entrance_polygon"]
                positions = jps.distribute_by_number(
                    polygon=entrance_polygon,
                    number_of_agents=1,
                    distance_to_agents=0.5,
                    distance_to_polygon=0.3,
                    seed=None,  # Random seed for variety
                )
                position = positions[0]
            else:
                raise ValueError(
                    "spawn_params must contain either 'position' or 'entrance_polygon'"
                )

            # Generate unique waypoint position for this agent within their platform
            if "platform_polygon" in spawn_params:
                platform_polygon = spawn_params["platform_polygon"]
                # Generate a random position within the platform for this agent's waypoint
                waypoint_positions = jps.distribute_by_number(
                    polygon=platform_polygon,
                    number_of_agents=1,
                    distance_to_agents=0.3,
                    distance_to_polygon=0.2,
                    seed=None,
                )
                waypoint_pos = waypoint_positions[0]

                # Create unique waypoint stage for this agent
                stage_id = self.simulation.add_waypoint_stage(waypoint_pos, distance=2.0)

                # Create unique journey for this agent
                journey = jps.JourneyDescription([stage_id])
                journey_id = self.simulation.add_journey(journey)
            else:
                # Use provided journey and stage
                journey_id = spawn_params["journey_id"]
                stage_id = spawn_params.get("stage_id")

            # Add agent to JuPedSim with journey
            jps_id = self.simulation.add_agent(
                jps.CollisionFreeSpeedModelAgentParameters(
                    position=position,
                    journey_id=journey_id,
                    stage_id=stage_id,
                    v0=agent.walking_speed,
                )
            )

            # Store JuPedSim agent ID
            agent.jps_agent_id = jps_id
            return True

        except Exception as e:
            print(f"Failed to spawn agent {agent.id} in JuPedSim: {e}")
            return False

    def update_agent_position(self, agent: Any) -> tuple[float, float]:  # type: ignore[no-any-return]
        """
        Get agent's current position from JuPedSim.

        Args:
            agent: StationAgent with jps_agent_id attribute

        Returns:
            (x, y) coordinates
        """
        if not hasattr(agent, "jps_agent_id"):
            return (0.0, 0.0)

        try:
            # JuPedSim library returns Any type for agent positions
            position = self.simulation.agent_position(agent.jps_agent_id)
            return (float(position[0]), float(position[1]))  # type: ignore[no-any-return]
        except Exception:
            return agent.position  # type: ignore[no-any-return]  # Return last known position

    def set_agent_target(self, agent: Any, target: Any) -> bool:
        """
        Set new journey/stage for agent.

        Args:
            agent: StationAgent with jps_agent_id
            target: Dictionary with 'journey_id' or 'stage_id' for new target

        Returns:
            True if successfully redirected
        """
        if not hasattr(agent, "jps_agent_id"):
            return False

        try:
            # JuPedSim uses journey switching for rerouting
            if "journey_id" in target:
                self.simulation.switch_agent_journey(agent.jps_agent_id, target["journey_id"])
                return True
            return False

        except Exception as e:
            print(f"Failed to reroute agent {agent.id}: {e}")
            return False

    def is_agent_active(self, agent: Any) -> bool:
        """
        Check if agent still exists in JuPedSim simulation.

        Args:
            agent: StationAgent with jps_agent_id

        Returns:
            True if agent still in simulation
        """
        if not hasattr(agent, "jps_agent_id"):
            return False

        try:
            # Check if agent ID is in current agent list
            agent_ids = [a.id for a in self.simulation.agents()]
            return agent.jps_agent_id in agent_ids
        except Exception:
            return False

    def remove_agent(self, agent: Any) -> None:
        """
        Remove agent from JuPedSim simulation.

        Args:
            agent: StationAgent with jps_agent_id
        """
        if hasattr(agent, "jps_agent_id"):
            try:
                self.simulation.mark_agent_for_removal(agent.jps_agent_id)
            except Exception:
                pass

    def get_agent_location_info(self, agent: Any) -> dict[str, Any]:
        """
        Get agent's current position and zone from JuPedSim.

        Args:
            agent: StationAgent with jps_agent_id

        Returns:
            Dictionary with position (x, y) tuple and zone name
        """
        if not hasattr(agent, "jps_agent_id"):
            return {"position": (0.0, 0.0), "zone": "unknown"}

        try:
            position = self.simulation.agent_position(agent.jps_agent_id)
            pos_tuple = (position[0], position[1])

            # Determine which zone the agent is in
            zone_name = "unknown"
            from shapely.geometry import Point

            point = Point(pos_tuple)

            # Check each zone to see if agent is inside
            for name, polygon in self.zones.items():
                if polygon.contains(point):
                    zone_name = name
                    break

            return {"position": pos_tuple, "zone": zone_name}
        except Exception:
            fallback_pos = agent.position if hasattr(agent, "position") else (0.0, 0.0)
            return {"position": fallback_pos, "zone": "unknown"}

    def reroute_to_evacuation_exit(self, agent: Any, exit_name: str) -> bool:
        """
        Reroute agent to a specific evacuation exit.

        Switches the agent's journey to one that leads directly to the named exit.
        Requires that evacuation journeys have been configured via the
        evacuation_journeys dictionary.

        Args:
            agent: StationAgent to reroute
            exit_name: Name of evacuation exit to route to

        Returns:
            True if successfully rerouted, False otherwise

        Args:
            agent: StationAgent to reroute
            exit_name: Name of the evacuation exit entrance

        Returns:
            True if successfully rerouted
        """
        if not hasattr(agent, "jps_agent_id"):
            return False

        # Get the evacuation journey and exit stage for this exit
        if exit_name not in self.evacuation_journeys:
            print(f"Warning: No evacuation journey found for exit '{exit_name}'")
            return False

        journey_id = self.evacuation_journeys[exit_name]
        stage_id = self.evacuation_exits[exit_name]

        try:
            # Switch agent to evacuation journey with the exit stage
            self.simulation.switch_agent_journey(agent.jps_agent_id, journey_id, stage_id)
            return True
        except Exception as e:
            print(f"Failed to reroute agent {agent.id} to {exit_name}: {e}")
            return False
