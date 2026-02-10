"""
Observation Coordinator

Coordinates the generation of observations for all agents by gathering information from:
- Agent positions and nearby agents (via state queries)
- Recent events (via event manager)
- Received messages and conversation history (via message system)
- Agent destinations (for enriching nearby agent info)

This module acts as the glue between multiple systems to create comprehensive
observations for agent decision-making.
"""

from typing import Any

from scenarios.common.logger import get_logger

logger = get_logger(__name__)


class ObservationCoordinator:
    """Coordinates observation generation for all agents."""

    def __init__(
        self,
        concordia_agents: dict[str, Any],
        exited_agents: set[str],
        observation_generator,
        state_queries,
        event_manager,
        message_system,
        agent_destinations: dict[str, str],
        agent_status: dict[str, str],
        test_scenarios: dict[str, Any],
    ):
        """
        Initialize observation coordinator.

        Args:
            concordia_agents: Dict of agent_id -> Concordia entity
            exited_agents: Set of agent IDs who have exited
            observation_generator: ObservationGenerator for formatting observations
            state_queries: Simulation state query interface
            event_manager: EventManager for accessing event history and blocked exits
            message_system: MessageSystem for agent messages and conversations
            agent_destinations: Dict of agent_id -> current exit name
            agent_status: Dict of agent_id -> status (EVACUATING|HELPING|WAITING|INJURED)
            test_scenarios: Test scenario configuration for observation radius
        """
        self.concordia_agents = concordia_agents
        self.exited_agents = exited_agents
        self.observation_generator = observation_generator
        self.state_queries = state_queries
        self.event_manager = event_manager
        self.message_system = message_system
        self.agent_destinations = agent_destinations
        self.agent_status = agent_status
        self.test_scenarios = test_scenarios

    def generate_all_observations(self, current_sim_time: float) -> dict[str, str]:
        """
        Generate observations for all agents based on simulation state.

        Args:
            current_sim_time: Current simulation time in seconds

        Returns:
            Dict of agent_id -> observation string
        """
        observations = {}

        for agent_id in self.concordia_agents.keys():
            # Skip exited agents
            if agent_id in self.exited_agents:
                continue

            try:
                # Get agent state from JuPedSim
                position = self.state_queries.get_agent_position(agent_id)

                # Get observation radius from config
                help_config = self.test_scenarios.get("help_behavior", {})
                observation_radius = help_config.get("observation_radius", 20.0)
                nearby_agents = self.state_queries.get_nearby_agents(
                    agent_id, radius=observation_radius
                )

                # Enrich nearby_agents with target exit info
                for agent_info in nearby_agents:
                    other_id = agent_info.get("id")
                    if other_id:
                        agent_info["target_exit"] = self.agent_destinations.get(other_id)

                # Get recent events
                recent_events = self.state_queries.get_recent_events(
                    self.event_manager.event_history, current_sim_time
                )

                # Get messages received by this agent
                received_messages = self.message_system.get_received_messages(agent_id)

                # Get conversation history for this agent
                conversation_history = self.message_system.get_conversation_history(agent_id)

                # Generate observation
                obs = self.observation_generator.generate_observation(
                    agent_id=agent_id,
                    position=position,
                    nearby_agents=nearby_agents,
                    events=recent_events,
                    sim_time=current_sim_time,
                    blocked_exits=self.event_manager.blocked_exits,
                    agent_status=self.agent_status,
                    received_messages=received_messages,
                    conversation_history=conversation_history,
                )

                observations[agent_id] = obs

            except Exception as e:
                logger.error(f"Error generating observation for {agent_id}: {e}")
                observations[agent_id] = f"[Time: {current_sim_time:.1f}s] You are in the station."

        return observations
