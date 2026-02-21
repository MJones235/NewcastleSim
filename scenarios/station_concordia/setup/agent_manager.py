"""
Agent manager for Station Concordia simulations.

This module is responsible for:
- Complete agent lifecycle management
- Creating agent configurations
- Generating spawn positions
- Adding agents to the pedestrian simulation with appropriate speeds
- Coordinating all agent-related operations
"""

from typing import Any

from scenarios.common.logger import get_logger
from scenarios.common.walking_speed import sample_walking_speed
from scenarios.station_concordia.jps_integration.simulation_interface import PedestrianSimulation
from scenarios.station_concordia.setup.agent_factory import AgentFactory
from scenarios.station_concordia.setup.spawn_manager import SpawnManager

logger = get_logger(__name__)


class AgentManager:
    """Handles complete agent lifecycle management."""

    @staticmethod
    def create_and_populate_agents(
        jps_sim: PedestrianSimulation, config: dict
    ) -> list[dict[str, Any]]:
        """
        Create agents and add them to the pedestrian simulation.

        This is the main entry point for all agent-related operations.
        It handles:
        - Determining how many agents to create
        - Generating spawn positions
        - Creating agent configurations
        - Adding agents to the simulation with appropriate walking speeds

        Args:
            jps_sim: Pedestrian simulation instance (implements PedestrianSimulation)
            config: Configuration dictionary

        Returns:
            List of agent configuration dictionaries
        """
        # Determine number of agents
        agent_config = config.get("agents", {})
        num_agents = agent_config.get("count", 1)

        # Generate spawn positions
        spawn_positions = SpawnManager.generate_spawn_positions(jps_sim, num_agents)

        # Create agent configurations
        agents_config, injured_agents = AgentFactory.create_agents(num_agents, config)

        # Add agents to JuPedSim
        AgentManager._add_agents_to_jupedsim(
            jps_sim, agents_config, spawn_positions, injured_agents, config
        )

        logger.info(f"Agent population complete: {num_agents} agents ready")
        return agents_config

    @staticmethod
    def _add_agents_to_jupedsim(jps_sim, agents_config, spawn_positions, injured_agents, config):
        """
        Add agents to JuPedSim simulation with appropriate walking speeds.

        Args:
            jps_sim: JuPedSim simulation instance
            agents_config: List of agent configuration dictionaries
            spawn_positions: List of spawn position tuples (x, y) or (x, y, level_id)
            injured_agents: Set of injured agent indices
            config: Configuration dictionary
        """
        help_config = config.get("test_scenarios", {}).get("help_behavior", {})

        for i, agent_cfg in enumerate(agents_config):
            agent_id = agent_cfg["id"]
            spawn_data = spawn_positions[i]

            # Handle both (x, y) and (x, y, level_id) formats
            if len(spawn_data) == 3:
                # Multi-level: (x, y, level_id)
                x, y, level_id = spawn_data
                start_pos = (x, y)
            else:
                # Single-level: (x, y)
                start_pos = spawn_data
                level_id = "0"  # Default level

            # Store level_id in agent config for use during agent initialization
            agent_cfg["level_id"] = level_id

            is_injured = i in injured_agents

            if is_injured:
                walking_speed = help_config.get("injured_walking_speed", 0.5)
            else:
                walking_speed = sample_walking_speed()

            # Add agent with level_id for multi-level simulations
            if hasattr(jps_sim, "simulations"):
                # Multi-level simulation
                jps_sim.add_agent(
                    agent_id, start_pos, walking_speed=walking_speed, level_id=level_id
                )
            else:
                # Single-level simulation
                jps_sim.add_agent(agent_id, start_pos, walking_speed=walking_speed)
