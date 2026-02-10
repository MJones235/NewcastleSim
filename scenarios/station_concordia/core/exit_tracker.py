"""
Exit Tracker

Manages tracking of agents who have exited the simulation through evacuation exits.
Monitors agent positions and detects when agents are no longer present in the
JuPedSim simulation, marking them as successfully evacuated.
"""

from typing import Any

from scenarios.common.logger import get_logger

logger = get_logger(__name__)


class ExitTracker:
    """Tracks agents who have exited the simulation."""

    def __init__(
        self,
        concordia_agents: dict[str, Any],
        exited_agents: set[str],
        agent_destinations: dict[str, str],
        jps_sim,
    ):
        """
        Initialize exit tracker.

        Args:
            concordia_agents: Dict of agent_id -> Concordia entity
            exited_agents: Set of agent IDs who have exited
            agent_destinations: Dict of agent_id -> current exit name
            jps_sim: JuPedSim simulation instance
        """
        self.concordia_agents = concordia_agents
        self.exited_agents = exited_agents
        self.agent_destinations = agent_destinations
        self.jps_sim = jps_sim

    def check_exited_agents(self, current_sim_time: float, current_step: int):
        """
        Check for agents who have reached exits and mark them as exited.

        Args:
            current_sim_time: Current simulation time in seconds
            current_step: Current simulation step number
        """
        if not hasattr(self.jps_sim, "get_all_agent_positions"):
            return

        # Get current agent positions from JuPedSim
        current_positions = self.jps_sim.get_all_agent_positions()

        # Log agent count for debugging
        total_agents = len(self.concordia_agents)
        active_agents = len(current_positions)
        exited_count = len(self.exited_agents)

        # Find agents that are no longer in JuPedSim (they've exited)
        newly_exited = []
        for agent_id in list(self.concordia_agents.keys()):
            if agent_id not in self.exited_agents and agent_id not in current_positions:
                self.exited_agents.add(agent_id)
                exit_name = self.agent_destinations.get(agent_id, "unknown")
                newly_exited.append((agent_id, exit_name))

        # Log newly exited agents
        for agent_id, exit_name in newly_exited:
            logger.info(f"✅ {agent_id} has evacuated through {exit_name}")

        # Periodic status update every 50 steps
        if current_step % 50 == 0 and current_step > 0:
            logger.info(
                f"📊 Agent status: {active_agents} active, {exited_count} exited, "
                f"{total_agents} total (t={current_sim_time:.1f}s)"
            )
