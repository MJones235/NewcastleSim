"""
Mock JuPedSim simulation for testing Concordia integration.

This is a temporary mock to enable end-to-end testing of the Concordia
decision-making layer without requiring full JuPedSim integration.

TODO: Replace this entire file with real JuPedSim integration
TODO: See scenarios/station_jupedsim/core/simulation.py for reference
"""

from typing import Any

from scenarios.common.logger import get_logger

logger = get_logger(__name__)


class MockJuPedSimAgent:
    """Mock agent for testing."""

    def __init__(self, agent_id: str, initial_position: tuple[float, float]):
        self.id = agent_id
        self.position = list(initial_position)
        self.velocity = [0.0, 0.0]
        self.target = None
        self.is_active = True

    def set_target(self, target: tuple[float, float]):
        """Set movement target."""
        self.target = target
        logger.debug(f"Agent {self.id} target set to {target}")

    def step(self, dt: float):
        """Advance agent by one timestep."""
        if self.target and self.is_active:
            # Simple movement toward target
            dx = self.target[0] - self.position[0]
            dy = self.target[1] - self.position[1]
            dist = (dx**2 + dy**2) ** 0.5

            if dist > 0.1:  # Not at target yet
                # Move at 1.2 m/s toward target
                speed = 1.2
                self.velocity[0] = (dx / dist) * speed
                self.velocity[1] = (dy / dist) * speed

                self.position[0] += self.velocity[0] * dt
                self.position[1] += self.velocity[1] * dt
            else:
                # Reached target
                self.velocity = [0.0, 0.0]


class MockJuPedSimulation:
    """
    Mock JuPedSim simulation for testing Concordia integration.

    Provides minimal functionality to test the hybrid architecture:
    - Agent position tracking
    - Simple movement toward targets
    - Spatial queries (nearby agents)

    TODO: Replace with real JuPedSim simulation
    TODO: Connect to scenarios.station_jupedsim.core.simulation.StationSimulation
    """

    def __init__(self, dt: float = 0.05):
        """
        Initialize mock simulation.

        Args:
            dt: Timestep in seconds (matches JuPedSim convention)
        """
        self.dt = dt
        self.agents: dict[str, MockJuPedSimAgent] = {}
        self.current_step = 0
        self.is_complete = False

        logger.info("MockJuPedSimulation initialized (TODO: Replace with real JuPedSim)")

    def add_agent(self, agent_id: str, position: tuple[float, float]):
        """Add an agent to the simulation."""
        agent = MockJuPedSimAgent(agent_id, position)
        self.agents[agent_id] = agent
        logger.debug(f"Added agent {agent_id} at position {position}")

    def step(self) -> bool:
        """
        Advance simulation by one timestep.

        Returns:
            True if simulation should continue, False if complete
        """
        if self.is_complete:
            return False

        # Move all agents
        for agent in self.agents.values():
            agent.step(self.dt)

        self.current_step += 1

        # Continue indefinitely - let the HybridSimulationRunner control max_steps
        return True

    def get_agent_position(self, agent_id: str) -> tuple[float, float]:
        """Get agent's current position."""
        if agent_id in self.agents:
            pos = self.agents[agent_id].position
            return (pos[0], pos[1])
        return (0.0, 0.0)

    def set_agent_target(self, agent_id: str, target: tuple[float, float]):
        """Set an agent's movement target."""
        if agent_id in self.agents:
            self.agents[agent_id].set_target(target)

    def get_nearby_agents(self, agent_id: str, radius: float) -> list[dict[str, Any]]:
        """
        Get information about agents within radius of given agent.

        Args:
            agent_id: ID of the querying agent
            radius: Search radius in meters

        Returns:
            List of nearby agent info dictionaries
        """
        if agent_id not in self.agents:
            return []

        center_pos = self.agents[agent_id].position
        nearby = []

        for other_id, other_agent in self.agents.items():
            if other_id == agent_id:
                continue

            # Calculate distance
            dx = other_agent.position[0] - center_pos[0]
            dy = other_agent.position[1] - center_pos[1]
            dist = (dx**2 + dy**2) ** 0.5

            if dist <= radius:
                # Check if moving
                is_moving = (other_agent.velocity[0] ** 2 + other_agent.velocity[1] ** 2) > 0.1

                nearby.append(
                    {
                        "id": other_id,
                        "distance": dist,
                        "position": tuple(other_agent.position),
                        "is_moving": is_moving,
                    }
                )

        return nearby

    def get_simulation_time(self) -> float:
        """Get current simulation time in seconds."""
        return self.current_step * self.dt
