"""
Real JuPedSim integration for Concordia station evacuation simulation.

This module provides a concrete implementation of JuPedSim pedestrian dynamics
for the Concordia evacuation scenario, replacing the mock simulation.

Features:
    - Real pedestrian movement physics
    - Station geometry loading from SUMO network files
    - Exit and evacuation routing
    - Spatial queries for agent observations
    - Real-time position tracking for visualization
"""

from pathlib import Path
from typing import Any

import jupedsim as jps

from scenarios.common.logger import get_logger
from scenarios.station_concordia.core.agent_tracker import AgentTracker
from scenarios.station_concordia.core.exit_manager import ExitManager
from scenarios.station_concordia.core.geometry_manager import GeometryManager
from scenarios.station_jupedsim.core.stage_manager import StageManager

logger = get_logger(__name__)


class ConcordiaJuPedSimulation:
    """
    Real JuPedSim simulation wrapper for Concordia integration.

    This class provides the same interface as MockJuPedSimulation but uses
    real JuPedSim pedestrian dynamics underneath.
    """

    def __init__(
        self, network_path: Path | None = None, dt: float = 0.05, exit_radius: float = 10.0
    ):
        """
        Initialize JuPedSim simulation with station geometry.

        Args:
            network_path: Path to network directory containing walking_areas.add.xml
            dt: Timestep in seconds (matches JuPedSim convention)
            exit_radius: Radius of circular exits in meters
        """
        self.dt = dt
        self.exit_radius = exit_radius
        self.current_step = 0
        self.is_complete = False

        if network_path is None:
            raise ValueError("network_path required")

        # Initialize geometry manager
        self.geometry_manager = GeometryManager(network_path, dt)
        self.simulation = self.geometry_manager.simulation
        self.stage_manager = StageManager(self.simulation)

        # Initialize exit manager
        self.exit_manager = ExitManager(
            self.stage_manager,
            self.geometry_manager.entrance_areas,
            self.geometry_manager.walkable_areas_with_obstacles,
        )

        # Initialize agent tracker
        self.agent_tracker = AgentTracker(self.simulation)

        logger.info(
            f"JuPedSim simulation initialized: "
            f"{len(self.geometry_manager.walkable_areas)} walkable areas, "
            f"{len(self.geometry_manager.entrance_areas)} entrances, "
            f"{len(self.exit_manager.evacuation_exits)} exits"
        )

    # Properties for backward compatibility - expose geometry and exit data
    @property
    def walkable_areas(self) -> dict:
        """Get walkable areas from geometry manager."""
        return self.geometry_manager.walkable_areas

    @property
    def walkable_areas_with_obstacles(self) -> dict:
        """Get walkable areas with obstacles integrated."""
        return self.geometry_manager.walkable_areas_with_obstacles

    @property
    def entrance_areas(self) -> dict:
        """Get entrance areas from geometry manager."""
        return self.geometry_manager.entrance_areas

    @property
    def platform_areas(self) -> dict:
        """Get platform areas from geometry manager."""
        return self.geometry_manager.platform_areas

    @property
    def obstacles(self) -> list:
        """Get obstacles from geometry manager."""
        return self.geometry_manager.obstacles

    @property
    def evacuation_exits(self) -> dict[str, int]:
        """Get evacuation exit IDs from exit manager."""
        return self.exit_manager.evacuation_exits

    @property
    def evacuation_journeys(self) -> dict[str, int]:
        """Get evacuation journey IDs from exit manager."""
        return self.exit_manager.evacuation_journeys

    def add_agent(
        self, agent_id: str, position: tuple[float, float], walking_speed: float = 1.34
    ) -> None:
        """
        Add an agent to the simulation.

        Args:
            agent_id: Concordia agent ID
            position: Initial (x, y) position
            walking_speed: Desired walking speed in m/s (default: 1.34 m/s)

        Raises:
            RuntimeError: If no evacuation exits are available
            Exception: If JuPedSim fails to add the agent
        """
        # Get default exit from exit manager
        exit_name, stage_id, journey_id = self.exit_manager.get_default_exit()

        # Add agent to JuPedSim with specified walking speed
        jps_id = self.simulation.add_agent(
            jps.CollisionFreeSpeedModelAgentParameters(
                position=position,
                journey_id=journey_id,
                stage_id=stage_id,
                v0=walking_speed,
            )
        )

        # Register in agent tracker
        self.agent_tracker.add_agent(agent_id, jps_id)

        logger.info(
            f"Added agent {agent_id} at {position} "
            f"with speed {walking_speed:.2f} m/s (JPS ID: {jps_id})"
        )

    def step(self) -> bool:
        """
        Advance simulation by one timestep.

        Returns:
            True if simulation should continue, False if complete
        """
        if self.is_complete:
            return False

        # Run JuPedSim step
        self.simulation.iterate()
        self.current_step += 1

        # Check if any agents remain
        if self.simulation.agent_count() == 0:
            logger.info("All agents have exited the simulation")
            self.is_complete = True
            return False

        return True

    def get_agent_position(self, agent_id: str) -> tuple[float, float]:
        """
        Get agent's current position.

        Args:
            agent_id: Concordia agent ID

        Returns:
            Agent's (x, y) position, or (0.0, 0.0) if agent has exited
        """
        return self.agent_tracker.get_position(agent_id)

    def set_agent_target(self, agent_id: str, target: tuple[float, float]) -> None:
        """
        Set an agent's movement target by creating a waypoint.

        Args:
            agent_id: Concordia agent ID
            target: Target (x, y) position

        Raises:
            KeyError: If agent is not in the simulation (may have exited)
            Exception: If JuPedSim fails to create waypoint or switch journey
        """
        if not self.agent_tracker.is_agent_active(agent_id):
            logger.debug(f"Cannot set target for agent {agent_id} - already exited")
            return

        jps_id = self.agent_tracker.get_jps_id(agent_id)
        self.agent_tracker.set_target(agent_id, target)

        # Create a waypoint stage at the target location
        stage_id = self.simulation.add_waypoint_stage(target, distance=2.0)

        # Create a journey to this waypoint
        journey = jps.JourneyDescription([stage_id])
        journey_id = self.simulation.add_journey(journey)

        # Update agent's journey and stage
        self.simulation.switch_agent_journey(
            agent_id=jps_id,
            journey_id=journey_id,
            stage_id=stage_id,
        )

        logger.info(f"Set target for agent {agent_id} to {target}")

    def set_agent_evacuation_exit(self, agent_id: str, exit_name: str) -> None:
        """
        Direct an agent to a specific evacuation exit.

        Args:
            agent_id: Concordia agent ID
            exit_name: Name of the evacuation exit

        Raises:
            KeyError: If agent or exit is not found
            Exception: If JuPedSim fails to switch journey
        """
        if not self.agent_tracker.is_agent_active(agent_id):
            logger.debug(f"Cannot set exit for agent {agent_id} - already exited")
            return

        jps_id = self.agent_tracker.get_jps_id(agent_id)
        stage_id, journey_id = self.exit_manager.get_exit_ids(exit_name)

        self.simulation.switch_agent_journey(
            agent_id=jps_id,
            journey_id=journey_id,
            stage_id=stage_id,
        )

        logger.info(f"Directed agent {agent_id} to exit '{exit_name}'")

    def set_agent_speed(self, agent_id: str, speed: float) -> None:
        """
        Set an agent's walking speed mid-simulation.

        Args:
            agent_id: Concordia agent ID
            speed: Walking speed in m/s

        Raises:
            Exception: If JuPedSim fails to update agent's speed
        """
        if not self.agent_tracker.is_agent_active(agent_id):
            logger.debug(f"Cannot set speed for agent {agent_id} - already exited")
            return

        jps_id = self.agent_tracker.get_jps_id(agent_id)

        # Get the agent object and update its desired speed
        agent = self.simulation.agent(jps_id)
        agent.model.v0 = speed

        logger.info(f"Set agent {agent_id} speed to {speed:.2f} m/s")

    def get_nearby_agents(self, agent_id: str, radius: float) -> list[dict[str, Any]]:
        """
        Get information about agents within radius of given agent.

        Args:
            agent_id: ID of the querying agent
            radius: Search radius in meters

        Returns:
            List of nearby agent info dictionaries
        """
        return self.agent_tracker.get_nearby_agents(agent_id, radius)

    def get_simulation_time(self) -> float:
        """Get current simulation time in seconds."""
        return self.current_step * self.dt

    def get_all_agent_positions(self) -> dict[str, tuple[float, float]]:
        """
        Get positions of all agents for visualization.

        Returns:
            Dictionary mapping Concordia agent IDs to (x, y) positions
        """
        return self.agent_tracker.get_all_positions()

    def get_geometry(self) -> dict[str, Any]:
        """
        Get geometry information for visualization.

        Returns:
            Dictionary with walkable areas, entrances, platforms, etc.
        """
        geometry_data = self.geometry_manager.get_geometry_data()

        # Add evacuation exits info
        geometry_data["evacuation_exits"] = {
            name: geometry_data["entrance_areas"].get(name)
            for name in self.exit_manager.evacuation_exits.keys()
            if name in geometry_data["entrance_areas"]
        }

        return geometry_data
