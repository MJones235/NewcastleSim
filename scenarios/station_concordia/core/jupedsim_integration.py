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
from shapely.geometry import Point

from scenarios.common.logger import get_logger
from scenarios.station_jupedsim.core.stage_manager import StageManager
from scenarios.station_jupedsim.geometry import (
    GeometryProcessor,
    load_entrance_areas,
    load_obstacles,
    load_platform_areas,
    load_walkable_areas,
)

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

        # Load geometry
        logger.info("Loading station geometry from network files...")
        if network_path is None:
            raise ValueError("network_path required")
        (
            self.walkable_areas,
            self.walkable_areas_with_obstacles,
            self.entrance_areas,
            self.platform_areas,
            self.obstacles,
        ) = self._load_geometry(network_path)

        # Create JuPedSim simulation
        logger.info("Initializing JuPedSim simulation...")
        self.simulation = self._create_simulation()
        self.stage_manager = StageManager(self.simulation)

        # Setup evacuation exits and routes
        logger.info("Setting up evacuation exits...")
        self.evacuation_exits, self.evacuation_journeys = self._setup_evacuation_exits()

        logger.info(
            f"Created {len(self.evacuation_exits)} exits: {list(self.evacuation_exits.keys())}"
        )
        logger.info(f"Exit IDs: {self.evacuation_exits}")
        logger.info(f"Journey IDs: {self.evacuation_journeys}")

        # Track agents
        self.agent_ids: dict[str, int] = {}  # Concordia ID -> JuPedSim ID
        self.jps_to_concordia: dict[int, str] = {}  # JuPedSim ID -> Concordia ID
        self.agent_targets: dict[str, tuple[float, float]] = {}

        logger.info(
            f"JuPedSim simulation initialized: "
            f"{len(self.walkable_areas)} walkable areas, "
            f"{len(self.entrance_areas)} entrances, "
            f"{len(self.evacuation_exits)} exits"
        )

    def _load_geometry(self, network_path: Path) -> tuple[dict, dict, dict, dict, list]:
        """Load station geometry from SUMO network files."""
        walking_areas_file = network_path / "walking_areas.add.xml"

        if not walking_areas_file.exists():
            raise FileNotFoundError(f"Geometry file not found: {walking_areas_file}")

        walkable_areas = load_walkable_areas(str(walking_areas_file))
        entrance_areas = load_entrance_areas(str(walking_areas_file))
        platform_areas = load_platform_areas(str(walking_areas_file))
        obstacles = load_obstacles(str(walking_areas_file))

        # Integrate obstacles into walkable areas as polygon holes
        walkable_areas_with_obstacles, fixed_obstacles = GeometryProcessor.integrate_obstacles(
            walkable_areas, obstacles
        )

        logger.info(f"  Loaded {len(walkable_areas)} walkable areas")
        logger.info(f"  Loaded {len(entrance_areas)} entrance areas")
        logger.info(f"  Loaded {len(platform_areas)} platform areas")
        logger.info(f"  Loaded {len(obstacles)} obstacles")
        logger.info(f"  Integrated {len(fixed_obstacles)} obstacles into walkable areas")

        return (
            walkable_areas,
            walkable_areas_with_obstacles,
            entrance_areas,
            platform_areas,
            fixed_obstacles,
        )

    def _create_simulation(self) -> jps.Simulation:
        """Create JuPedSim simulation with loaded geometry."""
        # Merge all walkable areas (with obstacles removed) into one geometry
        all_areas = list(self.walkable_areas_with_obstacles.values())

        if not all_areas:
            raise ValueError("No walkable areas found in geometry")

        # Combine into a single geometry
        main_area = GeometryProcessor.combine_geometry(all_areas)

        # Create JuPedSim simulation
        simulation = jps.Simulation(
            model=jps.CollisionFreeSpeedModel(),
            geometry=main_area,
            dt=self.dt,
        )

        area = main_area.area if hasattr(main_area, "area") else 0.0
        logger.info(f"  Created simulation with area: {area:.1f} m²")

        return simulation

    def _setup_evacuation_exits(self) -> tuple[dict[str, int], dict[str, int]]:
        """
        Create evacuation exit stages at entrance locations.

        Raises:
            RuntimeError: If no valid exits can be created from entrance areas
        """
        evacuation_exits = {}
        evacuation_journeys = {}

        if not self.entrance_areas:
            raise RuntimeError(
                "No entrance areas found in geometry. "
                "Check that walking_areas.add.xml contains entrance area definitions."
            )

        walkable_geometry = GeometryProcessor.combine_geometry(
            list(self.walkable_areas_with_obstacles.values())
        )

        failed_exits = []
        for entrance_name, entrance_polygon in self.entrance_areas.items():
            exit_id = self._create_convex_exit_from_polygon(
                entrance_name, entrance_polygon, walkable_geometry
            )

            if exit_id is None:
                failed_exits.append(entrance_name)
                logger.warning(f"Failed to create exit at '{entrance_name}'")
                continue

            evacuation_exits[entrance_name] = exit_id

            # Create journey to this exit
            journey_id = self.stage_manager.create_simple_exit_journey(
                journey_name=f"journey_to_{entrance_name}", exit_id=exit_id
            )
            evacuation_journeys[entrance_name] = journey_id

            logger.info(
                f"Created evacuation exit '{entrance_name}' "
                f"(exit={exit_id}, journey={journey_id})"
            )

        if not evacuation_exits:
            raise RuntimeError(
                f"Failed to create any evacuation exits. "
                f"Attempted exits: {list(self.entrance_areas.keys())}. "
                f"All exits failed. Check geometry configuration and ensure "
                f"entrance areas overlap with walkable areas."
            )

        if failed_exits:
            logger.warning(
                f"Successfully created {len(evacuation_exits)} exits, "
                f"but {len(failed_exits)} failed: {failed_exits}"
            )

        return evacuation_exits, evacuation_journeys

    def _create_convex_exit_from_polygon(
        self,
        exit_name: str,
        polygon,
        walkable_geometry,
    ) -> int | None:
        """
        Create a convex rectangular exit centered on polygon centroid.

        Args:
            exit_name: Name for the exit
            polygon: Entrance polygon to create exit from
            walkable_geometry: Combined walkable geometry for validation

        Returns:
            Exit ID if successful, None if exit couldn't be placed
        """
        # Find a valid centroid within walkable geometry
        centroid = polygon.centroid
        if not walkable_geometry.contains(centroid):
            intersection = polygon.intersection(walkable_geometry)
            if not intersection.is_empty:
                centroid = intersection.representative_point()
            else:
                logger.error(
                    f"Exit '{exit_name}': polygon doesn't intersect walkable geometry. "
                    f"Polygon bounds: {polygon.bounds}"
                )
                return None

        # Create exit with standard size
        exit_size = 4.0  # 4x4 meter exit (standard)
        exit_coords = [
            (centroid.x - exit_size / 2, centroid.y - exit_size / 2),
            (centroid.x + exit_size / 2, centroid.y - exit_size / 2),
            (centroid.x + exit_size / 2, centroid.y + exit_size / 2),
            (centroid.x - exit_size / 2, centroid.y + exit_size / 2),
        ]

        exit_id = self.stage_manager.create_exit_at_coordinates(exit_name, exit_coords)
        logger.info(
            f"Created {exit_size}m x {exit_size}m exit '{exit_name}' " f"at {centroid.coords[0]}"
        )
        return exit_id

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
        if not self.evacuation_exits:
            raise RuntimeError(
                "Cannot add agent: no evacuation exits available. " "Check geometry configuration."
            )

        exit_name = list(self.evacuation_exits.keys())[0]
        stage_id = self.evacuation_exits[exit_name]
        journey_id = self.evacuation_journeys[exit_name]

        # Add agent to JuPedSim with specified walking speed
        jps_id = self.simulation.add_agent(
            jps.CollisionFreeSpeedModelAgentParameters(
                position=position,
                journey_id=journey_id,
                stage_id=stage_id,
                v0=walking_speed,
            )
        )

        # Track agent mapping
        self.agent_ids[agent_id] = jps_id
        self.jps_to_concordia[jps_id] = agent_id

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
        if agent_id not in self.agent_ids:
            # Agent may have exited - this is expected
            return (0.0, 0.0)

        jps_id = self.agent_ids[agent_id]

        # Find agent in current simulation state
        for agent in self.simulation.agents():
            if agent.id == jps_id:
                return (float(agent.position[0]), float(agent.position[1]))

        # Agent not found - likely exited (expected condition)
        return (0.0, 0.0)

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
        if agent_id not in self.agent_ids:
            # Agent may have exited - this is expected
            logger.debug(f"Cannot set target for agent {agent_id} - already exited")
            return

        jps_id = self.agent_ids[agent_id]
        self.agent_targets[agent_id] = target

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
        if agent_id not in self.agent_ids:
            # Agent may have exited - this is expected
            logger.debug(f"Cannot set exit for agent {agent_id} - already exited")
            return

        if exit_name not in self.evacuation_journeys:
            raise KeyError(
                f"Unknown exit: {exit_name}. "
                f"Available exits: {list(self.evacuation_journeys.keys())}"
            )

        jps_id = self.agent_ids[agent_id]
        journey_id = self.evacuation_journeys[exit_name]
        stage_id = self.evacuation_exits[exit_name]

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
        if agent_id not in self.agent_ids:
            # Agent may have exited - this is expected
            logger.debug(f"Cannot set speed for agent {agent_id} - already exited")
            return

        jps_id = self.agent_ids[agent_id]

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
        if agent_id not in self.agent_ids:
            return []

        center_pos = self.get_agent_position(agent_id)
        if center_pos == (0.0, 0.0):
            return []

        nearby = []

        # Get all agents currently in simulation
        all_agents = list(self.simulation.agents())

        for agent in all_agents:
            # Skip self
            other_id = self.jps_to_concordia.get(agent.id)
            if not other_id or other_id == agent_id:
                continue

            # Calculate distance
            dx = agent.position[0] - center_pos[0]
            dy = agent.position[1] - center_pos[1]
            dist = (dx**2 + dy**2) ** 0.5

            if dist <= radius:
                # Determine if moving based on orientation (has target)
                is_moving = hasattr(agent, "orientation") and agent.orientation is not None

                nearby.append(
                    {
                        "id": other_id,
                        "distance": dist,
                        "position": (float(agent.position[0]), float(agent.position[1])),
                        "is_moving": is_moving,
                        "target_exit": None,  # Will be enriched by HybridSimulationRunner
                    }
                )

        return nearby

    def get_simulation_time(self) -> float:
        """Get current simulation time in seconds."""
        return self.current_step * self.dt

    def get_all_agent_positions(self) -> dict[str, tuple[float, float]]:
        """
        Get positions of all agents for visualization.

        Returns:
            Dictionary mapping Concordia agent IDs to (x, y) positions
        """
        positions = {}

        for agent in self.simulation.agents():
            concordia_id = self.jps_to_concordia.get(agent.id)
            if concordia_id:
                positions[concordia_id] = (
                    float(agent.position[0]),
                    float(agent.position[1]),
                )

        return positions

    def get_geometry(self) -> dict[str, Any]:
        """
        Get geometry information for visualization.

        Returns:
            Dictionary with walkable areas, entrances, platforms, etc.
        """
        return {
            "walkable_areas": self.walkable_areas,
            "walkable_areas_with_obstacles": self.walkable_areas_with_obstacles,
            "entrance_areas": self.entrance_areas,
            "platform_areas": self.platform_areas,
            "obstacles": self.obstacles,
            "evacuation_exits": {
                name: self.entrance_areas.get(name)
                for name in self.evacuation_exits.keys()
                if name in self.entrance_areas
            },
        }
