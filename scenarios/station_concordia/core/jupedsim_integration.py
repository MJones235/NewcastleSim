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
        """Create evacuation exit stages at entrance locations."""
        evacuation_exits = {}
        evacuation_journeys = {}

        walkable_geometry = GeometryProcessor.combine_geometry(
            list(self.walkable_areas_with_obstacles.values())
        )

        for entrance_name, entrance_polygon in self.entrance_areas.items():
            try:
                exit_id = self._create_convex_exit_from_polygon(
                    entrance_name, entrance_polygon, walkable_geometry
                )
                if exit_id is None:
                    raise RuntimeError("Unable to place convex exit within walkable area")

                evacuation_exits[entrance_name] = exit_id

                # Create journey to this exit
                journey_id = self.stage_manager.create_simple_exit_journey(
                    journey_name=f"journey_to_{entrance_name}", exit_id=exit_id
                )
                evacuation_journeys[entrance_name] = journey_id

                logger.info(
                    f"  Created evacuation exit '{entrance_name}' "
                    f"(exit={exit_id}, journey={journey_id})"
                )
            except Exception as e:
                logger.warning(f"  Failed to create exit at '{entrance_name}': {e}")
                continue

        if not evacuation_exits:
            # If entrance-based exits fail, try creating exits within walkable areas
            logger.warning("Failed to create entrance-based exits, using walkable area exits")
            try:
                return self._setup_fallback_exits()
            except Exception as e:
                logger.error(f"Fallback exit creation also failed: {e}", exc_info=True)
                raise

        return evacuation_exits, evacuation_journeys

    def _create_convex_exit_from_polygon(
        self,
        exit_name: str,
        polygon,
        walkable_geometry,
    ) -> int | None:
        """Create a convex rectangular exit centered on polygon centroid.

        Uses the entrance centroid (or a nearby point inside the walkable
        geometry) and creates a small convex rectangle around it.
        """
        centroid = polygon.centroid

        if not walkable_geometry.contains(centroid):
            intersection = polygon.intersection(walkable_geometry)
            if not intersection.is_empty:
                centroid = intersection.representative_point()
            else:
                centroid = polygon.representative_point()

        # Try progressively smaller exits without strict containment checks
        for width, height in [(6.0, 6.0), (4.0, 4.0), (2.0, 2.0)]:
            exit_coords = [
                (centroid.x - width / 2, centroid.y - height / 2),
                (centroid.x + width / 2, centroid.y - height / 2),
                (centroid.x + width / 2, centroid.y + height / 2),
                (centroid.x - width / 2, centroid.y + height / 2),
            ]

            try:
                return self.stage_manager.create_exit_at_coordinates(exit_name, exit_coords)
            except Exception:
                continue

        return None

    def _setup_fallback_exits(self) -> tuple[dict[str, int], dict[str, int]]:
        """Create fallback exits within the walkable areas."""
        logger.info("Creating fallback exits...")
        evacuation_exits = {}
        evacuation_journeys = {}

        # Use multiple points within the main walkable area as exits
        main_area = list(self.walkable_areas.values())[0]
        logger.info(f"Main walkable area bounds: {main_area.bounds}, area: {main_area.area:.1f} m²")

        # Get interior points by buffering inward
        interior_area = main_area.buffer(-5.0)  # 5m inward from edges

        if interior_area.is_empty:
            logger.warning("Walkable area too small to create interior buffer, using main area")
            interior_area = main_area

        # Try to create exit at the centroid
        centroid = interior_area.centroid
        exit_positions = [("exit_center", (centroid.x, centroid.y))]
        logger.info(f"Testing centroid exit position: {(centroid.x, centroid.y)}")

        # Add boundary points if we can find them
        bounds = main_area.bounds
        test_points = [
            ("exit_north", ((bounds[0] + bounds[2]) / 2, bounds[3] - 10)),
            ("exit_south", ((bounds[0] + bounds[2]) / 2, bounds[1] + 10)),
            ("exit_east", (bounds[2] - 10, (bounds[1] + bounds[3]) / 2)),
            ("exit_west", (bounds[0] + 10, (bounds[1] + bounds[3]) / 2)),
        ]

        for name, pos in test_points:
            point = Point(pos)
            if main_area.contains(point):
                exit_positions.append((name, pos))

        # Try to create exits
        logger.info(f"Trying to create {len(exit_positions)} exit positions...")
        for exit_name, position in exit_positions:
            logger.debug(f"Testing exit '{exit_name}' at {position}")
            try:
                # Create small circular exit  (2m radius)
                exit_polygon = Point(position).buffer(2.0)

                # Ensure exit is fully within walkable area
                if not main_area.contains(exit_polygon):
                    logger.debug("  2m exit doesn't fit, trying 1m...")
                    # Try smaller radius
                    exit_polygon = Point(position).buffer(1.0)
                    if not main_area.contains(exit_polygon):
                        logger.debug(f"  Skipping {exit_name} - outside walkable area")
                        continue

                logger.debug(f"  Adding exit stage for '{exit_name}'...")
                exit_id = self.simulation.add_exit_stage(polygon=exit_polygon)
                evacuation_exits[exit_name] = exit_id
                logger.debug(f"  Exit stage created with ID: {exit_id}")

                # Create journey to this exit
                journey = jps.JourneyDescription([exit_id])
                journey_id = self.simulation.add_journey(journey)
                evacuation_journeys[exit_name] = journey_id

                logger.info(
                    f"  Created fallback exit '{exit_name}' (exit={exit_id}, journey={journey_id})"
                )

            except Exception as e:
                logger.warning(f"  Failed to create fallback exit '{exit_name}': {e}")
                import traceback

                traceback.print_exc()
                continue

        if not evacuation_exits:
            raise RuntimeError("Failed to create any evacuation exits (including fallbacks)")

        return evacuation_exits, evacuation_journeys

    def add_agent(self, agent_id: str, position: tuple[float, float]) -> bool:
        """
        Add an agent to the simulation.

        Args:
            agent_id: Concordia agent ID
            position: Initial (x, y) position

        Returns:
            True if agent added successfully
        """
        try:
            # Use first available exit and its stage
            if not self.evacuation_exits:
                logger.error("No evacuation exits available")
                return False

            exit_name = list(self.evacuation_exits.keys())[0]
            stage_id = self.evacuation_exits[exit_name]
            journey_id = self.evacuation_journeys[exit_name]

            logger.debug(
                f"Using exit '{exit_name}' (stage={stage_id}, journey={journey_id}) for agent {agent_id}"
            )

            # Add agent to JuPedSim
            # Set both journey_id AND stage_id (the first stage in the journey)
            jps_id = self.simulation.add_agent(
                jps.CollisionFreeSpeedModelAgentParameters(
                    position=position,
                    journey_id=journey_id,
                    stage_id=stage_id,  # Start at the first stage of the journey
                )
            )

            # Track agent mapping
            self.agent_ids[agent_id] = jps_id
            self.jps_to_concordia[jps_id] = agent_id

            logger.debug(f"Added agent {agent_id} at position {position} (JuPedSim ID: {jps_id})")
            return True

        except Exception as e:
            logger.error(f"Failed to add agent {agent_id}: {e}")
            return False

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
        """Get agent's current position."""
        if agent_id not in self.agent_ids:
            logger.warning(f"Agent {agent_id} not found in simulation")
            return (0.0, 0.0)

        jps_id = self.agent_ids[agent_id]

        try:
            # Use simulation.agents() to get agent data
            for agent in self.simulation.agents():
                if agent.id == jps_id:
                    return (float(agent.position[0]), float(agent.position[1]))
            # Agent not found (may have exited)
            logger.debug(f"Agent {agent_id} (JPS {jps_id}) has likely exited")
            return (0.0, 0.0)
        except Exception as e:
            logger.warning(f"Error getting position for agent {agent_id}: {e}")
            return (0.0, 0.0)

    def set_agent_target(self, agent_id: str, target: tuple[float, float]):
        """
        Set an agent's movement target by creating a waypoint.

        Args:
            agent_id: Concordia agent ID
            target: Target (x, y) position
        """
        if agent_id not in self.agent_ids:
            logger.warning(f"Cannot set target for unknown agent {agent_id}")
            return

        jps_id = self.agent_ids[agent_id]
        self.agent_targets[agent_id] = target

        try:
            # Create a waypoint stage at the target location
            stage_id = self.simulation.add_waypoint_stage(target, distance=2.0)

            # Create a journey to this waypoint
            journey = jps.JourneyDescription([stage_id])
            journey_id = self.simulation.add_journey(journey)

            # Update agent's journey and stage
            self.simulation.switch_agent_journey(
                agent_id=jps_id,
                journey_id=journey_id,
                stage_id=stage_id,  # Start at the waypoint stage
            )

            logger.debug(f"Set target for agent {agent_id} to {target}")

        except Exception as e:
            logger.warning(f"Failed to set target for agent {agent_id}: {e}")

    def set_agent_evacuation_exit(self, agent_id: str, exit_name: str):
        """
        Direct an agent to a specific evacuation exit.

        Args:
            agent_id: Concordia agent ID
            exit_name: Name of the evacuation exit
        """
        if agent_id not in self.agent_ids:
            logger.warning(f"Cannot set exit for unknown agent {agent_id}")
            return

        if exit_name not in self.evacuation_journeys:
            logger.warning(f"Unknown exit: {exit_name}")
            return

        jps_id = self.agent_ids[agent_id]
        journey_id = self.evacuation_journeys[exit_name]
        stage_id = self.evacuation_exits[exit_name]  # Get the exit stage ID

        try:
            self.simulation.switch_agent_journey(
                agent_id=jps_id,
                journey_id=journey_id,
                stage_id=stage_id,  # Start at the exit stage
            )
            logger.debug(f"Directed agent {agent_id} to exit '{exit_name}'")

        except Exception as e:
            logger.warning(f"Failed to set evacuation exit for agent {agent_id}: {e}")

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

        # Get all agents
        try:
            all_agents = list(self.simulation.agents())
        except Exception:
            return []

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

        try:
            for agent in self.simulation.agents():
                concordia_id = self.jps_to_concordia.get(agent.id)
                if concordia_id:
                    positions[concordia_id] = (float(agent.position[0]), float(agent.position[1]))
        except Exception as e:
            logger.warning(f"Failed to get all agent positions: {e}")

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
