"""
Multi-level JuPedSim simulation for Monument Station.

Manages multiple levels (concourse + platforms) and agent transfers between them
via escalators. Each level has its own JuPedSim simulation instance.
"""

from pathlib import Path
from typing import Any

from scenarios.common.logger import get_logger
from scenarios.station_concordia.coordination.level_transfer_manager import LevelTransferManager
from scenarios.station_concordia.jps_integration.jupedsim_integration import (
    ConcordiaJuPedSimulation,
)

logger = get_logger(__name__)


class MultiLevelJuPedSimulation:
    """
    Manages multiple JuPedSim simulations for multi-level stations.

    Each level has its own simulation instance, and agents can transfer
    between levels via escalators/stairs.
    """

    def __init__(
        self,
        network_path: Path,
        dt: float = 0.05,
        exit_radius: float = 10.0,
        levels: list[str] | None = None,
    ):
        """
        Initialize multi-level simulation.

        Args:
            network_path: Path to network directory containing level_*.xml files
            dt: Timestep in seconds
            exit_radius: Radius of circular exits in meters
            levels: List of level IDs to load (default: ["0", "-1"])
        """
        self.dt = dt
        self.exit_radius = exit_radius
        self.network_path = Path(network_path)
        self.current_step = 0
        self.is_complete = False

        if levels is None:
            levels = ["0", "-1"]
        self.levels = levels

        # Create simulation instance for each level
        self.simulations: dict[str, ConcordiaJuPedSimulation] = {}
        for level_id in levels:
            logger.info(f"Initializing level {level_id}...")
            self.simulations[level_id] = ConcordiaJuPedSimulation(
                network_path=network_path,
                dt=dt,
                exit_radius=exit_radius,
                level_id=level_id,
            )

        # Track which level each agent is on
        self.agent_levels: dict[str, str] = {}  # agent_id -> level_id
        self.recently_transferred_agents: set[str] = set()

        # Setup level transfer manager
        self.transfer_manager = LevelTransferManager(network_path, levels)

        logger.info(
            f"Multi-level simulation initialized with {len(self.simulations)} levels: "
            f"{', '.join(levels)}"
        )
        logger.info(f"Transfer info: {self.transfer_manager.get_transfer_info()}")

    @property
    def geometry_manager(self):
        """
        Get geometry manager from level 0 (concourse level).

        For multi-level simulations, this exposes the concourse geometry
        which contains the street exits and main walkable areas.

        Returns:
            Geometry manager from level 0
        """
        return self.simulations["0"].geometry_manager

    def add_agent(
        self,
        agent_id: str,
        position: tuple[float, float],
        walking_speed: float = 1.34,
        level_id: str = "0",
    ) -> None:
        """
        Add an agent to a specific level.

        Args:
            agent_id: Concordia agent ID
            position: Initial (x, y) position
            walking_speed: Desired walking speed in m/s
            level_id: Level to spawn on (default: "0")
        """
        if level_id not in self.simulations:
            raise ValueError(f"Level {level_id} not loaded")

        self.simulations[level_id].add_agent(agent_id, position, walking_speed)
        self.agent_levels[agent_id] = level_id

        logger.info(f"Added agent {agent_id} to level {level_id} at {position}")

    def step(self) -> bool:
        """
        Advance all level simulations and process agent transfers between levels.

        Returns:
            True if simulation should continue, False if complete
        """
        if self.is_complete:
            return False

        # Step 1: Check for agents that exited through escalators and transfer them
        self._process_escalator_exits()

        # Step 2: Step each level's simulation
        any_active = False
        for sim in self.simulations.values():
            if sim.step():
                any_active = True

        self.current_step += 1

        # Check if simulation is complete (no agents left anywhere)
        total_agents = sum(sim.simulation.agent_count() for sim in self.simulations.values())
        if total_agents == 0:
            logger.info("All agents have exited the simulation")
            self.is_complete = not any_active
            return False

        return True

    def _process_escalator_exits(self):
        """
        Check each level for agents that have exited through escalators.

        Escalators are exits that connect two levels. When an agent exits through
        an escalator on one level, they are spawned into the target level.
        """
        # Check each level for agent exits
        for level_id, sim in self.simulations.items():
            exited_agents = sim.check_exits()

            if exited_agents:
                logger.info(
                    f"Level {level_id}: {len(exited_agents)} agents exited - {exited_agents}"
                )
            else:
                logger.debug(f"Level {level_id}: No agents exited")

            # Process each exited agent
            for agent_id, exit_name in exited_agents.items():
                # Check if exit is an escalator (starts with "escalator_")
                if not exit_name.startswith("escalator_"):
                    logger.info(f"Agent {agent_id} exited station through street exit {exit_name}")
                    # Remove from level tracking - agent has truly exited station
                    if agent_id in self.agent_levels:
                        del self.agent_levels[agent_id]
                    continue

                # This is an escalator exit - transfer to target level
                logger.info(
                    f"Agent {agent_id} reached escalator exit {exit_name} on level {level_id} - initiating transfer"
                )
                self._transfer_agent_through_escalator(agent_id, level_id, exit_name)

    def _transfer_agent_through_escalator(self, agent_id: str, current_level: str, exit_name: str):
        """
        Transfer an agent from one level to another through an escalator.

        Escalators have the same name on both levels (e.g., escalator_a_up exists on both level -1 and level 0).
        When an agent exits through an escalator, they spawn at the same escalator zone on the other level.

        Args:
            agent_id: Concordia agent ID
            current_level: Current level ID (e.g., "0" or "-1")
            exit_name: Name of the escalator exit (e.g., "escalator_a_up")
        """
        # Determine target level (flip between 0 and -1)
        if current_level == "0":
            target_level = "-1"
        elif current_level == "-1":
            target_level = "0"
        else:
            logger.error(f"Unknown current level: {current_level}")
            return

        # Make sure target level exists
        if target_level not in self.simulations:
            logger.warning(f"Target level {target_level} not in simulation")
            return

        # Get the corresponding escalator zone on the target level
        # The escalator has the same name on both levels (e.g., escalator_a_up)
        target_zone_name = (
            f"L{target_level}_esc_{exit_name.split('_')[1]}_{exit_name.split('_')[2]}"
        )

        # Get spawn position inside the escalator zone
        if target_zone_name not in self.transfer_manager.escalator_zones:
            logger.error(
                f"Target escalator zone not found for agent {agent_id}: {target_zone_name}"
            )
            logger.error(f"Available zones: {list(self.transfer_manager.escalator_zones.keys())}")
            logger.error(
                f"Exit name parsing: {exit_name} -> parts [1]={exit_name.split('_')[1]}, parts [2]={exit_name.split('_')[2]}"
            )
            return

        target_zone_poly = self.transfer_manager.escalator_zones[target_zone_name]
        spawn_pos = (target_zone_poly.centroid.x, target_zone_poly.centroid.y)

        # Spawn agent in target level
        try:
            self.simulations[target_level].add_agent(agent_id, spawn_pos)
            self.agent_levels[agent_id] = target_level
            self.recently_transferred_agents.add(agent_id)

            logger.info(
                f"Transferred agent {agent_id} from level {current_level} to {target_level} "
                f"through {exit_name} at {spawn_pos}"
            )
        except Exception as e:
            logger.error(f"Failed to transfer agent {agent_id} to level {target_level}: {e}")
            # Remove agent from tracking if transfer failed
            if agent_id in self.agent_levels:
                del self.agent_levels[agent_id]

    def consume_recently_transferred_agents(self) -> set[str]:
        """Return and clear agents transferred since last consume call."""
        transferred = set(self.recently_transferred_agents)
        self.recently_transferred_agents.clear()
        return transferred

    def get_agent_position(self, agent_id: str) -> tuple[float, float] | None:
        """
        Get agent's current position.

        Args:
            agent_id: Concordia agent ID

        Returns:
            Agent's (x, y) position, or None if agent has exited
        """
        if agent_id not in self.agent_levels:
            return None

        level_id = self.agent_levels[agent_id]
        return self.simulations[level_id].get_agent_position(agent_id)

    def get_agent_level(self, agent_id: str) -> str | None:
        """Get the level an agent is currently on."""
        return self.agent_levels.get(agent_id)

    def set_agent_target(self, agent_id: str, target: tuple[float, float]) -> None:
        """Set an agent's movement target on their current level."""
        if agent_id not in self.agent_levels:
            return

        level_id = self.agent_levels[agent_id]
        self.simulations[level_id].set_agent_target(agent_id, target)

    def set_agent_evacuation_exit(self, agent_id: str, exit_name: str) -> None:
        """
        Direct an agent to a specific evacuation exit on their current level.

        The exit must exist on the agent's current level. For multi-level evacuations:
        - On platform levels: Agents route to escalator exits (e.g., "escalator_a_up")
        - On concourse levels: Agents route to street exits (e.g., "eldon_square")

        Args:
            agent_id: ID of the agent
            exit_name: Name of the exit to route - must exist on current level
        """
        if agent_id not in self.agent_levels:
            return

        level_id = self.agent_levels[agent_id]
        level_sim = self.simulations[level_id]

        # Check if exit exists on this level
        if exit_name not in level_sim.exit_manager.evacuation_exits:
            logger.warning(
                f"Agent {agent_id} on level {level_id} tried to route to exit '{exit_name}' "
                f"which doesn't exist on this level. Available exits: "
                f"{list(level_sim.exit_manager.evacuation_exits.keys())}"
            )
            return

        # Route to the exit on this level
        level_sim.set_agent_evacuation_exit(agent_id, exit_name)

    def set_agent_speed(self, agent_id: str, speed: float) -> None:
        """Set an agent's walking speed."""
        if agent_id not in self.agent_levels:
            return

        level_id = self.agent_levels[agent_id]
        self.simulations[level_id].set_agent_speed(agent_id, speed)

    def get_nearby_agents(self, agent_id: str, radius: float) -> list[dict[str, Any]]:
        """Get information about agents within radius on the same level."""
        if agent_id not in self.agent_levels:
            return []

        level_id = self.agent_levels[agent_id]
        return self.simulations[level_id].get_nearby_agents(agent_id, radius)

    def get_simulation_time(self) -> float:
        """Get current simulation time in seconds."""
        return self.current_step * self.dt

    def get_all_agent_positions(self) -> dict[str, tuple[float, float]]:
        """
        Get positions of all agents across all levels.

        Returns:
            Dictionary mapping agent IDs to (x, y) positions
        """
        all_positions = {}
        for sim in self.simulations.values():
            positions = sim.get_all_agent_positions()
            all_positions.update(positions)
        return all_positions

    def get_geometry(self, level_id: str | None = None) -> dict[str, Any]:
        """
        Get geometry information for visualization.

        Args:
            level_id: Specific level to get geometry for, or None for all levels

        Returns:
            Geometry data for the requested level(s)
        """
        if level_id is not None:
            return self.simulations[level_id].get_geometry()

        # Return all levels
        all_geometry = {}
        for lid, sim in self.simulations.items():
            all_geometry[f"level_{lid}"] = sim.get_geometry()
        return all_geometry

    def generate_spawn_positions(
        self, num_agents: int, seed: int = 42
    ) -> list[tuple[float, float, str]]:
        """
        Generate spawn positions distributed across all levels.

        Returns list of (x, y, level_id) tuples so agents can be spawned on correct level.
        Distribution is proportional to walkable area on each level.

        Args:
            num_agents: Total number of agents to spawn
            seed: Random seed for reproducibility

        Returns:
            List of (x, y, level_id) tuples
        """
        import random

        random.seed(seed)

        # Calculate total walkable area per level
        level_areas = {}
        for level_id, sim in self.simulations.items():
            total_area = sum(
                poly.area for poly in sim.geometry_manager.walkable_areas_with_obstacles.values()
            )
            level_areas[level_id] = total_area

        total_area = sum(level_areas.values())

        # Distribute agents proportionally by area
        spawn_positions = []
        agents_placed = 0

        for idx, (level_id, area) in enumerate(sorted(level_areas.items())):
            # Calculate proportional number of agents for this level
            if idx == len(level_areas) - 1:
                # Last level gets remainder to ensure exact count
                level_agents = num_agents - agents_placed
            else:
                level_agents = int(num_agents * (area / total_area))

            if level_agents > 0:
                # Generate positions on this level
                positions = self.simulations[level_id].generate_spawn_positions(
                    level_agents, seed + idx
                )

                # Add level_id to each position
                for x, y in positions:
                    spawn_positions.append((x, y, level_id))

                agents_placed += len(positions)
                logger.info(f"Spawning {len(positions)} agents on level {level_id}")

        return spawn_positions
