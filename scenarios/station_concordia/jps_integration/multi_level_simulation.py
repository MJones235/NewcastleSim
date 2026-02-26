"""
Multi-level JuPedSim simulation for Monument Station.

Manages multiple levels (concourse + platforms) and agent transfers between them
via escalators. Each level has its own JuPedSim simulation instance.
"""

import math
import random
from pathlib import Path
from typing import Any

from shapely.geometry import Point

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
        # Positions used for transfers in the current step (cleared each step).
        # Prevents same-step transfers from landing on top of each other.
        self._pending_spawn_positions: list[tuple[float, float]] = []
        # Transfers deferred because the landing zone was too crowded.
        # Retried at the start of the next step.
        self._deferred_transfers: list[tuple[str, str, str]] = (
            []
        )  # (agent_id, current_level, exit_name)
        # Cooldown: minimum steps between consecutive transfers for the same agent.
        # At dt=0.05 s, 100 steps = 5 seconds — enough time to walk clear of the
        # arrival zone before a return trip could be triggered accidentally.
        self._transfer_cooldown_steps: int = 100
        self._last_transfer_step: dict[str, int] = {}  # agent_id -> step number

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
        # Reset same-step spawn tracking so each step starts fresh.
        self._pending_spawn_positions.clear()

        # Retry any transfers that were deferred last step because the zone was crowded.
        if self._deferred_transfers:
            deferred = self._deferred_transfers[:]
            self._deferred_transfers.clear()
            for agent_id, from_level, esc_name in deferred:
                logger.info(f"Retrying deferred transfer for {agent_id} via {esc_name}")
                self._transfer_agent_through_escalator(agent_id, from_level, esc_name)

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

                # Enforce cooldown to prevent immediate bounce-back transfers.
                last_step = self._last_transfer_step.get(agent_id, -self._transfer_cooldown_steps)
                steps_since = self.current_step - last_step
                if steps_since < self._transfer_cooldown_steps:
                    remaining_s = (self._transfer_cooldown_steps - steps_since) * self.dt
                    logger.warning(
                        f"Agent {agent_id} tried to transfer again via {exit_name} only "
                        f"{steps_since} steps after last transfer (cooldown: "
                        f"{self._transfer_cooldown_steps} steps). "
                        f"Ignoring for {remaining_s:.1f}s more."
                    )
                    continue

                # This is an escalator exit - transfer to target level
                logger.info(
                    f"Agent {agent_id} reached escalator exit {exit_name} on level {level_id} - initiating transfer"
                )
                self._transfer_agent_through_escalator(agent_id, level_id, exit_name)

    def _transfer_agent_through_escalator(self, agent_id: str, current_level: str, exit_name: str):
        """
        Transfer an agent from one level to another through an escalator.

        The up/down suffix in the exit name is for agent decision-making only.
        For the physical transfer, only the escalator letter (a-f) matters:
        an agent exiting through any escalator_X_* on level N spawns at the
        escalator_X zone on the other level, regardless of direction.

        Args:
            agent_id: Concordia agent ID
            current_level: Current level ID (e.g., "0" or "-1")
            exit_name: Name of the escalator exit (e.g., "escalator_a_down")
        """
        # Two levels only: flip between them
        target_level = "-1" if current_level == "0" else "0"

        if target_level not in self.simulations:
            logger.warning(f"Target level {target_level} not in simulation")
            return

        # Extract escalator letter — only this matters for locating the arrival zone.
        # exit_name format: "escalator_{letter}_{direction}" e.g. "escalator_a_down"
        parts = exit_name.split("_")
        if len(parts) < 2:
            logger.error(f"Cannot parse escalator letter from '{exit_name}'")
            return
        esc_letter = parts[1]  # 'a', 'b', 'c', etc.

        # Find whatever zone exists for this letter on the target level.
        # Zone naming: L{level}_esc_{letter}_{direction}
        target_zone_name = next(
            (
                k
                for k in self.transfer_manager.escalator_zones
                if k.startswith(f"L{target_level}_esc_{esc_letter}_")
            ),
            None,
        )

        if target_zone_name is None:
            logger.error(
                f"No arrival zone found for escalator '{esc_letter}' on level {target_level}. "
                f"Available: {list(self.transfer_manager.escalator_zones.keys())}"
            )
            return

        target_zone_poly = self.transfer_manager.escalator_zones[target_zone_name]
        centroid = target_zone_poly.centroid

        # Erode the polygon by JuPedSim's minimum boundary clearance (0.2 m) plus a
        # small margin so random candidates are never too close to walls.
        BOUNDARY_MARGIN = 0.3
        safe_zone = target_zone_poly.buffer(-BOUNDARY_MARGIN)
        if safe_zone.is_empty:
            # Polygon too small to erode — fall back to centroid only
            safe_zone = target_zone_poly

        # Choose a spawn position that doesn't collide with:
        #   (a) agents already present on the target level, and
        #   (b) other agents being transferred to this level in the same step.
        MIN_AGENT_SEP = 0.4
        existing_positions = (
            list(self.simulations[target_level].get_all_agent_positions().values())
            + self._pending_spawn_positions
        )

        spawn_pos = None
        for _attempt in range(60):
            angle = random.uniform(0, 2 * math.pi)
            radius = random.uniform(0.0, 0.7)
            candidate = (
                centroid.x + math.cos(angle) * radius,
                centroid.y + math.sin(angle) * radius,
            )
            if not safe_zone.contains(Point(candidate)):
                continue
            if any(
                math.hypot(candidate[0] - p[0], candidate[1] - p[1]) < MIN_AGENT_SEP
                for p in existing_positions
            ):
                continue
            spawn_pos = candidate
            break

        if spawn_pos is None:
            # Escalator zone is too crowded — wait until next step rather than crash.
            logger.warning(
                f"Cannot find free spawn point for {agent_id} in {target_zone_name} "
                f"({len(existing_positions)} agents nearby). Deferring transfer."
            )
            self._deferred_transfers.append((agent_id, current_level, exit_name))
            return

        self._pending_spawn_positions.append(spawn_pos)

        # Spawn agent in target level
        try:
            self.simulations[target_level].add_agent(agent_id, spawn_pos)
            self.agent_levels[agent_id] = target_level
            self.recently_transferred_agents.add(agent_id)
            self._last_transfer_step[agent_id] = self.current_step

            logger.info(
                f"Transferred agent {agent_id} from level {current_level} to {target_level} "
                f"through {exit_name} at {spawn_pos}"
            )

            # Assign a temporary waypoint inside the main walkable area of the
            # target level so the agent keeps moving until the next decision
            # cycle gives them a real goal.  We do NOT route to an exit here
            # because the agent may have transferred to catch a train, not to
            # leave the building.
            level_sim = self.simulations[target_level]
            try:
                walkable = level_sim.geometry_manager.walkable_areas_with_obstacles
                # Use the largest walkable polygon as the roaming area
                main_poly = max(walkable.values(), key=lambda p: p.area)
                safe_main = main_poly.buffer(-0.3)
                if safe_main.is_empty:
                    safe_main = main_poly
                main_centroid = safe_main.centroid
                temp_pos = None
                for _wp_attempt in range(60):
                    wp_angle = random.uniform(0, 2 * math.pi)
                    wp_radius = random.uniform(
                        0.5, min(3.0, safe_main.bounds[2] - safe_main.bounds[0]) / 3
                    )
                    wp_candidate = (
                        main_centroid.x + math.cos(wp_angle) * wp_radius,
                        main_centroid.y + math.sin(wp_angle) * wp_radius,
                    )
                    if safe_main.contains(Point(wp_candidate)):
                        temp_pos = wp_candidate
                        break
                if temp_pos is None:
                    temp_pos = (main_centroid.x, main_centroid.y)
                level_sim.set_agent_target(agent_id, temp_pos)
                logger.debug(
                    f"Assigned temporary waypoint {temp_pos} to "
                    f"transferred agent {agent_id} on level {target_level}"
                )
            except Exception as dest_err:
                logger.warning(f"Could not assign temporary waypoint to {agent_id}: {dest_err}")

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
