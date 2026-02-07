"""
Hybrid simulation runner that integrates Concordia with JuPedSim.

This module implements the translation layer between:
- Concordia: Agent cognition and decision-making
- JuPedSim: Pedestrian movement simulation

Key features:
- Event-driven LLM queries (not every timestep)
- Batch processing of agent decisions
- Translation of NL actions to waypoints
- Observation generation from simulation state
"""

import asyncio
import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from concordia.associative_memory import basic_associative_memory
from concordia.language_model import language_model
from concordia.typing import entity as entity_lib
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

try:
    import jupedsim as jps
except ImportError:
    jps = None
    print("Warning: jupedsim not available")

from scenarios.common.logger import get_logger
from scenarios.station_concordia.core.evacuation_agent import EvacuationAgent
from scenarios.station_concordia.core.game_master import ActionTranslator, ObservationGenerator

logger = get_logger(__name__)


# Performance timing utility
class PerformanceTimer:
    """Simple performance timer for profiling simulation bottlenecks."""

    def __init__(self):
        self.timings = {}
        self.counts = {}
        self.parallel_operations = set()  # Track which operations run in parallel

    def record(self, name: str, duration: float, is_parallel: bool = False):
        """Record a timing measurement."""
        if name not in self.timings:
            self.timings[name] = 0.0
            self.counts[name] = 0

        if is_parallel:
            # For parallel operations, store max duration instead of sum
            self.parallel_operations.add(name)
            self.timings[name] = max(self.timings[name], duration)
            self.counts[name] += 1
        else:
            # For sequential operations, sum as normal
            self.timings[name] += duration
            self.counts[name] += 1

    @contextmanager
    def measure(self, name: str, is_parallel: bool = False):
        """Context manager for timing a block of code.

        Args:
            name: Name of the operation
            is_parallel: If True, uses max instead of sum (for parallel operations)
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            self.record(name, duration, is_parallel=is_parallel)

    def report(self):
        """Generate performance report."""
        if not self.timings:
            return "No timings recorded"

        lines = ["\n=== PERFORMANCE PROFILE (Wall-Clock Time) ==="]
        total_time = sum(self.timings.values())

        # Sort by total time descending
        sorted_items = sorted(self.timings.items(), key=lambda x: x[1], reverse=True)

        for name, total in sorted_items:
            count = self.counts[name]
            avg = total / count if count > 0 else 0
            percent = (total / total_time * 100) if total_time > 0 else 0

            # Add indicator for parallel operations
            parallel_mark = " [parallel]" if name in self.parallel_operations else ""

            lines.append(
                f"{name:30s}: {total:8.3f}s total | {avg:8.3f}s avg | {count:5d} calls | {percent:5.1f}%{parallel_mark}"
            )

        lines.append(f"{'TOTAL':30s}: {total_time:8.3f}s")
        lines.append("=" * 80)

        return "\n".join(lines)


class HybridSimulationRunner:
    """
    Manages the hybrid Concordia + JuPedSim simulation.

    Architecture:
    1. JuPedSim runs continuously at fine time resolution (dt=0.05s)
    2. Concordia agents make decisions at coarse intervals (5-10s)
    3. Decisions are triggered by events (announcements, observations)
    4. Actions are translated to JuPedSim waypoints
    5. Simulation state is converted to observations for agents
    """

    def __init__(
        self,
        jupedsim_simulation: Any,  # StationSimulation instance
        agents_config: list[dict[str, Any]],
        station_layout: dict[str, Any],
        language_model: language_model.LanguageModel,
        embedder: Any,  # Sentence embedder function
        decision_interval: float = 5.0,
        max_steps: int = 3600,
        output_file: Path | None = None,
        test_scenarios: dict[str, Any] | None = None,
    ):
        """
        Initialize the hybrid simulation runner.

        Args:
            jupedsim_simulation: Configured JuPedSim simulation
            agents_config: List of agent configuration dictionaries
            station_layout: Station geometry and exit information
            language_model: LLM for Concordia agents
            embedder: Sentence embedding function
            decision_interval: Time between Concordia decisions (seconds)
            max_steps: Maximum simulation steps
        """
        self.jps_sim = jupedsim_simulation
        self.station_layout = station_layout
        self.model = language_model
        self.embedder = embedder
        self.decision_interval = decision_interval
        self.max_steps = max_steps
        self.output_file = output_file
        self.test_scenarios = test_scenarios or {}

        # Store LLM provider reference (for usage stats)
        # The language_model is an AzureLLMConcordia instance directly
        self.llm_provider = language_model if hasattr(language_model, "get_usage_stats") else None

        # Translation layer components
        self.action_translator = ActionTranslator(station_layout, language_model)
        self.observation_generator = ObservationGenerator(station_layout)

        # Build Concordia agents (each with their own memory bank)
        self.concordia_agents: dict[str, entity_lib.Entity] = {}
        self.agent_configs = agents_config

        # Phase 4.1: Help behavior tracking (must be initialized before _build_agents)
        self.agent_status: dict[str, str] = (
            {}
        )  # agent_id -> "EVACUATING"|"HELPING"|"WAITING"|"INJURED"
        self.agents_being_helped: dict[str, str] = {}  # helped_agent_id -> helper_agent_id
        self.help_events: list[dict[str, Any]] = []  # Track all help interactions
        self.active_helping_pairs: dict[str, dict[str, Any]] = (
            {}
        )  # helper_id -> {helped, start_time, duration}
        self.agent_original_speeds: dict[str, float] = (
            {}
        )  # Store original walking speeds for restoration

        self._build_agents()

        # Tracking
        self.last_decision_time = (
            -decision_interval
        )  # Start negative so first decision happens immediately
        self.current_sim_time = 0.0
        self.current_step = 0  # Track current simulation step for logging
        self.agent_decisions: dict[str, dict[str, Any]] = {}
        self.event_history: list[dict[str, Any]] = []
        self.last_observations: dict[str, str] = {}  # Cache observations for change detection
        self.last_actions: dict[str, str] = {}  # Cache actions to reuse

        # Phase 4.2: Route changing tracking
        self.agent_destinations: dict[str, str] = {}  # agent_id -> current exit name
        self.blocked_exits: set[str] = set()  # Set of blocked exit names

        # Phase 4.3: Waiting and information seeking tracking
        self.wait_events: list[dict[str, Any]] = []  # Track all wait decisions with reasons

        # Track exited agents (those who have evacuated)
        self.exited_agents: set[str] = set()  # agent_ids who have reached exits

        # Performance profiling
        self.perf_timer = PerformanceTimer()

    def _build_agents(self):
        """Build Concordia agents from configurations."""
        logger.info(f"Building {len(self.agent_configs)} Concordia agents...")

        for agent_config in self.agent_configs:
            agent_id = agent_config["id"]
            logger.info(f"Building {agent_id}...")

            # Create separate memory bank for each agent
            memory_bank = basic_associative_memory.AssociativeMemoryBank(
                sentence_embedder=self.embedder
            )

            # Create agent prefab
            prefab = EvacuationAgent(params=agent_config)

            # Build agent
            agent = prefab.build(
                model=self.model,
                memory_bank=memory_bank,
            )

            self.concordia_agents[agent_id] = agent

            # Phase 4.1: Initialize agent status
            if agent_config.get("is_injured", False):
                self.agent_status[agent_id] = "INJURED"
            else:
                self.agent_status[agent_id] = "EVACUATING"

            # Add initial memories
            self._initialize_agent_memory(agent, agent_config)

        logger.info(f"Built {len(self.concordia_agents)} Concordia agents")

    def _initialize_agent_memory(self, agent: entity_lib.Entity, config: dict[str, Any]):
        """Initialize an agent's memory with background knowledge."""
        # Add station layout as formative memory
        layout_description = self.observation_generator._describe_geometry()

        initial_memories = [
            "I am at a train station.",
            f"I am in the {config.get('initial_zone', 'platform')} area.",
            "I am waiting for my train.",
            "I am on my way to my destination.",
            layout_description,  # Station layout info
            "The station has clear signage for platforms and exits.",
            "I notice other passengers waiting and walking around.",
            "The atmosphere is calm and routine.",
        ]

        # Phase 4.1: Add injury-specific memories
        if config.get("is_injured", False):
            initial_memories.extend(
                [
                    "I am injured and moving slowly.",
                    "I may need assistance during the evacuation.",
                    "I am moving at a reduced pace due to my injury.",
                ]
            )

        for memory in initial_memories:
            agent.observe(memory)

    def run(self) -> dict[str, Any]:
        """
        Run the hybrid simulation.

        Returns:
            Dictionary with simulation results and statistics
        """
        logger.info("Starting hybrid Concordia + JuPedSim simulation")
        start_time = time.time()

        results = {
            "steps": 0,
            "sim_time": 0.0,
            "decisions_made": 0,
            "events_triggered": 0,
            "agents": {},
        }

        try:
            # Main simulation loop with progress bar
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]Simulating:"),
                BarColumn(complete_style="green", finished_style="bold green"),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TextColumn("•"),
                TextColumn("Step {task.completed}/{task.total}"),
                TextColumn("•"),
                TimeElapsedColumn(),
                TextColumn("•"),
                TimeRemainingColumn(),
            ) as progress:
                task = progress.add_task("simulation", total=self.max_steps)

                for step in range(self.max_steps):
                    step_start = time.perf_counter()
                    self.current_step = step

                    # Advance JuPedSim simulation
                    with self.perf_timer.measure("jupedsim_step"):
                        if not self._step_jupedsim():
                            logger.info("JuPedSim simulation complete")
                            break

                    self.current_sim_time = step * self.jps_sim.dt

                    # Check for agents who have exited and remove them
                    self._check_exited_agents()

                    # Phase 4.1: Update helping relationships (check for expired help durations)
                    self._update_helping_relationships()

                    # Check if it's time for Concordia decisions
                    if self._should_make_decisions():
                        with self.perf_timer.measure("agent_decisions_total"):
                            self._process_agent_decisions()

                    # Check for events
                    with self.perf_timer.measure("event_checking"):
                        self._check_and_trigger_events()

                    # Save positions every 10 steps (0.5s) for smooth visualization
                    # Writing every single step (0.05s) is too slow for file I/O
                    if self.output_file and step % 10 == 0:
                        with self.perf_timer.measure("file_io"):
                            self._save_incremental()

                    results["steps"] = step + 1
                    results["sim_time"] = self.current_sim_time

                    # Update progress bar
                    progress.update(task, advance=1)

                    # Pace simulation to real time for smooth visualization
                    if self.jps_sim.dt > 0:
                        elapsed = time.perf_counter() - step_start
                        sleep_time = self.jps_sim.dt - elapsed
                        if sleep_time > 0:
                            time.sleep(sleep_time)

        except KeyboardInterrupt:
            logger.info("Simulation interrupted by user")
        except Exception as e:
            logger.error(f"Simulation error: {e}", exc_info=True)

        # Compute final statistics
        elapsed_time = time.time() - start_time
        results["elapsed_time"] = elapsed_time
        results["decisions_made"] = sum(
            len(d.get("decisions", [])) for d in self.agent_decisions.values()
        )
        results["events_triggered"] = len(self.event_history)

        logger.info(
            f"Simulation complete: {results['steps']} steps, "
            f"{results['sim_time']:.1f}s sim time, "
            f"{elapsed_time:.1f}s real time"
        )

        # Print performance profile
        print(self.perf_timer.report())

        # Print financial report
        print(self._generate_financial_report())

        return results

    def _step_jupedsim(self) -> bool:
        """
        Advance JuPedSim simulation by one timestep.

        Returns:
            True if simulation should continue, False if complete
        """
        try:
            # TODO: This works with MockJuPedSim, verify with real JuPedSim
            return self.jps_sim.step()
        except Exception as e:
            logger.error(f"JuPedSim step error: {e}")
            return False

    def _should_make_decisions(self) -> bool:
        """Check if it's time for agents to make decisions."""
        return (self.current_sim_time - self.last_decision_time) >= self.decision_interval

    def _check_exited_agents(self):
        """Check for agents who have reached exits and mark them as exited."""
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
        if self.current_step % 50 == 0 and self.current_step > 0:
            logger.info(
                f"📊 Agent status: {active_agents} active, {exited_count} exited, "
                f"{total_agents} total (t={self.current_sim_time:.1f}s)"
            )

    def _update_helping_relationships(self):
        """Update active helping relationships and release them when duration expires."""
        if not self.active_helping_pairs:
            return

        expired_pairs = []
        for helper_id, pair_info in self.active_helping_pairs.items():
            helped_id = pair_info["helped"]
            start_time = pair_info["start_time"]
            duration = pair_info["duration"]
            phase = pair_info.get("phase", "traveling")

            # Phase 1: Approaching - check if helper has reached injured agent
            if phase == "approaching":
                helper_pos = self._get_agent_position(helper_id)
                injured_pos = self._get_agent_position(helped_id)

                # Calculate distance between helper and injured agent
                distance = (
                    (helper_pos[0] - injured_pos[0]) ** 2 + (helper_pos[1] - injured_pos[1]) ** 2
                ) ** 0.5

                # Get approach distance threshold from config
                help_config = self.test_scenarios.get("help_behavior", {})
                approach_distance = help_config.get("approach_distance", 1.5)

                # Injured agent should wait (speed = 0) for helper to arrive
                self.jps_sim.set_agent_speed(helped_id, 0.0)

                # Log progress occasionally
                if int(self.current_sim_time) % 5 == 0:  # Every 5 seconds
                    logger.debug(
                        f"🚶 {helper_id} approaching {helped_id}: distance={distance:.1f}m "
                        f"(threshold={approach_distance}m)"
                    )

                # If within approach distance, transition to traveling phase
                if distance < approach_distance:
                    pair_info["phase"] = "traveling"

                    # Phase 2: Traveling together
                    # Get assisted speed from config
                    help_config = self.test_scenarios.get("help_behavior", {})
                    assisted_speed = help_config.get("assisted_speed", 0.8)
                    self.jps_sim.set_agent_speed(helper_id, assisted_speed)
                    self.jps_sim.set_agent_speed(helped_id, assisted_speed)

                    # Both agents target the same exit (helper's current destination)
                    helper_exit = self.agent_destinations.get(helper_id)
                    if helper_exit:
                        if hasattr(self.jps_sim, "set_agent_evacuation_exit"):
                            self.jps_sim.set_agent_evacuation_exit(helped_id, helper_exit)
                            logger.debug(f"Set {helped_id} to follow {helper_id} to {helper_exit}")

                    logger.info(
                        f"🚶 {helper_id} reached {helped_id} - "
                        f"now traveling together at {assisted_speed} m/s toward {helper_exit}"
                    )

            # Phase 2: Traveling - continuously update helped agent to follow helper
            if phase == "traveling":
                # Check if helper has exited - if so, terminate helping relationship
                if helper_id in self.exited_agents:
                    logger.info(
                        f"🚪 {helper_id} has exited, releasing {helped_id} to evacuate independently"
                    )
                    expired_pairs.append(helper_id)

                    # Restore helped agent's speed and status
                    if helped_id in self.agent_original_speeds:
                        original_speed = self.agent_original_speeds[helped_id]
                        self.jps_sim.set_agent_speed(helped_id, original_speed)

                    self.agent_status[helped_id] = "INJURED"

                    # Remove from being helped tracking
                    if helped_id in self.agents_being_helped:
                        del self.agents_being_helped[helped_id]

                    # Skip to next pair (don't try to update positions)
                    continue

                # Ensure helper maintains their evacuation journey and assisted speed
                help_config = self.test_scenarios.get("help_behavior", {})
                assisted_speed = help_config.get("assisted_speed", 0.8)
                self.jps_sim.set_agent_speed(helper_id, assisted_speed)
                self.jps_sim.set_agent_speed(helped_id, assisted_speed)

                # Keep helped agent following helper by setting their target to helper's current position
                helper_pos = self._get_agent_position(helper_id)

                # Sanity check: if position is (0, 0), helper may have been removed
                if helper_pos == (0.0, 0.0) or helper_pos == (0, 0):
                    logger.warning(
                        f"⚠️ {helper_id} has invalid position (0,0), releasing {helped_id}"
                    )
                    expired_pairs.append(helper_id)
                    continue

                self.jps_sim.set_agent_target(helped_id, helper_pos)

                # Log occasionally to verify they're staying together
                if int(self.current_sim_time) % 5 == 0:  # Every 5 seconds
                    helped_pos = self._get_agent_position(helped_id)
                    distance = (
                        (helper_pos[0] - helped_pos[0]) ** 2 + (helper_pos[1] - helped_pos[1]) ** 2
                    ) ** 0.5
                    logger.debug(
                        f"👥 {helper_id} and {helped_id} traveling together (distance: {distance:.1f}m)"
                    )

            # Check if help duration has expired (only for traveling phase)
            if phase == "traveling" and self.current_sim_time >= start_time + duration:
                expired_pairs.append(helper_id)

                # Restore original speeds using JuPedSim agent.model.v0
                if helper_id in self.agent_original_speeds:
                    original_speed = self.agent_original_speeds[helper_id]
                    self.jps_sim.set_agent_speed(helper_id, original_speed)

                if helped_id in self.agent_original_speeds:
                    original_speed = self.agent_original_speeds[helped_id]
                    self.jps_sim.set_agent_speed(helped_id, original_speed)

                # Update statuses back to EVACUATING/INJURED
                self.agent_status[helper_id] = "EVACUATING"
                self.agent_status[helped_id] = "INJURED"  # Still injured but now independent

                # Remove from being helped tracking
                if helped_id in self.agents_being_helped:
                    del self.agents_being_helped[helped_id]

                logger.info(
                    f"👋 {helper_id} finished helping {helped_id} - "
                    f"both resuming independent evacuation"
                )

        # Remove expired pairs
        for helper_id in expired_pairs:
            del self.active_helping_pairs[helper_id]

    def _process_agent_decisions(self):
        """Process decision-making for all agents (parallel processing)."""
        logger.info(f"Agent decisions at t={self.current_sim_time:.1f}s")

        # Generate observations for all agents
        with self.perf_timer.measure("generate_observations"):
            observations = self._generate_observations()

        # Get available exits and zones for structured output
        exits = [
            {"name": name, "coords": coords}
            for name, coords in self.action_translator.exits.items()
        ]
        zones = list(self.action_translator.zones_polygons.keys()) or list(
            self.action_translator.zones.keys()
        )

        # Run agent processing in parallel using asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._process_agents_parallel(observations, exits, zones))
        finally:
            loop.close()

        self.last_decision_time = self.current_sim_time

    async def _process_agents_parallel(self, observations: dict, exits: list, zones: list):
        """Process all agents in parallel using async/await."""
        tasks = []
        agents_to_process = []  # Track which agents will make decisions this cycle

        for agent_id, agent in self.concordia_agents.items():
            # Skip agents who have already exited
            if agent_id in self.exited_agents:
                continue

            # Skip agents who are currently being helped (they're passively following the helper)
            if agent_id in self.agents_being_helped:
                logger.debug(f"{agent_id} is being helped, skipping decision-making")
                continue

            # Note: Helper agents are NOT skipped - they make free decisions
            # Their observations will indicate they're helping someone
            # We enforce speed synchronization separately in _update_helping_relationships()

            agents_to_process.append(agent_id)
            task = self._process_single_agent(agent_id, agent, observations, exits, zones)
            tasks.append(task)

        # Process all agents concurrently
        with self.perf_timer.measure("parallel_agent_processing"):
            await asyncio.gather(*tasks, return_exceptions=True)

        # Phase 4.1: Check if any helpers abandoned their helping commitment
        # Helpers can make free decisions, but if they change away from HELPING status, they abandoned
        abandoned_helps = []
        for helper_id, pair_info in list(self.active_helping_pairs.items()):
            # Check if helper's status is still HELPING
            if self.agent_status.get(helper_id) != "HELPING":
                helped_id = pair_info["helped"]
                logger.warning(
                    f"⚠️ {helper_id} abandoned helping {helped_id} "
                    f"(status changed from HELPING to {self.agent_status.get(helper_id)}) - "
                    f"releasing {helped_id} to independent movement"
                )
                abandoned_helps.append(helper_id)

        # Remove abandoned helping relationships
        for helper_id in abandoned_helps:
            pair_info = self.active_helping_pairs[helper_id]
            helped_id = pair_info["helped"]

            # Remove from tracking
            if helped_id in self.agents_being_helped:
                del self.agents_being_helped[helped_id]

            # Restore helped agent's original status (INJURED)
            self.agent_status[helped_id] = "INJURED"

            # Restore helped agent's speed
            if helped_id in self.agent_original_speeds:
                original_speed = self.agent_original_speeds[helped_id]
                self.jps_sim.set_agent_speed(helped_id, original_speed)

            # Remove the helping pair
            del self.active_helping_pairs[helper_id]

    async def _process_single_agent(
        self, agent_id: str, agent, observations: dict, exits: list, zones: list
    ):
        """Process a single agent's decision (async)."""
        try:
            # Get observation for this agent
            observation = observations.get(agent_id, "")

            # Check if observation changed since last decision
            observation_changed = (
                agent_id not in self.last_observations
                or self.last_observations[agent_id] != observation
            )

            if observation_changed:
                # Observation changed - provide to agent and call LLM
                with self.perf_timer.measure("agent_observe", is_parallel=True):
                    agent.observe(observation)

                # Call LLM with comprehensive single prompt
                action_spec = entity_lib.ActionSpec(
                    call_to_action=(
                        "Analyze the situation and decide your next action. Respond with ONLY valid JSON:\n\n"
                        "{{\n"
                        '  "situation": "Brief 1-2 sentence situation summary",\n'
                        '  "risk_level": "low|moderate|high",\n'
                        '  "risk_assessment": "Brief danger/threat assessment",\n'
                        '  "social_context": "What others are doing (if any)",\n'
                        '  "reasoning": "Why you chose this action (1-2 sentences)",\n'
                        '  "action_type": "wait|move",\n'
                        '  "target_type": "current_position|exit|zone",\n'
                        '  "exit_name": "exit name or null",\n'
                        '  "zone_name": "zone name or null",\n'
                        '  "wait_reason": "seeking_information|waiting_for_help|observing_others|assessing_situation or null",\n'
                        '  "speed": "slow_walk|normal_walk|brisk_walk|jog|run or null (m/s: 0.5|1.0|1.5|2.0|2.5)"\n'
                        "}}\n\n"
                        f"Available exits: {[e['name'] for e in exits]}\n"
                        f"Available zones: {zones}\n\n"
                        "Action rules:\n"
                        "- Use action_type='wait' and target_type='current_position' if staying put\n"
                        "  * Set wait_reason='seeking_information' if looking for directions/information\n"
                        "  * Set wait_reason='waiting_for_help' if injured and need assistance\n"
                        "  * Set wait_reason='observing_others' if watching to see what others do\n"
                        "  * Set wait_reason='assessing_situation' if evaluating risk/options\n"
                        "  * Set speed='slow_walk' (0.5 m/s) for seeking_information\n"
                        "- Use action_type='move' and target_type='exit' to evacuate (set exit_name or use 'nearest')\n"
                        "  * Set speed based on risk: 'normal_walk' (1.0 m/s) for low risk, 'brisk_walk' (1.5 m/s) for moderate, 'jog' (2.0 m/s) or 'run' (2.5 m/s) for high risk\n"
                        "- Use action_type='move' and target_type='zone' to move to a platform/area\n"
                        "- Use action_type='help' and target_type='current_position' to stop and assist an injured person nearby"
                    ),
                    output_type=entity_lib.OutputType.FREE,
                )

                # Run the LLM call in a thread pool to avoid blocking
                # (agent.act() is synchronous, so we wrap it in run_in_executor)
                with self.perf_timer.measure("agent_act_llm", is_parallel=True):
                    loop = asyncio.get_event_loop()
                    action = await loop.run_in_executor(None, agent.act, action_spec)

                self.last_observations[agent_id] = observation
                self.last_actions[agent_id] = action
                logger.info(f"{agent_id}: Observation changed, calling LLM")
            else:
                # Observation unchanged - reuse last action without calling observe/act
                action = self.last_actions.get(
                    agent_id, '{"action_type": "wait", "target_type": "current_position"}'
                )
                logger.info(f"{agent_id}: Observation unchanged, reusing last action")

            # Parse JSON response
            with self.perf_timer.measure("parse_json_response", is_parallel=True):
                reasoning = self._parse_json_response(action)

            # Translate action to JuPedSim command
            position = self._get_agent_position(agent_id)
            with self.perf_timer.measure("translate_action", is_parallel=True):
                translated = self.action_translator.translate(agent_id, action, position)

            # Phase 4.2: Detect route changes
            new_exit = self._extract_exit_name(translated)
            old_exit = self.agent_destinations.get(agent_id)
            route_changed = False

            if new_exit:
                if old_exit and old_exit != new_exit:
                    # Route change detected!
                    logger.info(f"🔄 {agent_id} changed route: {old_exit} → {new_exit}")
                    route_changed = True

                # Update destination tracking
                self.agent_destinations[agent_id] = new_exit

            # Store decision
            if agent_id not in self.agent_decisions:
                self.agent_decisions[agent_id] = {"decisions": []}

            decision_record = {
                "time": self.current_sim_time,
                "observation": observation,
                "prompt": action_spec.call_to_action if observation_changed else "cached",
                "action": action,
                "reasoning": reasoning,
                "translated": translated,
            }

            # Add route change metadata if it occurred
            if route_changed:
                decision_record["route_change"] = {
                    "from_exit": old_exit,
                    "to_exit": new_exit,
                    "reason": reasoning.get("reasoning", ""),
                }

            self.agent_decisions[agent_id]["decisions"].append(decision_record)

            # Apply to JuPedSim
            with self.perf_timer.measure("apply_to_jupedsim", is_parallel=True):
                self._apply_action_to_jupedsim(agent_id, translated)

            logger.info(f"{agent_id} action: {action[:100]}...")

        except Exception as e:
            logger.error(f"Error processing {agent_id}: {e}", exc_info=True)

    def _parse_json_response(self, response: str) -> dict[str, str]:
        """Parse JSON response from agent, extracting reasoning components."""
        try:
            # Strip agent name prefix (e.g., "Agent 0 {" -> "{")
            json_start = response.find("{")
            if json_start > 0:
                response = response[json_start:]

            data = json.loads(response)
            return {
                "situation": data.get("situation", ""),
                "risk_level": data.get("risk_level", ""),
                "risk_assessment": data.get("risk_assessment", ""),
                "social_context": data.get("social_context", ""),
                "reasoning": data.get("reasoning", ""),
            }
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse JSON response: {response[:200]}")
            return {
                "situation": "Parse error",
                "reasoning": response[:200],
            }

    def _extract_agent_reasoning(self, agent: entity_lib.Entity) -> dict[str, str]:
        """
        Extract reasoning from agent's components for display.

        Returns dict with component outputs.
        """
        reasoning = {}

        try:
            # Try to get component states
            components = agent.get_component_states()

            # Extract key reasoning components
            if "SelfPerception" in components:
                reasoning["self_perception"] = components["SelfPerception"]
            if "SituationPerception" in components:
                reasoning["situation"] = components["SituationPerception"]
            if "RiskPerception" in components:
                reasoning["risk"] = components["RiskPerception"]
            if "SocialInfluence" in components:
                reasoning["social"] = components["SocialInfluence"]
            if "PersonBySituation" in components:
                reasoning["strategy"] = components["PersonBySituation"]

        except Exception as e:
            logger.debug(f"Could not extract reasoning: {e}")

        return reasoning

    def _generate_observations(self) -> dict[str, str]:
        """Generate observations for all agents based on simulation state."""
        observations = {}

        for agent_id in self.concordia_agents.keys():
            # Skip exited agents
            if agent_id in self.exited_agents:
                continue

            try:
                # Get agent state from JuPedSim
                position = self._get_agent_position(agent_id)
                # Get observation radius from config
                help_config = self.test_scenarios.get("help_behavior", {})
                observation_radius = help_config.get("observation_radius", 20.0)
                nearby_agents = self._get_nearby_agents(agent_id, radius=observation_radius)

                # Enrich nearby_agents with target exit info (Phase 4.2)
                for agent_info in nearby_agents:
                    other_id = agent_info.get("id")
                    if other_id:
                        agent_info["target_exit"] = self.agent_destinations.get(other_id)

                recent_events = self._get_recent_events()

                # Generate observation (Phase 4.1: Include agent status)
                obs = self.observation_generator.generate_observation(
                    agent_id=agent_id,
                    position=position,
                    nearby_agents=nearby_agents,
                    events=recent_events,
                    sim_time=self.current_sim_time,
                    blocked_exits=self.blocked_exits,
                    agent_status=self.agent_status,
                )

                observations[agent_id] = obs

            except Exception as e:
                logger.error(f"Error generating observation for {agent_id}: {e}")
                observations[agent_id] = (
                    f"[Time: {self.current_sim_time:.1f}s] You are in the station."
                )

        return observations

    def _get_agent_position(self, agent_id: str) -> tuple[float, float]:
        """Get agent's current position from JuPedSim."""
        # TODO: Verify this works with real JuPedSim simulation
        try:
            return self.jps_sim.get_agent_position(agent_id)
        except Exception as e:
            logger.warning(f"Failed to get position for {agent_id}: {e}")
            return (0.0, 0.0)

    def _get_nearby_agents(self, agent_id: str, radius: float) -> list[dict[str, Any]]:
        """Get information about nearby agents."""
        # TODO: Verify this works with real JuPedSim simulation
        try:
            return self.jps_sim.get_nearby_agents(agent_id, radius)
        except Exception as e:
            logger.warning(f"Failed to get nearby agents for {agent_id}: {e}")
            return []

    def _get_recent_events(self) -> list[str]:
        """Get recent events relevant to agents (only events that have already occurred)."""
        # Return only events that have occurred (time <= current_sim_time)
        occurred_events = [
            e["message"] for e in self.event_history if e["time"] <= self.current_sim_time
        ]
        # Return last 3
        return occurred_events[-3:]

    def _extract_exit_name(self, translated_action: dict[str, Any]) -> str | None:
        """
        Extract the target exit name from a translated action.

        Returns:
            Exit name if action is moving to an exit, None otherwise
        """
        if translated_action["action_type"] != "move":
            return None

        target_coords = translated_action.get("target")
        if not target_coords:
            return None

        # Match coordinates to exit name
        for exit_name, exit_coords in self.station_layout["exits"].items():
            # Check if coordinates match (within 1m tolerance)
            if (
                abs(target_coords[0] - exit_coords[0]) < 1.0
                and abs(target_coords[1] - exit_coords[1]) < 1.0
            ):
                return exit_name

        return None

    def _convert_speed_to_ms(self, speed_str: str | None) -> float | None:
        """Convert speed string to m/s value sampled from normal distribution.

        Args:
            speed_str: Speed descriptor (slow_walk, normal_walk, brisk_walk, jog, run)

        Returns:
            Speed in m/s sampled from appropriate distribution, or None if not specified
        """
        if not speed_str:
            return None

        import numpy as np

        # Define mean and std for each speed category
        # Based on pedestrian dynamics research, with variation for different urgency levels
        speed_distributions = {
            "slow_walk": {"mean": 0.5, "std": 0.10, "min": 0.3, "max": 0.8},
            "normal_walk": {"mean": 1.0, "std": 0.20, "min": 0.6, "max": 1.4},
            "brisk_walk": {"mean": 1.5, "std": 0.25, "min": 1.0, "max": 2.0},
            "jog": {"mean": 2.0, "std": 0.30, "min": 1.5, "max": 2.5},
            "run": {"mean": 2.5, "std": 0.35, "min": 2.0, "max": 3.0},
        }

        dist = speed_distributions.get(speed_str.lower())
        if not dist:
            return None

        # Sample from normal distribution and clip to min/max
        speed = np.random.normal(dist["mean"], dist["std"])
        return max(dist["min"], min(dist["max"], speed))

    def _apply_action_to_jupedsim(self, agent_id: str, translated_action: dict[str, Any]):
        """Apply a translated action to the JuPedSim simulation."""
        action_type = translated_action["action_type"]
        target = translated_action["target"]

        logger.info(
            f"Agent {agent_id}: {action_type} to {target} "
            f"(confidence: {translated_action['confidence']:.2f}) - {translated_action['reasoning']}"
        )

        # TODO: With real JuPedSim, use proper waypoint/goal setting API
        # For now, using simple target setting
        try:
            # Phase 4.3: Apply dynamic speed if specified
            speed_str = translated_action.get("speed")
            if speed_str:
                speed_ms = self._convert_speed_to_ms(speed_str)
                if speed_ms:
                    self.jps_sim.set_agent_speed(agent_id, speed_ms)
                    logger.debug(f"Set {agent_id} speed to {speed_ms:.2f} m/s ({speed_str})")
            else:
                # No speed specified - for wait actions without speed, keep current speed
                # This prevents agents from getting stuck at speed=0
                pass

            if action_type == "help":
                # Phase 4.1: Agent is helping an injured person
                # Helper and injured agent will travel together at intermediate speed
                self.agent_status[agent_id] = "HELPING"

                # Get observation radius from config
                help_config = self.test_scenarios.get("help_behavior", {})
                observation_radius = help_config.get("observation_radius", 20.0)

                # Find nearest injured agent within observation radius
                position = self._get_agent_position(agent_id)
                nearby_agents = self._get_nearby_agents(agent_id, radius=observation_radius)

                injured_nearby = None
                for agent_info in nearby_agents:
                    other_id = agent_info.get("id")
                    if other_id and self.agent_status.get(other_id) == "INJURED":
                        # Check if this injured agent is already being helped
                        if other_id not in self.agents_being_helped:
                            injured_nearby = other_id
                            break

                if injured_nearby:
                    # Record help event
                    helper_config = next((c for c in self.agent_configs if c["id"] == agent_id), {})

                    # Get help configuration from test_scenarios
                    help_config = self.test_scenarios.get("help_behavior", {})
                    help_duration = help_config.get(
                        "help_duration", 15.0
                    )  # Default 15s if not in config

                    # Get injured agent's position
                    injured_position = self._get_agent_position(injured_nearby)

                    self.help_events.append(
                        {
                            "time": self.current_sim_time,
                            "helper": agent_id,
                            "helped": injured_nearby,
                            "helper_personality": helper_config.get("personality_type", "UNKNOWN"),
                            "location": position,
                            "duration": help_duration,
                        }
                    )

                    # Track active helping pair with two phases:
                    # 1. "approaching" - helper walks to injured agent (who stops)
                    # 2. "traveling" - both travel together at intermediate speed
                    self.active_helping_pairs[agent_id] = {
                        "helped": injured_nearby,
                        "start_time": self.current_sim_time,
                        "duration": help_duration,
                        "phase": "approaching",
                        "injured_position": injured_position,
                    }

                    # Update helped agent's status to WAITING (receiving assistance)
                    self.agents_being_helped[injured_nearby] = agent_id
                    self.agent_status[injured_nearby] = "WAITING"

                    # Store original speeds if not already stored
                    help_config = self.test_scenarios.get("help_behavior", {})
                    if agent_id not in self.agent_original_speeds:
                        normal_speed = help_config.get("normal_walking_speed", 1.34)
                        self.agent_original_speeds[agent_id] = normal_speed
                    if injured_nearby not in self.agent_original_speeds:
                        injured_speed = help_config.get("injured_walking_speed", 0.5)
                        self.agent_original_speeds[injured_nearby] = injured_speed

                    # Phase 1: Approaching
                    # - Injured agent stops (speed = 0) so helper can reach them
                    # - Helper walks toward injured agent's current position
                    self.jps_sim.set_agent_speed(injured_nearby, 0.0)
                    self.jps_sim.set_agent_target(agent_id, injured_position)

                    logger.info(
                        f"🤝 {agent_id} is approaching {injured_nearby} to help "
                        f"(distance: {((position[0]-injured_position[0])**2 + (position[1]-injured_position[1])**2)**0.5:.1f}m)"
                    )
                else:
                    # Check if there were injured agents but all already being helped
                    injured_already_helped = [
                        agent_info.get("id")
                        for agent_info in nearby_agents
                        if agent_info.get("id")
                        and self.agent_status.get(agent_info.get("id")) == "INJURED"
                        and agent_info.get("id") in self.agents_being_helped
                    ]
                    if injured_already_helped:
                        logger.info(
                            f"ℹ️ {agent_id} wanted to help but all nearby injured agents "
                            f"already being helped: {injured_already_helped}"
                        )
                    else:
                        logger.warning(f"{agent_id} wanted to help but no injured agents nearby")

            elif action_type == "move" and target:
                # Extract the NEW exit name from this action (if moving to an exit)
                new_exit_name = self._extract_exit_name(translated_action)

                if new_exit_name:
                    # Agent is moving to an exit - update destination tracking
                    self.agent_destinations[agent_id] = new_exit_name

                    # Check if agent is trying to switch to a blocked exit
                    if new_exit_name in self.blocked_exits:
                        logger.debug(
                            f"⚠️ {agent_id} tried to switch to blocked exit {new_exit_name} - "
                            f"keeping waypoint only"
                        )
                        # Only set waypoint, don't switch journey (would let them evacuate through blocked exit)
                        self.jps_sim.set_agent_target(agent_id, target)
                    else:
                        # Switch the agent's evacuation journey to this exit
                        if hasattr(self.jps_sim, "set_agent_evacuation_exit"):
                            self.jps_sim.set_agent_evacuation_exit(agent_id, new_exit_name)
                            logger.debug(f"Switched {agent_id} to journey for {new_exit_name}")
                        else:
                            self.jps_sim.set_agent_target(agent_id, target)
                else:
                    # Not moving to an exit, just a waypoint
                    self.jps_sim.set_agent_target(agent_id, target)
            elif action_type == "wait":
                # Phase 4.3: Track wait events with reasons
                current_position = self._get_agent_position(agent_id)
                wait_reason = translated_action.get("wait_reason", "unspecified")

                # Different behavior based on wait reason
                if wait_reason == "seeking_information":
                    # Seeking information: move slowly in a small random direction (looking around)
                    import math
                    import random

                    # Generate a random nearby point within 3-5 meters
                    distance = random.uniform(3.0, 5.0)
                    angle = random.uniform(0, 2 * math.pi)
                    target_x = current_position[0] + distance * math.cos(angle)
                    target_y = current_position[1] + distance * math.sin(angle)

                    self.jps_sim.set_agent_target(agent_id, (target_x, target_y))
                    logger.debug(
                        f"{agent_id} seeking information - moving slowly to nearby point "
                        f"({distance:.1f}m away)"
                    )
                else:
                    # All other wait types: stand still at current position
                    self.jps_sim.set_agent_target(agent_id, current_position)

                # Record wait event
                agent_config = next((c for c in self.agent_configs if c["id"] == agent_id), {})
                self.wait_events.append(
                    {
                        "time": self.current_sim_time,
                        "agent": agent_id,
                        "personality": agent_config.get("personality_type", "UNKNOWN"),
                        "wait_reason": wait_reason if wait_reason else "unspecified",
                        "location": current_position,
                    }
                )

        except Exception as e:
            logger.error(f"Failed to apply action for {agent_id}: {e}")

    def _check_and_trigger_events(self):
        """Check for and trigger simulation events."""
        # Phase 4.2: Check for exit blocking test scenario
        if hasattr(self, "test_block_exit_time") and self.test_block_exit_time:
            # Check if we've reached the blocking time (within one timestep)
            if self.current_sim_time >= self.test_block_exit_time and not hasattr(
                self, "_test_exit_blocked"
            ):
                # Trigger the blocking
                self.block_exit(self.test_block_exit_name)
                self._test_exit_blocked = True  # Flag to prevent repeated blocking

    def block_exit(self, exit_name: str):
        """
        Block an exit by placing a physical obstacle in JuPedSim.

        Agents will discover the blockage when they get close (visual range ~20m)
        or observe others turning back from it.

        Args:
            exit_name: Name of the exit to block
        """
        if exit_name not in self.station_layout["exits"]:
            logger.warning(f"Cannot block unknown exit: {exit_name}")
            return

        exit_pos = self.station_layout["exits"][exit_name]

        # Add to blocked exits set (for observations)
        self.blocked_exits.add(exit_name)

        # Place physical obstacle in JuPedSim (if supported)
        # Use a radius that blocks the entrance (typically 2-3m wide, so 3-4m radius covers it)
        # This makes the exit unreachable in pathfinding - agents cannot get close enough
        # to evacuate through it, and will naturally reroute when they observe the blockage
        try:
            if hasattr(self.jps_sim, "add_obstacle"):
                # Obstacle radius sized for typical entrance width (2-3m)
                obstacle_radius = 4.0
                self.jps_sim.add_obstacle(exit_pos, radius=obstacle_radius)
                logger.info(
                    f"🚧 Exit {exit_name} physically blocked at {exit_pos} "
                    f"(obstacle radius: {obstacle_radius}m)"
                )
            else:
                logger.info(
                    f"🚧 Exit {exit_name} marked as blocked (visual only, "
                    f"JuPedSim obstacle not supported)"
                )
        except Exception as e:
            logger.warning(f"Failed to add physical obstacle at {exit_name}: {e}")

        # NO announcement - agents discover naturally through observation
        # NOTE: We do NOT immediately reroute agents heading to this exit.
        # Agents should be allowed to travel to the blocked exit and discover it naturally,
        # then they will observe the blockage and choose alternative routes in their next decision.
        logger.info(f"Exit {exit_name} blocked - agents will discover naturally")

    def broadcast_event(self, event_message: str):
        """
        Broadcast an event to all agents.

        Args:
            event_message: The event message to broadcast
        """
        logger.info(f"Broadcasting event: {event_message}")

        # Store event
        self.event_history.append(
            {
                "time": self.current_sim_time,
                "message": event_message,
            }
        )

        # Notify all agents
        for agent in self.concordia_agents.values():
            agent.observe(f"[ANNOUNCEMENT] {event_message}")

    def _save_incremental(self):
        """Save current results incrementally for live viewing."""
        if not self.output_file:
            return

        # Get current agent positions from JuPedSim
        agent_positions = {}
        if hasattr(self.jps_sim, "get_all_agent_positions"):
            agent_positions = self.jps_sim.get_all_agent_positions()
        else:
            # Fallback: get individual positions
            for agent_id in self.concordia_agents.keys():
                agent_positions[agent_id] = self.jps_sim.get_agent_position(agent_id)

        results = {
            "agent_decisions": self.agent_decisions,
            "agent_positions": agent_positions,  # Current positions only
            "current_time": self.current_sim_time,
            "events": self.event_history,
            "blocked_exits": list(self.blocked_exits),  # Phase 4.2: For visualization
            "active_helping_pairs": {
                helper: {"helped": info["helped"], "phase": info.get("phase", "traveling")}
                for helper, info in self.active_helping_pairs.items()
            },  # Phase 4.1: For visualization
            "config": {
                "decision_interval": self.decision_interval,
                "max_steps": self.max_steps,
                "num_agents": len(self.concordia_agents),
            },
        }

        try:
            self.output_file.parent.mkdir(parents=True, exist_ok=True)
            tmp_file = self.output_file.with_suffix(self.output_file.suffix + ".tmp")
            with open(tmp_file, "w") as f:
                json.dump(results, f, indent=2)
            tmp_file.replace(self.output_file)
        except Exception as e:
            logger.warning(f"Failed to save incremental results: {e}")

    def save_results(self, output_path: Path):
        """Save simulation results to file."""
        # Get final agent positions
        agent_positions = {}
        if hasattr(self.jps_sim, "get_all_agent_positions"):
            agent_positions = self.jps_sim.get_all_agent_positions()
        else:
            for agent_id in self.concordia_agents.keys():
                agent_positions[agent_id] = self.jps_sim.get_agent_position(agent_id)

        # Phase 4.2: Extract route changes for analytics
        route_changes = []
        for agent_id, data in self.agent_decisions.items():
            for decision in data.get("decisions", []):
                if "route_change" in decision:
                    route_changes.append(
                        {
                            "agent": agent_id,
                            "time": decision["time"],
                            "from_exit": decision["route_change"]["from_exit"],
                            "to_exit": decision["route_change"]["to_exit"],
                            "reason": decision["route_change"]["reason"],
                        }
                    )

        results = {
            "agent_decisions": self.agent_decisions,
            "agent_positions": agent_positions,
            "final_time": self.current_sim_time,
            "events": self.event_history,
            "blocked_exits": list(self.blocked_exits),  # Phase 4.2: For visualization
            "route_changes": route_changes,  # Phase 4.2: Route change analytics
            "active_helping_pairs": {
                helper: {"helped": info["helped"], "phase": info.get("phase", "traveling")}
                for helper, info in self.active_helping_pairs.items()
            },  # Phase 4.1: For visualization
            "config": {
                "decision_interval": self.decision_interval,
                "max_steps": self.max_steps,
                "num_agents": len(self.concordia_agents),
            },
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results saved to {output_path}")

        # Save performance report
        perf_report_path = output_path.parent / "performance_report.txt"
        with open(perf_report_path, "w") as f:
            f.write(self.perf_timer.report())
        logger.info(f"Performance report saved to {perf_report_path}")

        # Save financial report
        financial_report_path = output_path.parent / "financial_report.txt"
        financial_report = self._generate_financial_report()
        with open(financial_report_path, "w") as f:
            f.write(financial_report)
        logger.info(f"Financial report saved to {financial_report_path}")

        # Save route change analytics (Phase 4.2)
        if route_changes:
            route_changes_path = output_path.parent / "route_changes.txt"
            with open(route_changes_path, "w") as f:
                f.write("=== ROUTE CHANGE ANALYTICS ===\n\n")
                f.write(f"Total route changes: {len(route_changes)}\n")
                f.write(
                    f"Agents who changed routes: {len({rc['agent'] for rc in route_changes})}\n\n"
                )
                f.write("Route Changes:\n")
                for rc in route_changes:
                    f.write(
                        f"  - {rc['agent']} at t={rc['time']:.1f}s: "
                        f"{rc['from_exit']} → {rc['to_exit']}\n"
                        f"    Reason: {rc['reason']}\n"
                    )
            logger.info(f"Route change analytics saved to {route_changes_path}")

        # Phase 4.1: Save help behavior analytics
        if self.help_events:
            help_analytics_path = output_path.parent / "help_behavior.txt"
            with open(help_analytics_path, "w") as f:
                f.write("=== HELP BEHAVIOR ANALYTICS ===\n\n")
                f.write(f"Total help events: {len(self.help_events)}\n")
                f.write(f"Agents who helped: {len({h['helper'] for h in self.help_events})}\n\n")

                # Breakdown by personality type
                personality_counts = {}
                for event in self.help_events:
                    personality = event.get("helper_personality", "UNKNOWN")
                    personality_counts[personality] = personality_counts.get(personality, 0) + 1

                f.write("Help events by personality:\n")
                for personality, count in sorted(
                    personality_counts.items(), key=lambda x: x[1], reverse=True
                ):
                    percentage = (count / len(self.help_events)) * 100 if self.help_events else 0
                    f.write(f"  {personality}: {count} helps ({percentage:.1f}%)\n")

                f.write("\nHelp Events:\n")
                for event in self.help_events:
                    f.write(
                        f"  - t={event['time']:.1f}s: {event['helper']} ({event['helper_personality']}) "
                        f"helped {event['helped']}\n"
                    )
            logger.info(f"Help behavior analytics saved to {help_analytics_path}")

        # Phase 4.3: Save waiting behavior analytics
        if self.wait_events:
            wait_analytics_path = output_path.parent / "wait_behavior.txt"
            with open(wait_analytics_path, "w") as f:
                f.write("=== WAITING BEHAVIOR ANALYTICS ===\n\n")
                f.write(f"Total wait events: {len(self.wait_events)}\n")
                f.write(f"Agents who waited: {len({w['agent'] for w in self.wait_events})}\n\n")

                # Breakdown by wait reason
                reason_counts = {}
                for event in self.wait_events:
                    reason = event.get("wait_reason", "unspecified")
                    reason_counts[reason] = reason_counts.get(reason, 0) + 1

                f.write("Wait events by reason:\n")
                for reason, count in sorted(
                    reason_counts.items(), key=lambda x: x[1], reverse=True
                ):
                    percentage = (count / len(self.wait_events)) * 100 if self.wait_events else 0
                    f.write(f"  {reason}: {count} waits ({percentage:.1f}%)\n")

                # Breakdown by personality type
                personality_counts = {}
                for event in self.wait_events:
                    personality = event.get("personality", "UNKNOWN")
                    personality_counts[personality] = personality_counts.get(personality, 0) + 1

                f.write("\nWait events by personality:\n")
                for personality, count in sorted(
                    personality_counts.items(), key=lambda x: x[1], reverse=True
                ):
                    percentage = (count / len(self.wait_events)) * 100 if self.wait_events else 0
                    f.write(f"  {personality}: {count} waits ({percentage:.1f}%)\n")

                f.write("\nRecent Wait Events (last 20):\n")
                for event in self.wait_events[-20:]:
                    reason = event.get("wait_reason", "unspecified")
                    f.write(
                        f"  - t={event['time']:.1f}s: {event['agent']} ({event['personality']}) "
                        f"waited ({reason})\n"
                    )
            logger.info(f"Wait behavior analytics saved to {wait_analytics_path}")

    def _generate_financial_report(self) -> str:
        """Generate a financial report from LLM usage statistics."""
        if not self.llm_provider or not hasattr(self.llm_provider, "get_usage_stats"):
            return "\n=== FINANCIAL REPORT ===\nLLM provider usage stats not available\n"

        try:
            stats = self.llm_provider.get_usage_stats()

            lines = []
            lines.append("\n=== FINANCIAL REPORT ===")
            lines.append("\nLLM Token Usage:")
            lines.append(f"  Prompt tokens:      {stats['prompt_tokens']:,}")
            lines.append(f"  Completion tokens:  {stats['completion_tokens']:,}")
            lines.append(f"  Total tokens:       {stats['total_tokens']:,}")
            lines.append(f"  Total requests:     {stats['total_requests']:,}")
            lines.append("\nCost Breakdown (£):")
            lines.append(f"  Input cost:         £{stats['input_cost_gbp']:.4f}")
            lines.append(f"  Output cost:        £{stats['output_cost_gbp']:.4f}")
            lines.append(f"  TOTAL COST:         £{stats['estimated_cost_gbp']:.4f}")
            lines.append("\nPer-Agent Averages:")
            num_agents = len(self.concordia_agents)
            if num_agents > 0:
                lines.append(f"  Tokens per agent:   {stats['total_tokens'] / num_agents:.0f}")
                lines.append(
                    f"  Cost per agent:     £{stats['estimated_cost_gbp'] / num_agents:.4f}"
                )
                lines.append(f"  Requests per agent: {stats['total_requests'] / num_agents:.1f}")
            lines.append("\n" + "=" * 40)

            return "\n".join(lines)
        except Exception as e:
            logger.warning(f"Failed to generate financial report: {e}")
            return f"\n=== FINANCIAL REPORT ===\nError generating report: {e}\n"
