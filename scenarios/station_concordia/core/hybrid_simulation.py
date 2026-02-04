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

        # Store LLM provider reference (for usage stats)
        # The language_model is an AzureLLMConcordia instance directly
        self.llm_provider = language_model if hasattr(language_model, "get_usage_stats") else None

        # Translation layer components
        self.action_translator = ActionTranslator(station_layout, language_model)
        self.observation_generator = ObservationGenerator(station_layout)

        # Build Concordia agents (each with their own memory bank)
        self.concordia_agents: dict[str, entity_lib.Entity] = {}
        self.agent_configs = agents_config

        self._build_agents()

        # Tracking
        self.last_decision_time = (
            -decision_interval
        )  # Start negative so first decision happens immediately
        self.current_sim_time = 0.0
        self.agent_decisions: dict[str, dict[str, Any]] = {}
        self.event_history: list[dict[str, Any]] = []
        self.last_observations: dict[str, str] = {}  # Cache observations for change detection
        self.last_actions: dict[str, str] = {}  # Cache actions to reuse

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

                    # Advance JuPedSim simulation
                    with self.perf_timer.measure("jupedsim_step"):
                        if not self._step_jupedsim():
                            logger.info("JuPedSim simulation complete")
                            break

                    self.current_sim_time = step * self.jps_sim.dt

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
        for agent_id, agent in self.concordia_agents.items():
            task = self._process_single_agent(agent_id, agent, observations, exits, zones)
            tasks.append(task)

        # Process all agents concurrently
        with self.perf_timer.measure("parallel_agent_processing"):
            await asyncio.gather(*tasks, return_exceptions=True)

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
                        '  "zone_name": "zone name or null"\n'
                        "}}\n\n"
                        f"Available exits: {[e['name'] for e in exits]}\n"
                        f"Available zones: {zones}\n\n"
                        "Action rules:\n"
                        "- Use action_type='wait' and target_type='current_position' if staying put\n"
                        "- Use action_type='move' and target_type='exit' to evacuate (set exit_name or use 'nearest')\n"
                        "- Use action_type='move' and target_type='zone' to move to a platform/area"
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

            # Store decision
            if agent_id not in self.agent_decisions:
                self.agent_decisions[agent_id] = {"decisions": []}

            self.agent_decisions[agent_id]["decisions"].append(
                {
                    "time": self.current_sim_time,
                    "observation": observation,
                    "prompt": action_spec.call_to_action if observation_changed else "cached",
                    "action": action,
                    "reasoning": reasoning,
                    "translated": translated,
                }
            )

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
            try:
                # Get agent state from JuPedSim
                position = self._get_agent_position(agent_id)
                nearby_agents = self._get_nearby_agents(agent_id, radius=10.0)
                recent_events = self._get_recent_events()

                # Generate observation
                obs = self.observation_generator.generate_observation(
                    agent_id=agent_id,
                    position=position,
                    nearby_agents=nearby_agents,
                    events=recent_events,
                    sim_time=self.current_sim_time,
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
            if action_type == "move" and target:
                self.jps_sim.set_agent_target(agent_id, target)
            elif action_type == "wait":
                current_position = self._get_agent_position(agent_id)
                self.jps_sim.set_agent_target(agent_id, current_position)
        except Exception as e:
            logger.error(f"Failed to apply action for {agent_id}: {e}")

    def _check_and_trigger_events(self):
        """Check for and trigger simulation events."""
        # TODO: Implement event triggering logic
        # Events could include:
        # - Announcements at specific times
        # - Alarms based on conditions
        # - Dynamic obstacles
        pass

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

        results = {
            "agent_decisions": self.agent_decisions,
            "agent_positions": agent_positions,
            "final_time": self.current_sim_time,
            "events": self.event_history,
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
