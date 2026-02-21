"""
Decision Processor

Handles agent decision-making logic including:
- Parallel processing of agent decisions using asyncio
- LLM calls with comprehensive prompts for action selection
- JSON response parsing and reasoning extraction
- Action translation and decision recording
- Route change detection and tracking

This module coordinates the cognitive layer (Concordia) decision-making process.
"""

import asyncio
import json
from typing import Any

from concordia.typing import entity as entity_lib

from scenarios.common.logger import get_logger
from scenarios.station_concordia.decision.action_utils import extract_exit_name

logger = get_logger(__name__)


class DecisionProcessor:
    """Processes agent decision-making with parallel LLM calls."""

    def __init__(
        self,
        concordia_agents: dict[str, entity_lib.Entity],
        exited_agents: set[str],
        action_translator,
        action_executor,
        message_system,
        state_queries,
        station_layout: dict[str, Any],
        agent_decisions: dict[str, dict[str, Any]],
        agent_destinations: dict[str, str],
        last_observations: dict[str, str],
        last_actions: dict[str, str],
        perf_timer,
        jps_sim=None,
    ):
        """
        Initialize decision processor.

        Args:
            concordia_agents: Dict of agent_id -> Concordia entity
            exited_agents: Set of agent IDs who have exited
            action_translator: ActionTranslator for NL to JuPedSim commands
            action_executor: ActionExecutor for applying actions
            message_system: MessageSystem for agent communication
            state_queries: Simulation state query interface
            station_layout: Station geometry and exit information
            agent_decisions: Dict tracking all agent decisions
            agent_destinations: Dict of agent_id -> current exit name
            last_observations: Cache of last observations for change detection
            last_actions: Cache of last actions to reuse
            perf_timer: Performance monitoring timer
            jps_sim: JuPedSim simulation instance (for multi-level support)
        """
        self.concordia_agents = concordia_agents
        self.exited_agents = exited_agents
        self.action_translator = action_translator
        self.action_executor = action_executor
        self.message_system = message_system
        self.state_queries = state_queries
        self.station_layout = station_layout
        self.agent_decisions = agent_decisions
        self.agent_destinations = agent_destinations
        self.last_observations = last_observations
        self.last_actions = last_actions
        self.perf_timer = perf_timer
        self.jps_sim = jps_sim
        # Asyncio lock for shared state modifications during parallel processing
        self._state_lock = asyncio.Lock()

        logger.debug("DecisionProcessor initialized for parallel async processing")

    def process_all_agents(
        self,
        observations: dict[str, str],
        current_sim_time: float,
        agent_ids: list[str] | None = None,
    ) -> float:
        """
        Process decision-making for all agents (parallel processing).

        Args:
            observations: Dict of agent_id -> observation string
            current_sim_time: Current simulation time in seconds
            agent_ids: Optional subset of agent IDs to process

        Returns:
            Current simulation time (for updating last_decision_time)
        """
        if agent_ids is None:
            logger.info(f"Agent decisions at t={current_sim_time:.1f}s")
        else:
            logger.info(
                f"Targeted agent decisions at t={current_sim_time:.1f}s for {len(agent_ids)} agents"
            )

        # Get available zones (same for all levels)
        zones = list(self.action_translator.zones_polygons.keys()) or list(
            self.action_translator.zones.keys()
        )

        # Run agent processing in parallel using asyncio
        asyncio.run(self._process_agents_parallel(observations, zones, current_sim_time, agent_ids))

        return current_sim_time

    async def _process_agents_parallel(
        self,
        observations: dict,
        zones: list,
        current_sim_time: float,
        agent_ids: list[str] | None = None,
    ):
        """Process all agents in parallel using async/await."""
        # Filter and create tasks using list comprehension for efficiency
        candidate_agents = (
            agent_ids if agent_ids is not None else list(self.concordia_agents.keys())
        )
        agents_to_process = [
            agent_id
            for agent_id in candidate_agents
            if agent_id in self.concordia_agents and agent_id not in self.exited_agents
        ]

        tasks = [
            self._process_single_agent(
                agent_id,
                self.concordia_agents[agent_id],
                observations,
                zones,
                current_sim_time,
            )
            for agent_id in agents_to_process
        ]

        # Process all agents concurrently
        with self.perf_timer.measure("parallel_agent_processing"):
            await asyncio.gather(*tasks, return_exceptions=True)

        # No need to check for abandonment - it's handled naturally in action execution
        # Agents who choose move/wait over help automatically end relationships

    async def _process_single_agent(
        self,
        agent_id: str,
        agent,
        observations: dict,
        zones: list,
        current_sim_time: float,
    ):
        """Process a single agent's decision (async)."""
        # Get agent's level and filter exits accordingly
        agent_level = None
        if self.jps_sim and hasattr(self.jps_sim, "agent_levels"):
            agent_level = self.jps_sim.agent_levels.get(agent_id)

        # Get level-specific exits
        if agent_level and self.jps_sim and hasattr(self.jps_sim, "simulations"):
            # Multi-level: get exits from agent's current level
            level_sim = self.jps_sim.simulations.get(agent_level)
            if level_sim:
                exits = [
                    {"name": name, "coords": coords}
                    for name, coords in level_sim.exit_manager.evacuation_exits.items()
                ]
            else:
                exits = []
        else:
            # Single-level: use all exits
            exits = [
                {"name": name, "coords": coords}
                for name, coords in self.action_translator.exits.items()
            ]
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

                # Call LLM with clear, concise decision prompt
                action_spec = entity_lib.ActionSpec(
                    call_to_action=(
                        "DECIDE YOUR NEXT ACTION. Respond with ONLY this JSON:\n"
                        "{{\n"
                        '  "reasoning": "Why this action (1-2 sentences)",\n'
                        '  "action_type": "wait" or "move",\n'
                        '  "target_type": ONE OF: "current_position", "exit", "agent", "zone",\n'
                        '  "target_agent": null (or agent_id like "agent_5"),\n'
                        '  "exit_name": null (or exit name like "jps.entrance_1"),\n'
                        '  "zone_name": null (or zone name like "jps.platform_3"),\n'
                        '  "wait_reason": null (or reason if waiting),\n'
                        '  "speed": null (or "slow_walk", "normal_walk", "brisk_walk", "jog", "run"),\n'
                        '  "message": null (or your spoken words),\n'
                        '  "message_type": null (or "directed", "shout", "quiet"),\n'
                        "}}\n\n"
                        "VALID target_type VALUES (ONLY these four):\n"
                        "  'current_position' → Wait at your current location (action_type='wait')\n"
                        "  'exit' → Move to an exit for evacuation (requires exit_name)\n"
                        "  'agent' → Move toward another agent to follow/help (requires target_agent)\n"
                        "  'zone' → Move to a specific area or platform (requires zone_name)\n\n"
                        "═══ YOUR OPTIONS ═══\n"
                        "WAITING (action_type='wait', target_type='current_position'):\n"
                        "  waiting_with_injured: Someone nearby is injured, stay with them\n"
                        "  waiting_for_help: YOU are injured, need someone to help\n"
                        "  seeking_information: Looking around for directions/info\n"
                        "  observing_others: Watching what others do before deciding\n"
                        "  assessing_situation: Thinking through options\n\n"
                        "MOVING (action_type='move', choose ONE target_type):\n"
                        "  target_type='agent': Move toward another agent to follow them or help them\n"
                        "    Set target_agent='agent_5' (the person's ID)\n"
                        "  target_type='exit': Evacuate through a specific exit\n"
                        "    Set exit_name='jps.entrance_1'\n"
                        "  target_type='zone': Move to a specific platform or area\n"
                        "    Set zone_name='jps.platform_3'\n\n"
                        "═══ COMMUNICATION ═══\n"
                        "message: Short phrase (keep it REAL, not narrated)\n"
                        "message_type: 'directed' (to specific person), 'shout' (urgent), 'quiet' (<3m only)\n"
                        "target_agent: If directed/quiet message, who are you talking to?\n\n"
                        "═══ DECISION TREE ═══\n"
                        "1. Am I injured or helping someone? YES→ stay together or coordinate (agent)\n"
                        "2. Do I want to approach or stay with someone? YES→ target_type='agent'\n"
                        "3. Should I evacuate now? YES→ target_type='exit'\n"
                        "4. Else→ wait or move to a zone\n\n"
                        f"Available exits: {[e['name'] for e in exits]}\n"
                        f"Available zones: {zones}\n"
                    ),
                    output_type=entity_lib.OutputType.FREE,
                )

                # Run the LLM call in a separate thread to avoid blocking
                # (agent.act() is synchronous, so we use asyncio.to_thread)
                with self.perf_timer.measure("agent_act_llm", is_parallel=True):
                    action = await asyncio.to_thread(agent.act, action_spec)

                # Async-safe update of observation/action cache
                async with self._state_lock:
                    self.last_observations[agent_id] = observation
                    self.last_actions[agent_id] = action
                logger.info(f"{agent_id}: Observation changed, calling LLM")
            else:
                # Observation unchanged - reuse last action without calling observe/act
                # Async-safe read from action cache
                async with self._state_lock:
                    action = self.last_actions.get(
                        agent_id, '{"action_type": "wait", "target_type": "current_position"}'
                    )
                logger.info(f"{agent_id}: Observation unchanged, reusing last action")

            # Parse JSON response
            with self.perf_timer.measure("parse_json_response", is_parallel=True):
                reasoning = self._parse_json_response(action)

            # Translate action to JuPedSim command
            position = self.state_queries.get_agent_position(agent_id)
            if position is None:
                # Agent has likely exited - skip action execution
                logger.debug(f"{agent_id}: No position found, likely exited")
                return

            with self.perf_timer.measure("translate_action", is_parallel=True):
                translated = self.action_translator.translate(agent_id, action, position)

            # Extract and deliver any message
            with self.perf_timer.measure("message_delivery", is_parallel=True):
                self.message_system.extract_and_deliver_message(
                    sender_id=agent_id,
                    action=action,
                    sender_position=position,
                    current_sim_time=current_sim_time,
                    state_queries=self.state_queries,
                    exited_agents=self.exited_agents,
                )

            # Detect route changes and store decision
            with self.perf_timer.measure("decision_storage", is_parallel=True):
                new_exit = extract_exit_name(translated, self.station_layout)

                # Async-safe read from agent_destinations
                async with self._state_lock:
                    old_exit = self.agent_destinations.get(agent_id)

                route_changed = False

                # Prepare decision record before acquiring lock
                decision_record = {
                    "time": current_sim_time,
                    "observation": observation,
                    "prompt": action_spec.call_to_action if observation_changed else "cached",
                    "action": action,
                    "reasoning": reasoning,
                    "translated": translated,
                }

                # Async-safe update of shared state
                async with self._state_lock:
                    if new_exit:
                        if old_exit and old_exit != new_exit:
                            # Route change detected!
                            logger.info(f"🔄 {agent_id} changed route: {old_exit} → {new_exit}")
                            route_changed = True

                        # Update destination tracking
                        self.agent_destinations[agent_id] = new_exit

                    # Add route change metadata if it occurred
                    if route_changed:
                        decision_record["route_change"] = {
                            "from_exit": old_exit,
                            "to_exit": new_exit,
                            "reason": reasoning.get("reasoning", ""),
                        }

                    # Store decision
                    if agent_id not in self.agent_decisions:
                        self.agent_decisions[agent_id] = {"decisions": []}

                    self.agent_decisions[agent_id]["decisions"].append(decision_record)

            # Apply to JuPedSim
            with self.perf_timer.measure("apply_to_jupedsim", is_parallel=True):
                self.action_executor.execute_action(agent_id, translated, current_sim_time)

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
