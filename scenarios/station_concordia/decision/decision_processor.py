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
        helping_system_manager,
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
            helping_system_manager: HelpingSystemManager for checking abandoned helps
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
        self.helping_system_manager = helping_system_manager

    def process_all_agents(self, observations: dict[str, str], current_sim_time: float) -> float:
        """
        Process decision-making for all agents (parallel processing).

        Args:
            observations: Dict of agent_id -> observation string
            current_sim_time: Current simulation time in seconds

        Returns:
            Current simulation time (for updating last_decision_time)
        """
        logger.info(f"Agent decisions at t={current_sim_time:.1f}s")

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
            loop.run_until_complete(
                self._process_agents_parallel(observations, exits, zones, current_sim_time)
            )
        finally:
            loop.close()

        return current_sim_time

    async def _process_agents_parallel(
        self, observations: dict, exits: list, zones: list, current_sim_time: float
    ):
        """Process all agents in parallel using async/await."""
        tasks = []
        agents_to_process = []  # Track which agents will make decisions this cycle

        for agent_id, agent in self.concordia_agents.items():
            # Skip agents who have already exited
            if agent_id in self.exited_agents:
                continue

            # Agents being helped still make decisions (can receive/send messages)
            # but their movement is controlled by the helper via speed synchronization
            # NOTE: We don't skip them here anymore so they can respond to messages

            # Note: Helper agents are NOT skipped - they make free decisions
            # Their observations will indicate they're helping someone
            # We enforce speed synchronization separately in helping_system_manager

            agents_to_process.append(agent_id)
            task = self._process_single_agent(
                agent_id, agent, observations, exits, zones, current_sim_time
            )
            tasks.append(task)

        # Process all agents concurrently
        with self.perf_timer.measure("parallel_agent_processing"):
            await asyncio.gather(*tasks, return_exceptions=True)

        # Check if any helpers abandoned their helping commitment
        # Helpers can make free decisions, but if they change away from HELPING status, they abandoned
        self.helping_system_manager.check_abandoned_helps()

    async def _process_single_agent(
        self,
        agent_id: str,
        agent,
        observations: dict,
        exits: list,
        zones: list,
        current_sim_time: float,
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
                        '  "speed": "slow_walk|normal_walk|brisk_walk|jog|run or null (m/s: 0.5|1.0|1.5|2.0|2.5)",\n'
                        '  "message": "Short casual message or null",\n'
                        '  "message_type": "directed|shout|quiet or null",\n'
                        '  "target_agent": "agent_id, nearest_injured, or null"\n'
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
                        "- Use action_type='help' and target_type='current_position' to stop and assist an injured person nearby\n\n"
                        "Communication (SPOKEN, not text - keep it brief and natural):\n"
                        "- These are SPOKEN words, not written messages - be conversational\n"
                        "- If someone speaks to you (marked 'to you'), respond naturally\n"
                        "  * Use target_agent to reply (e.g., 'Person 15' → target_agent='agent_15')\n"
                        "- Look at your conversation history - PROGRESS the dialogue, don't repeat:\n"
                        "  * If you already asked 'You ok?' and they answered, move on or stay quiet\n"
                        "  * If you're coordinating help, confirm and act - don't keep discussing it\n"
                        "- Keep it SHORT: 'You ok?' not 'Are you okay and do you need assistance?'\n"
                        "- DO NOT narrate actions: Bad: 'I'm heading to exit' / Good: 'Come on, let's go'\n"
                        "- Message types:\n"
                        "  * directed: to specific person (set target_agent='agent_5')\n"
                        "  * shout: urgent warning to everyone nearby\n"
                        "  * quiet: brief comment to very close people (<3m)"
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
            position = self.state_queries.get_agent_position(agent_id)
            with self.perf_timer.measure("translate_action", is_parallel=True):
                translated = self.action_translator.translate(agent_id, action, position)

            # Extract and deliver any message
            self.message_system.extract_and_deliver_message(
                sender_id=agent_id,
                action=action,
                sender_position=position,
                current_sim_time=current_sim_time,
                state_queries=self.state_queries,
                agent_status=self.action_executor.agent_status,
                exited_agents=self.exited_agents,
            )

            # Detect route changes
            new_exit = extract_exit_name(translated, self.station_layout)
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
                "time": current_sim_time,
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
