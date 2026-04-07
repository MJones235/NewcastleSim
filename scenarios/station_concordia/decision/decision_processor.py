"""
Decision Processor

Handles agent decision-making logic including:
- Parallel processing of agent decisions using asyncio
- LLM calls with comprehensive prompts for action selection
- JSON response parsing and reasoning extraction
- Action translation and decision recording
- Route change detection and tracking
- Intelligent prompt caching to reduce redundant LLM calls

This module coordinates the cognitive layer (Concordia) decision-making process.
"""

import asyncio
import json
import re
from typing import Any

from concordia.typing import entity as entity_lib

from scenarios.common.logger import get_logger
from scenarios.station_concordia.decision.action_utils import extract_exit_name
from scenarios.station_concordia.decision.prompt_cache import PromptCache

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
        agent_configs: list[dict] | None = None,
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
        self._agent_cfg: dict[str, dict] = {cfg["id"]: cfg for cfg in (agent_configs or [])}
        # Asyncio lock for shared state modifications during parallel processing
        self._state_lock = asyncio.Lock()

        # Initialize prompt cache for intelligent LLM call reduction
        self.prompt_cache = PromptCache(enable_detailed_logging=True)
        self.llm_calls_skipped = 0  # Statistics tracking
        self.llm_calls_made = 0

        # Read geometry-specific exit lists from station config rather than hardcoding.
        # These are defined under station.platform_up_exits / station.street_exits /
        # station.default_platform_zone in the YAML config.
        self._PLATFORM_UP_EXITS: dict[str, list[str]] = station_layout.get("platform_up_exits", {})
        self._STREET_EXIT_IDS: list[str] = station_layout.get("street_exits", [])
        self._DOWN_ACCESS_EXIT_IDS: list[str] = list(
            station_layout.get("down_access_exits", {}).keys()
        )
        self._default_platform_zone: str = station_layout.get("default_platform_zone", "")

        logger.debug("DecisionProcessor initialized for parallel async processing")

    def _build_zones_section(self, zones: list[str], agent_level: str | None = None) -> str:
        """Build the AVAILABLE ZONES block for the prompt.

        Uses zone_labels from station config to show human-readable names alongside
        the technical zone_name the LLM must copy verbatim into its JSON response.
        Returns an empty string when no zones are configured.

        Platform zones (platform_N) are suppressed for level-0 (concourse) agents
        because those zones are physically on level -1 and cannot be walked to
        directly — agents should instead choose a down escalator exit.
        """
        if not zones:
            return ""
        zone_labels: dict[str, str] = self.station_layout.get("zone_labels", {})
        lines = []
        for z in zones:
            # Hide platform-level zones from concourse agents
            if agent_level == "0" and re.match(r"^platform_", z):
                continue
            label = zone_labels.get(z, z.replace("_", " ").title())
            lines.append(f"  \u2022 {label} (zone_name='{z}')")
        if not lines:
            return ""
        bullets = "\n".join(lines)
        return "\nAvailable zones:\n" f"{bullets}\n\n"

    def _get_valid_exits_section(
        self, agent_id: str, observation: str, agent_level: str | None
    ) -> str:
        """Build a prompt section listing the valid exit names for this agent.

        Commuters already know their platform's UP escalators (or all street exits on
        level 0) from memory.  Novices only know exits they can currently see; if none
        are visible they are directed to follow another agent instead.
        """
        registry = self.action_translator.exit_registry
        valid_ids = set(registry.get_all_ids())
        cfg = self._agent_cfg.get(agent_id, {})
        profile = cfg.get("knowledge_profile", "novice")
        initial_zone = cfg.get("initial_zone", self._default_platform_zone)

        def _extract_visible_distance_ranks(text: str) -> dict[str, int]:
            """Map visible exit display names to distance rank (0=closest)."""
            ranks: dict[str, int] = {}
            distance_patterns = [
                (r"You can see\s+([^\.]+?)\s+very close to you\.", 0),
                (r"You can see\s+([^\.]+?)\s+nearby\.", 1),
                (r"You can see\s+([^\.]+?)\s+in the distance\.", 2),
            ]

            for pattern, rank in distance_patterns:
                for m in re.finditer(pattern, text):
                    for raw_name in m.group(1).split(","):
                        name = raw_name.strip()
                        if not name:
                            continue
                        if name not in ranks or rank < ranks[name]:
                            ranks[name] = rank
            return ranks

        visible_distance_rank = _extract_visible_distance_ranks(observation)

        def _sort_display_exits(display_names: list[str]) -> list[str]:
            """Sort exits by observed visibility, then observed proximity, then original order."""
            indexed = list(enumerate(display_names))

            def key_fn(item: tuple[int, str]) -> tuple[int, int, int, str]:
                idx, name = item
                is_visible = 0 if name in visible_distance_rank else 1
                dist_rank = visible_distance_rank.get(name, 99)
                return (is_visible, dist_rank, idx, name.lower())

            return [name for _, name in sorted(indexed, key=key_fn)]

        def _is_standard_exit(eid: str) -> bool:
            """Only include exits that are valid departure points for the agent's level.
            - Level 0 (concourse): street exits + down escalators only (no up escalators)
            - Level -1 (platform): up escalators only (no down escalators, no street exits)
            - Unknown level: exclude escalator_X_down (legacy safe default)
            """
            if agent_level == "0":
                # Concourse: bar up-escalators (they are arrival zones here, not exits)
                return not re.match(r"^escalator_[a-f]_up$", eid)
            elif agent_level == "-1":
                # Platform: bar down-escalators (they are arrival zones here, not exits)
                return not re.match(r"^escalator_[a-f]_down$", eid)
            # Unknown level: keep original safe default
            return not re.match(r"^escalator_[a-f]_down$", eid)

        if profile == "commuter":
            if agent_level == "-1":
                fallback = next(iter(self._PLATFORM_UP_EXITS), None)
                exit_ids = self._PLATFORM_UP_EXITS.get(initial_zone) or (
                    self._PLATFORM_UP_EXITS.get(fallback, []) if fallback else []
                )
                exits = [registry.get_display_name(eid) for eid in exit_ids if eid in valid_ids]
                exits = _sort_display_exits(exits)
                if exits:
                    bullets = "".join(f"\n  \u2022 {e}" for e in exits)
                    return "\nAllowed exits now:" f"{bullets}\n\n"
            else:
                # Level 0: include both down-access escalators (to reach platforms)
                # and street exits (for evacuation). Agent decides which to use.
                combined_ids = self._DOWN_ACCESS_EXIT_IDS + self._STREET_EXIT_IDS
                exits = [registry.get_display_name(eid) for eid in combined_ids if eid in valid_ids]
                exits = _sort_display_exits(exits)
                if exits:
                    bullets = "".join(f"\n  \u2022 {e}" for e in exits)
                    return "\nAllowed exits now:" f"{bullets}\n\n"
        else:
            # Novice: only exits whose display name appears in this observation
            all_display = [
                registry.get_display_name(eid)
                for eid in registry.get_all_ids()
                if _is_standard_exit(eid)
            ]

            # Restrict "visible" exits to explicit visual sentences only.
            # This avoids false positives from lines like "People heading toward exits".
            visual_fragments = re.findall(r"You can see\s+([^\.]+)\.", observation)
            visual_fragments += re.findall(r"Visible exits right now:\s*([^\.]+)\.", observation)
            visual_text = " ".join(visual_fragments)
            visible = [n for n in all_display if n in visual_text]
            if visible:
                bullets = "".join(f"\n  \u2022 {e}" for e in visible)
                return "\nVisible exits right now:" f"{bullets}\n\n"
            elif agent_level != "-1":
                # On the concourse: fall back to street exits + down escalators from memory
                fallback_ids = self._DOWN_ACCESS_EXIT_IDS + self._STREET_EXIT_IDS
                fallback = [
                    registry.get_display_name(eid) for eid in fallback_ids if eid in valid_ids
                ]
                if fallback:
                    bullets = "".join(f"\n  \u2022 {e}" for e in fallback)
                    return "\nAllowed exits now:" f"{bullets}\n\n"
            return (
                "\nVisible exits right now: none.\n"
                "Do NOT use target_type='exit'. Follow someone or wait.\n\n"
            )
        return ""

    def _get_following_constraint_text(self, observation: str) -> str:
        """Build a hard movement constraint block when followers are detected.

        Two cases:
        1. Circular following — both agents follow each other: force exit choice.
        2. One-way follower   — someone is following this agent: forbid targeting them.
        """
        constraints = []

        # Case 1: circular — "You are following Person X, and Person X is following YOU."
        circular = re.search(
            r"You are following (Person (\w+)), and Person \2 is following YOU",
            observation,
        )
        if circular:
            person_label = circular.group(1)  # e.g. "Person 46"
            agent_id_str = f"agent_{circular.group(2)}"  # e.g. "agent_46"
            constraints.append(
                f"\n\u26d4 CIRCULAR FOLLOWING — BREAK THE LOOP: You and {person_label} are "
                f"following each other. Neither of you is heading toward an exit. "
                f"You MUST do something different this turn. "
                f"Do NOT set target_agent='{agent_id_str}'. "
                f"Choose one of: (a) target_type='exit' if an exit is listed in VALID EXIT OPTIONS, "
                f"(b) target_type='agent' toward a DIFFERENT person who is heading toward an exit, "
                f"or (c) target_type='current_position' to stop and reorient."
            )
        else:
            # Case 2: one-way followers — "⚠\ufe0f Person X is trying to follow YOU."
            followers = re.findall(
                r"\u26a0\ufe0f (Person (\w+)) is trying to follow YOU", observation
            )
            # Also handle plural: "Person X, Person Y are trying to follow YOU"
            if not followers:
                plural = re.search(r"\u26a0\ufe0f (.+?) are trying to follow YOU", observation)
                if plural:
                    followers = [
                        (f"Person {m}", m) for m in re.findall(r"Person (\w+)", plural.group(1))
                    ]
            if followers:
                names = ", ".join(p for p, _ in followers)
                avoid = ", ".join(f"'agent_{n}'" for _, n in followers)
                constraints.append(
                    f"\n\u26a0\ufe0f FOLLOWER RULE: {names} is following YOU — you are their leader. "
                    f"Do NOT set target_agent={avoid}. "
                    f"You must lead: choose target_type='exit' using your own knowledge, "
                    f"or follow a DIFFERENT person who is actually heading toward an exit."
                )

        if constraints:
            return (
                "\n\u2550\u2550\u2550 MOVEMENT CONSTRAINTS (READ FIRST) \u2550\u2550\u2550"
                + "".join(constraints)
                + "\n\n"
            )
        return ""

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
        """Process a single agent's decision (async) with intelligent prompt caching."""
        # Get agent's level and filter exits accordingly
        agent_level = None
        if self.jps_sim and hasattr(self.jps_sim, "agent_levels"):
            agent_level = self.jps_sim.agent_levels.get(agent_id)

        try:
            # Get observation for this agent
            observation = observations.get(agent_id, "")

            # Build profile-aware valid exit options block
            valid_exits_text = self._get_valid_exits_section(agent_id, observation, agent_level)

            # Build follower constraint block (injected when circular/following detected)
            following_constraint_text = self._get_following_constraint_text(observation)

            # Only expose zone-targeting when it is meaningful this turn.
            # In practice this is when exits are not currently available/visible,
            # so the agent needs an intermediate area-level move.
            include_zones = "Visible exits right now: none." in valid_exits_text
            zones_section = self._build_zones_section(zones, agent_level) if include_zones else ""
            has_zones = bool(zones_section.strip())

            # The 'zone' target_type is only valid when AVAILABLE ZONES is non-empty.
            zone_target_type_line = (
                '  "target_type": ONE OF: "current_position", "exit", "agent", "zone",\n'
                if has_zones
                else '  "target_type": ONE OF: "current_position", "exit", "agent",\n'
            )
            zone_name_example = (
                '  "zone_name": null (or zone name EXACTLY as shown in AVAILABLE ZONES),\n'
                if has_zones
                else '  "zone_name": null,\n'
            )
            zone_constraint_line = (
                "- If target_type='zone', zone_name is required.\n" if has_zones else ""
            )

            # Build the action spec with prompt text
            action_spec = entity_lib.ActionSpec(
                call_to_action=(
                    "Decision task: choose your next action now. Respond with ONLY this JSON:\n"
                    "{{\n"
                    '  "reasoning": "1-2 sentences grounded in goal + current observations",\n'
                    '  "action_type": "wait" or "move",\n'
                    f"{zone_target_type_line}"
                    '  "target_agent": null (or agent_id like "agent_5"),\n'
                    '  "exit_name": null (or the exit name EXACTLY as listed in visible/valid exits),\n'
                    f"{zone_name_example}"
                    '  "wait_reason": null (or reason if waiting),\n'
                    '  "speed": null (or "slow_walk", "normal_walk", "brisk_walk", "jog", "run"),\n'
                    '  "message": null (or your spoken words),\n'
                    '  "message_type": null (or "directed", "shout"),\n'
                    "}}\n\n"
                    "Hard constraints:\n"
                    "- If action_type='wait', target_type MUST be 'current_position'.\n"
                    "- If target_type='exit', exit_name is required and must match an allowed exit name exactly.\n"
                    "- If target_type='agent', target_agent is required.\n"
                    f"{zone_constraint_line}"
                    "- Keep spoken message short and natural; no narration.\n\n"
                    "Decision policy:\n"
                    "1. If injured/helping someone, stay coordinated with that person.\n"
                    "2. If evacuating now, choose target_type='exit'.\n"
                    "3. If following someone, choose target_type='agent'.\n"
                    "4. Otherwise wait or reorient.\n\n"
                    f"{zones_section}"
                    f"{following_constraint_text}"
                    f"{valid_exits_text}"
                ),
                output_type=entity_lib.OutputType.FREE,
            )

            prompt_text = action_spec.call_to_action

            # ===== INTELLIGENT PROMPT CACHING =====
            # Check if we should call LLM or reuse cached decision
            try:
                received_messages = self.message_system.get_received_messages(agent_id)
                messages_list = (
                    [
                        msg.get("message", str(msg)) if isinstance(msg, dict) else msg
                        for msg in received_messages
                    ]
                    if received_messages
                    else None
                )
            except Exception as e:
                logger.debug(f"{agent_id}: Could not get messages: {e}")
                messages_list = None

            # Check cache to decide whether to call LLM
            should_call_llm, cached_decision = self.prompt_cache.should_call_llm(
                agent_id=agent_id,
                observation=observation,
                action_spec_text=prompt_text,
                received_messages=messages_list,
            )

            llm_was_called = False  # Track whether LLM was called for decision record

            if not should_call_llm and cached_decision:
                # Reuse cached decision - no new information to act on
                action = cached_decision
                logger.info(
                    f"{agent_id}: ✓ Prompt unchanged, reusing cached decision (saved LLM call)"
                )
                async with self._state_lock:
                    self.llm_calls_skipped += 1
            else:
                # Call LLM - significant new information detected
                with self.perf_timer.measure("agent_observe", is_parallel=True):
                    agent.observe(observation)

                # Run the LLM call in a separate thread to avoid blocking
                with self.perf_timer.measure("agent_act_llm", is_parallel=True):
                    action = await asyncio.to_thread(agent.act, action_spec)

                # Cache the decision for future reuse
                self.prompt_cache.cache_decision(agent_id, action)
                llm_was_called = True

                if should_call_llm and messages_list:
                    logger.info(
                        f"{agent_id}: ✓ Calling LLM (received {len(messages_list)} message(s))"
                    )
                elif should_call_llm:
                    logger.info(f"{agent_id}: ✓ Calling LLM (observation changed significantly)")
                else:
                    logger.info(f"{agent_id}: ✓ Calling LLM (first decision)")

                async with self._state_lock:
                    self.llm_calls_made += 1

                # Update observation/action cache
                async with self._state_lock:
                    self.last_observations[agent_id] = observation
                    self.last_actions[agent_id] = action

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

            # Inject the LLM's own reasoning text into the translated dict so that
            # the previous-decision memory shown to the agent next turn uses the
            # agent's actual words rather than the translator's description.
            llm_reasoning = reasoning.get("reasoning", "") if isinstance(reasoning, dict) else ""
            if llm_reasoning:
                translated["reasoning"] = llm_reasoning

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
                    "prompt": action_spec.call_to_action if llm_was_called else "cached",
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

    def on_agent_exit(self, agent_id: str) -> None:
        """
        Clean up cache when agent exits simulation.

        Args:
            agent_id: ID of exiting agent
        """
        self.prompt_cache.clear_agent(agent_id)
        logger.debug(f"{agent_id}: Cache cleared on exit")

    def get_cache_statistics(self) -> dict[str, Any]:
        """
        Get statistics about LLM call optimization.

        Returns:
            Dictionary with cache and call statistics
        """
        total_calls = self.llm_calls_made + self.llm_calls_skipped
        skip_rate = (self.llm_calls_skipped / total_calls * 100) if total_calls > 0 else 0

        return {
            "llm_calls_made": self.llm_calls_made,
            "llm_calls_skipped": self.llm_calls_skipped,
            "total_decision_cycles": total_calls,
            "skip_rate_percent": skip_rate,
            "cache_stats": self.prompt_cache.get_statistics(),
        }

    def log_cache_summary(self) -> None:
        """Log summary of cache statistics to console and logger."""
        stats = self.get_cache_statistics()
        logger.info("=" * 70)
        logger.info("LLM CALL OPTIMIZATION SUMMARY (Prompt Caching)")
        logger.info("=" * 70)
        logger.info(f"  Total decision cycles: {stats['total_decision_cycles']}")
        logger.info(f"  LLM calls made:        {stats['llm_calls_made']}")
        logger.info(f"  LLM calls skipped:     {stats['llm_calls_skipped']} ✓")
        logger.info(f"  Skip rate:             {stats['skip_rate_percent']:.1f}%")
        logger.info(f"  Cache size:            {stats['cache_stats']['cached_agents']} agents")
        logger.info("=" * 70)
