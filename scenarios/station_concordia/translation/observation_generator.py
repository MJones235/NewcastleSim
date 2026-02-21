"""
Generates natural language observations from JuPedSim simulation state.

Converts geometric and simulation data into observations that Concordia
agents can reason about.
"""

from typing import Any

from scenarios.common.logger import get_logger
from scenarios.station_concordia.translation.crowd_analyzer import CrowdAnalyzer
from scenarios.station_concordia.translation.observation_formatter import ObservationFormatter
from scenarios.station_concordia.translation.spatial_analyzer import SpatialAnalyzer

logger = get_logger(__name__)


class ObservationGenerator:
    """
    Generates natural language observations from JuPedSim simulation state.

    Converts geometric and simulation data into observations that Concordia
    agents can reason about.
    """

    def __init__(self, station_layout: dict[str, Any], jps_sim=None):
        """
        Initialize the observation generator.

        Args:
            station_layout: Station geometry and zone information (level 0 for multi-level)
            jps_sim: JuPedSim simulation instance (for multi-level exit access)
        """
        self.station_layout = station_layout
        self.exits = station_layout.get("exits", {})
        self._agents_with_geometry_intro: set[str] = set()
        self.jps_sim = jps_sim

        # Initialize analyzers
        self.spatial_analyzer = SpatialAnalyzer(station_layout)
        self.crowd_analyzer = CrowdAnalyzer(self.exits)

    def generate_observation(
        self,
        agent_id: str,
        position: tuple[float, float],
        nearby_agents: list[dict[str, Any]],
        events: list[str],
        sim_time: float,
        blocked_exits: set[str] | None = None,
        agent_injured: set[str] | None = None,
        agent_action: dict[str, str] | None = None,
        agent_level: str | None = None,
        agent_last_decision: dict[str, dict] | None = None,
        state_queries=None,
        received_messages: list[dict[str, Any]] | None = None,
        conversation_history: dict[str, list[dict]] | None = None,
    ) -> str:
        """
        Generate a natural language observation for an agent.

        Args:
            agent_id: ID of the observing agent
            position: Agent's current (x, y) position
            nearby_agents: List of nearby agents with their info
            events: Recent events (announcements, alarms, etc.)
            sim_time: Current simulation time
            blocked_exits: Set of blocked exit names (for visual observation)
            agent_injured: Set of injured agent IDs (physical capability dimension)
            agent_action: Dict of agent_id -> action ("moving"|"waiting")
            agent_level: Current level ID for multi-level simulations (e.g., "0", "-1")
            agent_last_decision: Dict of agent_id -> last decision (for memory)
            state_queries: SimulationStateQueries for position lookups (optional)
            received_messages: List of messages received from nearby agents
            conversation_history: Dict mapping other_agent_id to conversation history

        Returns:
            Natural language observation string
        """
        observations = []
        if blocked_exits is None:
            blocked_exits = set()
        if agent_injured is None:
            agent_injured = set()
        if agent_action is None:
            agent_action = {}
        if agent_last_decision is None:
            agent_last_decision = {}
        if received_messages is None:
            received_messages = []
        if conversation_history is None:
            conversation_history = {}

        # Note: Station layout is now in agent formative memory, not observations
        # This keeps observations stable when nothing changes
        # Time is also omitted as it's not meaningful for agent decisions

        # ---BEGIN: AGENT'S MEMORY OF PREVIOUS DECISION (Concordia formative memory)---
        # This allows agents to remember and reason about their commitments
        if agent_id in agent_last_decision:
            last_decision_lines = ObservationFormatter.format_last_decision(
                agent_id, agent_last_decision[agent_id]
            )
            observations.extend(last_decision_lines)
        # ---END: AGENT'S MEMORY OF PREVIOUS DECISION---

        # EARLY CHECK: Detect circular following BEFORE other observations
        # This makes it the most salient new information
        followers = [a for a in nearby_agents if a.get("is_following_me", False)]

        # CIRCULAR FOLLOWING DETECTION: Describe observable pattern without prescribing behavior
        if agent_id in agent_last_decision:
            last_decision = agent_last_decision[agent_id]
            if (
                last_decision.get("target_type") == "agent"
                and last_decision.get("target_agent") is not None
            ):
                # We're following someone - check if they're also following us
                our_target = last_decision.get("target_agent")
                circular_detected = False
                for follower in followers:
                    if follower.get("id") == our_target:
                        # Describe the observable situation without prescribing action
                        target_person = our_target.replace("agent_", "Person ")
                        observations.append(
                            f"You are following {target_person}, and {target_person} is following YOU. "
                            f"Neither of you is heading toward an exit."
                        )
                        circular_detected = True
                        break

                if not circular_detected and followers:
                    # Not circular but still being followed - show normal follower alerts
                    follower_ids = [a.get("id").replace("agent_", "Person ") for a in followers]
                    if len(followers) == 1:
                        observations.append(f"⚠️ {follower_ids[0]} is trying to follow YOU.")
                    else:
                        follower_list = ", ".join(follower_ids)
                        observations.append(f"⚠️ {follower_list} are trying to follow YOU.")
            else:
                # Not following anyone - show normal follower alerts only
                if followers:
                    follower_ids = [a.get("id").replace("agent_", "Person ") for a in followers]
                    if len(followers) == 1:
                        observations.append(f"⚠️ {follower_ids[0]} is trying to follow YOU.")
                    else:
                        follower_list = ", ".join(follower_ids)
                        observations.append(f"⚠️ {follower_list} are trying to follow YOU.")
        else:
            # No previous decision recorded - show normal follower alerts
            if followers:
                follower_ids = [a.get("id").replace("agent_", "Person ") for a in followers]
                if len(followers) == 1:
                    observations.append(f"⚠️ {follower_ids[0]} is trying to follow YOU.")
                else:
                    follower_list = ", ".join(follower_ids)
                    observations.append(f"⚠️ {follower_list} are trying to follow YOU.")

        # First-time geometry introduction (level-specific)
        if agent_id not in self._agents_with_geometry_intro:
            geometry_desc = self._describe_geometry(agent_level)
            observations.append(geometry_desc)
            self._agents_with_geometry_intro.add(agent_id)

        # Current location
        zone = self.spatial_analyzer.identify_zone(position)
        observations.append(f"You are in the {zone}.")

        # Crowd density (categorized to prevent constant LLM calls as people move)
        num_nearby = len(nearby_agents)
        density = CrowdAnalyzer.categorize_density(num_nearby)
        observations.append(f"The area is {density}.")

        # Phase 4.1: Agent's own status
        status_lines = ObservationFormatter.format_own_status(
            agent_id, agent_injured, agent_action, state_queries
        )
        observations.extend(status_lines)

        # Nearby agent behaviors
        if nearby_agents:
            behaviors = self.crowd_analyzer.summarize_behaviors(nearby_agents, agent_injured)
            observations.append(behaviors)

            # Phase 5.1: List nearby agent IDs for targeting messages
            nearby_ids = ObservationFormatter.format_nearby_agent_ids(nearby_agents)
            observations.extend(nearby_ids)

            # Exit crowd information (Phase 4.2: helps agents make informed route decisions)
            exit_crowds = self.crowd_analyzer.count_agents_per_exit(nearby_agents)
            exit_crowd_lines = ObservationFormatter.format_exit_crowds(
                exit_crowds, CrowdAnalyzer.categorize_count
            )
            observations.extend(exit_crowd_lines)

            # Phase 4.3: Overall movement pattern information for information seeking
            movement_pattern = CrowdAnalyzer.analyze_movement_pattern(nearby_agents)
            if movement_pattern:
                observations.append(movement_pattern)

        # Recent events
        event_lines = ObservationFormatter.format_events(events)
        observations.extend(event_lines)

        # Phase 5: Messages from nearby people
        message_lines = ObservationFormatter.format_received_messages(received_messages)
        observations.extend(message_lines)

        # Add conversation history context for active conversations
        if conversation_history:
            # Only show conversations with nearby people who have exchanged multiple messages
            active_conversations = []
            nearby_ids = {a.get("id") for a in nearby_agents}

            for other_agent_id, messages in conversation_history.items():
                if other_agent_id in nearby_ids and len(messages) >= 2:  # At least 2 exchanges
                    # Get last 3 messages in this conversation
                    recent = messages[-3:]
                    convo_summary = []
                    for m in recent:
                        direction = (
                            "You"
                            if m["from"] == agent_id
                            else other_agent_id.replace("agent_", "Person ")
                        )
                        convo_summary.append(f'{direction}: "{m["text"]}"')

                    active_conversations.append(
                        {
                            "other": other_agent_id.replace("agent_", "Person "),
                            "summary": " → ".join(convo_summary),
                        }
                    )

            if active_conversations:
                observations.append("Recent conversation context:")
                for convo in active_conversations[:2]:  # Max 2 to keep it concise
                    observations.append(f"  - With {convo['other']}: {convo['summary']}")

        # Visual observation of blocked exits (Phase 4.2: Realistic discovery)
        visible_blocked = self.spatial_analyzer.get_visible_blocked_exits(position, blocked_exits)
        blocked_lines = ObservationFormatter.format_blocked_exits(visible_blocked)
        observations.extend(blocked_lines)

        # Exit information (level-specific)
        nearest_exit = self.spatial_analyzer.get_nearest_exit_info(
            position, agent_level, self.jps_sim
        )
        observations.append(f"Nearest exit: {nearest_exit}")

        return " ".join(observations)

    def _describe_geometry(self, agent_level: str | None = None) -> str:
        """Create a short natural language summary of the station geometry for the agent's current level."""
        # Get level-specific exits if multi-level simulation
        if agent_level and self.jps_sim and hasattr(self.jps_sim, "simulations"):
            # Multi-level: get exits from the agent's current level
            level_sim = self.jps_sim.simulations.get(agent_level)
            if level_sim:
                exit_names = list(level_sim.exit_manager.evacuation_exits.keys())
            else:
                exit_names = []
        else:
            # Single-level or fallback: use station_layout exits
            exit_names = list(self.spatial_analyzer.exits_polygons.keys()) or list(
                self.spatial_analyzer.exits.keys()
            )

        # Check if we have escalator exits (indicates platform level)
        escalator_exits = [name for name in exit_names if name.startswith("escalator_")]
        street_exits = [name for name in exit_names if not name.startswith("escalator_")]

        if escalator_exits:
            # Platform level - be clear about the two-stage evacuation process
            up_escalators = [name for name in escalator_exits if "_up" in name]
            down_escalators = [name for name in escalator_exits if "_down" in name]

            escalator_desc = "You are on the PLATFORM LEVEL (underground). "
            escalator_desc += (
                "To evacuate, you must first take an escalator UP to the concourse level. "
            )
            escalator_desc += "The street exits (Blackett Street, Grey Street, Eldon Square) are on the concourse level above you. "
            escalator_desc += f"\\nAvailable escalators going UP: {len(up_escalators)} options"
            if down_escalators:
                escalator_desc += (
                    f" | Going DOWN: {len(down_escalators)} (away from exits, not for evacuation)"
                )
            return escalator_desc
        elif street_exits:
            # Concourse level - list street exits but don't mention platforms
            exit_list = ", ".join(street_exits)
            return f"You are on the CONCOURSE LEVEL. Final street exits: {exit_list}."
        else:
            # Fallback for other configurations
            return "You are in the station."
