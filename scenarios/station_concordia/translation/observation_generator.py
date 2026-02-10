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

    def __init__(self, station_layout: dict[str, Any]):
        """
        Initialize the observation generator.

        Args:
            station_layout: Station geometry and zone information
        """
        self.station_layout = station_layout
        self.exits = station_layout.get("exits", {})
        self._agents_with_geometry_intro: set[str] = set()

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
        agent_status: dict[str, str] | None = None,
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
            agent_status: Dict mapping agent_id to status (EVACUATING, HELPING, WAITING, INJURED)
            received_messages: List of messages received from nearby agents
            conversation_history: Dict mapping other_agent_id to conversation history

        Returns:
            Natural language observation string
        """
        observations = []
        if blocked_exits is None:
            blocked_exits = set()
        if agent_status is None:
            agent_status = {}
        if received_messages is None:
            received_messages = []
        if conversation_history is None:
            conversation_history = {}

        # Note: Station layout is now in agent formative memory, not observations
        # This keeps observations stable when nothing changes
        # Time is also omitted as it's not meaningful for agent decisions

        # Current location
        zone = self.spatial_analyzer.identify_zone(position)
        observations.append(f"You are in the {zone}.")

        # Crowd density (categorized to prevent constant LLM calls as people move)
        num_nearby = len(nearby_agents)
        density = CrowdAnalyzer.categorize_density(num_nearby)
        observations.append(f"The area is {density}.")

        # Phase 4.1: Agent's own status
        status_lines = ObservationFormatter.format_own_status(agent_id, agent_status)
        observations.extend(status_lines)

        # Nearby agent behaviors
        if nearby_agents:
            behaviors = self.crowd_analyzer.summarize_behaviors(nearby_agents, agent_status)
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

        # Exit information
        nearest_exit = self.spatial_analyzer.get_nearest_exit_info(position)
        observations.append(f"Nearest exit: {nearest_exit}")

        return " ".join(observations)

    def _describe_geometry(self) -> str:
        """Create a short natural language summary of the station geometry."""
        zone_names = list(self.spatial_analyzer.zones_polygons.keys()) or list(
            self.spatial_analyzer.zones.keys()
        )
        exit_names = list(self.spatial_analyzer.exits_polygons.keys()) or list(
            self.spatial_analyzer.exits.keys()
        )

        # Hardcoded operational context for Newcastle station
        footbridge_note = (
            "Platforms 3–8 are accessed via a footbridge. "
            "Each platform zone has both a flight of stairs and a ramp onto the footbridge."
        )

        platform_zone_note = (
            "Zone mapping: walkable_area_0 contains platforms 5–8; "
            "walkable_area_2 contains platforms 3–4; "
            "walkable_area_3 contains other platforms, the foyer, and all exits/entrances."
        )

        zone_part = (
            f"Zones: {', '.join(zone_names)}." if zone_names else "Zones are not clearly marked."
        )
        exit_part = (
            f"Exits: {', '.join(exit_names)}." if exit_names else "Exits are visible but unnamed."
        )
        return f"Station layout: {zone_part} {exit_part} {platform_zone_note} {footbridge_note}"
