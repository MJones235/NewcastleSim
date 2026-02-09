"""
Generates natural language observations from JuPedSim simulation state.

Converts geometric and simulation data into observations that Concordia
agents can reason about.
"""

from typing import Any

from scenarios.common.logger import get_logger

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
        self.zones = station_layout.get("zones", {})
        self.zones_polygons = station_layout.get("zones_polygons", {})
        self.exits = station_layout.get("exits", {})
        self.exits_polygons = station_layout.get("exits_polygons", {})
        self.walkable_areas = station_layout.get("walkable_areas", {})
        self.obstacles = station_layout.get("obstacles", [])
        self._agents_with_geometry_intro: set[str] = set()

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
        zone = self._identify_zone(position)
        observations.append(f"You are in the {zone}.")

        # Crowd density (categorized to prevent constant LLM calls as people move)
        num_nearby = len(nearby_agents)
        if num_nearby == 0:
            density = "empty (no one nearby)"
        elif num_nearby <= 3:
            density = "sparse (a few people nearby)"
        elif num_nearby <= 10:
            density = "moderate crowd nearby"
        else:
            density = "crowded (many people nearby)"

        observations.append(f"The area is {density}.")

        # Phase 4.1: Agent's own status
        own_status = agent_status.get(agent_id, "EVACUATING")
        if own_status == "HELPING":
            # Find who they're helping
            for helped_id, helper_id in agent_status.items():
                if helped_id.startswith("helped_by_") and helper_id == agent_id:
                    observations.append("You are currently helping another person.")
                    break
        elif own_status == "INJURED":
            observations.append("You are injured and moving slowly.")
        elif own_status == "WAITING":
            observations.append("You are waiting for assistance.")

        # Nearby agent behaviors
        if nearby_agents:
            behaviors = self._summarize_behaviors(nearby_agents, agent_status)
            observations.append(behaviors)

            # Phase 5.1: List nearby agent IDs for targeting messages
            if len(nearby_agents) > 0 and len(nearby_agents) <= 5:
                # Only list IDs when there are a few people (not in crowds)
                nearby_ids = [a.get("id") for a in nearby_agents[:5] if a.get("id")]
                if nearby_ids:
                    observations.append(f"Nearby: {', '.join(nearby_ids)}")

            observations.append(behaviors)

            # Exit crowd information (Phase 4.2: helps agents make informed route decisions)
            exit_crowds = self._count_agents_per_exit(nearby_agents)
            if exit_crowds:
                observations.append("People heading toward exits:")
                for exit_name, count in sorted(exit_crowds.items()):
                    observations.append(f"  - {exit_name}: {self._categorize_count(count)}")

            # Phase 4.3: Overall movement pattern information for information seeking
            moving_count = sum(1 for a in nearby_agents if a.get("is_moving", True))
            if len(nearby_agents) > 0:
                moving_pct = (moving_count / len(nearby_agents)) * 100
                if moving_pct > 70:
                    observations.append(
                        "Most people around you are moving purposefully toward exits."
                    )
                elif moving_pct > 40:
                    observations.append(
                        "The crowd is mixed - some evacuating, others waiting or uncertain."
                    )
                else:
                    observations.append(
                        "Many people around you are waiting or looking for information."
                    )

        # Recent events
        if events:
            observations.append("Recent events:")
            for event in events[-3:]:  # Last 3 events
                observations.append(f"  - {event}")

        # Phase 5: Messages from nearby people
        if received_messages:
            # Show recent unique messages (last 5)
            unique_messages = []
            seen_texts = set()
            for msg in reversed(received_messages):
                msg_key = msg["text"][:30].lower()  # Match dedup key
                if msg_key not in seen_texts:
                    unique_messages.append(msg)
                    seen_texts.add(msg_key)
                if len(unique_messages) >= 5:
                    break

            if unique_messages:
                observations.append("What people just said to you:")
                for msg in reversed(unique_messages):
                    sender_name = msg["from"].replace("agent_", "Person ")
                    msg_type = msg.get("message_type", "")
                    type_indicator = {
                        "directed": " (to you)",
                        "quiet": " (quietly)",
                        "shout": " (shouting)",
                    }.get(msg_type, "")
                    observations.append(f'  - {sender_name}{type_indicator}: "{msg["text"]}"')

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
        # Only observe blocked exits within visual range (~20m)
        if blocked_exits:
            visible_blocked = []
            for exit_name in blocked_exits:
                if exit_name in self.exits:
                    exit_pos = self.exits[exit_name]
                    distance = (
                        (position[0] - exit_pos[0]) ** 2 + (position[1] - exit_pos[1]) ** 2
                    ) ** 0.5

                    # Visual range: 20m
                    if distance < 20.0:
                        if distance >= 50:
                            dist_cat = "50-100m"
                        elif distance >= 10:
                            dist_cat = "<50m"
                        else:
                            dist_cat = "very close"

                        visible_blocked.append({"name": exit_name, "distance": dist_cat})

            if visible_blocked:
                observations.append("⚠️ Visual observations:")
                for blocked in visible_blocked:
                    observations.append(
                        f"  - The {blocked['name']} appears blocked/obstructed "
                        f"({blocked['distance']} away)"
                    )

        # Exit information
        nearest_exit = self._get_nearest_exit_info(position)
        observations.append(f"Nearest exit: {nearest_exit}")

        return " ".join(observations)

    def _identify_zone(self, position: tuple[float, float]) -> str:
        """Identify which zone a position is in.

        Priority order (highest to lowest):
        1. Main footbridge (foot_bridge)
        2. Platform zones (jps.platform_N)
        3. Connector zones (platform_N_to_M)
        4. General zones (everything else)
        """

        def _covers_or_contains(polygon, point):
            try:
                if polygon.covers(point):
                    return True
            except Exception:
                pass
            try:
                return polygon.contains(point)
            except Exception:
                return False

        def _is_main_footbridge(zone_name: str) -> bool:
            """Check if this is the main footbridge zone."""
            name_lower = zone_name.lower()
            # Match: foot_bridge, footbridge (but not connectors)
            return "foot" in name_lower and "bridge" in name_lower and "_to_" not in name_lower

        def _is_platform_zone(zone_name: str) -> bool:
            """Check if this is a platform zone."""
            name_lower = zone_name.lower()
            # Match: jps.platform_3, platform_3 (but not platform_3_to_4)
            return "platform" in name_lower and "_to_" not in name_lower

        def _is_connector_zone(zone_name: str) -> bool:
            """Check if this is a connector zone between platforms."""
            name_lower = zone_name.lower()
            # Match: platform_3_to_4, platform_1_to_2, etc.
            return "platform_" in name_lower and "_to_" in name_lower

        if self.zones_polygons:
            try:
                from shapely.geometry import Point

                point = Point(position)

                # Priority 1: Check main footbridge first
                for zone_name, polygon in self.zones_polygons.items():
                    if _is_main_footbridge(zone_name):
                        if _covers_or_contains(polygon, point):
                            return zone_name

                # Priority 2: Check platform zones
                for zone_name, polygon in self.zones_polygons.items():
                    if _is_platform_zone(zone_name):
                        if _covers_or_contains(polygon, point):
                            return zone_name

                # Priority 3: Check connector zones
                for zone_name, polygon in self.zones_polygons.items():
                    if _is_connector_zone(zone_name):
                        if _covers_or_contains(polygon, point):
                            return zone_name

                # Priority 4: Check all other zones
                for zone_name, polygon in self.zones_polygons.items():
                    if (
                        not _is_main_footbridge(zone_name)
                        and not _is_platform_zone(zone_name)
                        and not _is_connector_zone(zone_name)
                    ):
                        if _covers_or_contains(polygon, point):
                            return zone_name
            except Exception:
                pass

        if self.walkable_areas:
            try:
                from shapely.geometry import Point

                point = Point(position)
                for area_name, polygon in self.walkable_areas.items():
                    if _covers_or_contains(polygon, point):
                        return area_name
            except Exception:
                pass

        # Fallback to rectangular bounds
        for zone_name, zone_bounds in self.zones.items():
            if self._point_in_bounds(position, zone_bounds):
                return zone_name
        return "unknown area"

    def _describe_geometry(self) -> str:
        """Create a short natural language summary of the station geometry."""
        zone_names = list(self.zones_polygons.keys()) or list(self.zones.keys())
        exit_names = list(self.exits_polygons.keys()) or list(self.exits.keys())

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

    def _point_in_bounds(self, point: tuple[float, float], bounds: dict) -> bool:
        """Check if a point is within rectangular bounds."""
        x, y = point
        return bounds["x_min"] <= x <= bounds["x_max"] and bounds["y_min"] <= y <= bounds["y_max"]

    def _summarize_behaviors(
        self, nearby_agents: list[dict[str, Any]], agent_status: dict[str, str]
    ) -> str:
        """Summarize what nearby agents are doing."""
        # Phase 4.1: Detect injured/slow-moving agents
        injured_nearby = []
        helping_nearby = []

        for agent in nearby_agents:
            agent_id = agent.get("id")
            if agent_id:
                status = agent_status.get(agent_id, "EVACUATING")
                distance = agent.get("distance", 999)

                if (
                    status == "INJURED" and distance < 20.0
                ):  # Within 20m - can see struggling person from distance
                    injured_nearby.append(agent_id)
                elif status == "HELPING" and distance < 20.0:
                    helping_nearby.append(agent_id)

        # Build behavior summary
        parts = []

        # Count movement patterns
        moving_count = sum(1 for a in nearby_agents if a.get("is_moving", True))
        waiting_count = len(nearby_agents) - moving_count

        if moving_count > waiting_count:
            parts.append("Most people are moving toward exits.")
        elif waiting_count > moving_count:
            parts.append("Many people are waiting or stationary.")
        else:
            parts.append("People are mixed between moving and waiting.")

        # Phase 4.1: Note injured agents nearby
        if injured_nearby:
            if len(injured_nearby) == 1:
                parts.append(
                    f"You notice {injured_nearby[0]} appears injured or moving very slowly."
                )
            else:
                parts.append(
                    f"You notice {len(injured_nearby)} people nearby appear injured or moving very slowly: {', '.join(injured_nearby[:3])}"
                )

        if helping_nearby:
            parts.append("Someone nearby is helping another person.")

        return " ".join(parts)

    def _count_agents_per_exit(self, nearby_agents: list[dict[str, Any]]) -> dict[str, int]:
        """
        Count how many nearby agents appear to be heading toward each exit.

        Returns:
            Dict mapping exit name to approximate agent count
        """
        exit_counts: dict[str, int] = {}

        for agent in nearby_agents:
            # Check if agent has a target exit (if available from agent data)
            target_exit = agent.get("target_exit")
            if target_exit and target_exit in self.exits:
                exit_counts[target_exit] = exit_counts.get(target_exit, 0) + 1

        return exit_counts

    def _categorize_count(self, count: int) -> str:
        """Categorize people count to prevent minor changes from triggering LLM."""
        if count == 0:
            return "empty"
        elif count <= 3:
            return "sparse (few people)"
        elif count <= 10:
            return "moderate crowd"
        else:
            return "crowded (many people)"

    def _get_nearest_exit_info(self, position: tuple[float, float]) -> str:
        """Get information about the nearest exit."""
        exits = self.station_layout.get("exits", {})
        if not exits:
            return "unknown"

        min_dist = float("inf")
        nearest_name = "unknown"

        for name, coords in exits.items():
            dist = ((position[0] - coords[0]) ** 2 + (position[1] - coords[1]) ** 2) ** 0.5
            if dist < min_dist:
                min_dist = dist
                nearest_name = name

        # Categorize distance to prevent small changes from triggering LLM calls
        if min_dist >= 100:
            dist_category = "100m+"
        elif min_dist >= 50:
            dist_category = "50-100m"
        else:
            dist_category = "<50m"
        return f"{nearest_name} ({dist_category})"
