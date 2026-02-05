"""
Custom Game Master for station evacuation scenarios.

The Game Master interfaces between Concordia's cognitive layer and JuPedSim's
movement simulation, handling:
- Observation generation from JuPedSim state
- Action translation to waypoints
- Event broadcasting (announcements, alarms)
- Turn management
"""

import dataclasses
import json
from collections.abc import Mapping
from typing import Any

from concordia.language_model import language_model
from concordia.typing import prefab as prefab_lib

from scenarios.common.logger import get_logger

logger = get_logger(__name__)


@dataclasses.dataclass
class StationEvacuationGM(prefab_lib.Prefab):
    """
    Game Master for station evacuation scenarios.

    Manages the simulation environment, translates between Concordia's
    natural language interface and JuPedSim's geometric representation.
    """

    description: str = (
        "A Game Master that manages station evacuation simulations, "
        "interfacing between Concordia agents and JuPedSim movement."
    )

    params: Mapping[str, Any] = dataclasses.field(
        default_factory=lambda: {
            "name": "StationMaster",
            "jupedsim_interface": None,  # Will be set to JuPedSim simulation
            "station_layout": None,  # Station geometry information
            "event_system": None,  # Event system for announcements
        }
    )

    def build(
        self,
        model: language_model.LanguageModel,
    ):
        """
        Build the Game Master with evacuation-specific components.

        Args:
            model: Language model for reasoning

        Returns:
            Configured game master entity
        """
        # TODO: Implement custom GM components
        # For now, this is a placeholder structure
        # Full implementation will include:
        # 1. ObservationGenerator - converts JuPedSim state to NL
        # 2. ActionResolver - translates NL actions to waypoints
        # 3. EventBroadcaster - manages announcements and alarms
        # 4. TurnManager - decides when agents make decisions

        raise NotImplementedError(
            "StationEvacuationGM is a work in progress. "
            "Use the translation layer approach for initial implementation."
        )


class ActionTranslator:
    """
    Translates natural language actions from Concordia agents into
    JuPedSim waypoints and goals.

    Examples:
        "I will evacuate through the north exit" → waypoint at north exit
        "I will wait here for more information" → stay in current position
        "I will help the person nearby" → move toward nearest agent
    """

    def __init__(
        self, station_layout: dict[str, Any], model: language_model.LanguageModel | None = None
    ):
        """
        Initialize the action translator.

        Args:
            station_layout: Dictionary with station geometry info (exits, zones, etc.)
            model: Optional LLM for ambiguous action parsing
        """
        self.station_layout = station_layout
        self.model = model

        # Define exit locations from layout
        self.exits = station_layout.get("exits", {})
        self.zones = station_layout.get("zones", {})
        self.zones_polygons = station_layout.get("zones_polygons", {})

    def translate(
        self, agent_id: str, action: str, current_position: tuple[float, float]
    ) -> dict[str, Any]:
        """
        Translate a JSON action response to a concrete goal.

        Args:
            agent_id: ID of the acting agent
            action: JSON action from Concordia agent
            current_position: Agent's current (x, y) position

        Returns:
            Dictionary with:
                - action_type: "move", "wait", "help", "follow"
                - target: Target coordinates (x, y) or agent ID
                - confidence: Parsing confidence (0-1)
                - reasoning: Explanation of translation
        """
        # Try parsing as JSON first
        try:
            # Strip agent name prefix (e.g., "Agent 0 {" -> "{")
            json_start = action.find("{")
            if json_start > 0:
                action = action[json_start:]

            data = json.loads(action)
            action_type = data.get("action_type")
            target_type = data.get("target_type")
            exit_name = data.get("exit_name")
            zone_name = data.get("zone_name")

            if action_type == "wait" or target_type == "current_position":
                return {
                    "action_type": "wait",
                    "target": current_position,
                    "confidence": 0.9,
                    "reasoning": "Agent chose to wait at current position",
                }

            if action_type == "move" and target_type == "exit":
                if exit_name == "nearest" or not exit_name:
                    nearest_exit = self._find_nearest_exit(current_position)
                    return {
                        "action_type": "move",
                        "target": nearest_exit["coords"],
                        "confidence": 0.9,
                        "reasoning": f"Moving to nearest exit ({nearest_exit['name']})",
                    }

                if exit_name in self.exits:
                    return {
                        "action_type": "move",
                        "target": self.exits[exit_name],
                        "confidence": 0.95,
                        "reasoning": f"Moving to exit {exit_name}",
                    }

            if action_type == "move" and target_type == "zone" and zone_name:
                zone_target = self._find_zone_target(zone_name.lower())
                if zone_target:
                    zone_name, zone_coords = zone_target
                    return {
                        "action_type": "move",
                        "target": zone_coords,
                        "confidence": 0.9,
                        "reasoning": f"Moving to zone {zone_name}",
                    }

        except json.JSONDecodeError:
            logger.warning(f"Failed to parse action as JSON, trying LLM fallback: {action[:100]}")

        # Fallback to LLM if JSON parsing fails
        if self.model:
            llm_result = self._interpret_with_llm(action, current_position)
            if llm_result:
                return llm_result

        return {
            "action_type": "wait",
            "target": current_position,
            "confidence": 0.3,
            "reasoning": f"Parse failed, defaulting to wait: {action[:100]}",
        }

    def _find_nearest_exit(self, position: tuple[float, float]) -> dict[str, Any]:
        """Find the nearest exit to a given position."""
        min_dist = float("inf")
        nearest = None

        for exit_name, exit_coords in self.exits.items():
            dist = (
                (position[0] - exit_coords[0]) ** 2 + (position[1] - exit_coords[1]) ** 2
            ) ** 0.5
            if dist < min_dist:
                min_dist = dist
                nearest = {"name": exit_name, "coords": exit_coords}

        return nearest if nearest else {"name": "default", "coords": (0, 0)}

    def _find_zone_target(self, text: str) -> tuple[str, tuple[float, float]] | None:
        for zone_name in self.zones_polygons.keys():
            if zone_name.lower() in text:
                polygon = self.zones_polygons[zone_name]
                centroid = polygon.centroid
                return zone_name, (centroid.x, centroid.y)

        for zone_name, zone_bounds in self.zones.items():
            if zone_name.lower() in text:
                x_center = (zone_bounds["x_min"] + zone_bounds["x_max"]) / 2
                y_center = (zone_bounds["y_min"] + zone_bounds["y_max"]) / 2
                return zone_name, (x_center, y_center)

        return None

    def _interpret_with_llm(
        self, action: str, current_position: tuple[float, float]
    ) -> dict[str, Any] | None:
        if not self.model:
            return None

        try:
            exits = []
            for name, coords in self.exits.items():
                dx = current_position[0] - coords[0]
                dy = current_position[1] - coords[1]
                dist = (dx**2 + dy**2) ** 0.5
                exits.append({"name": name, "distance_m": round(dist, 1)})

            zones = list(self.zones_polygons.keys()) or list(self.zones.keys())

            prompt = (
                "You are the simulation game master. Interpret the agent action into a structured intent.\n"
                "Return ONLY valid JSON with these keys: action_type, target_type, exit_name, zone_name.\n"
                "- action_type: wait | move | help | follow\n"
                "- target_type: current_position | exit | zone | none\n"
                "- exit_name: one of the exit names listed below or 'nearest' or null\n"
                "- zone_name: one of the zone names listed below or null\n"
                "Rules: If the action mainly describes waiting/monitoring and movement is conditional, choose wait.\n"
                "If there is no explicit target, prefer wait.\n\n"
                f"Exits: {exits}\n"
                f"Zones: {zones}\n\n"
                f"Action: {action}\n"
            )

            response = self.model.sample_text(prompt, max_tokens=200)
            try:
                data = json.loads(response)
            except json.JSONDecodeError:
                logger.warning(
                    "LLM action interpreter returned non-JSON response. Response=%s",
                    response,
                )
                return None

            action_type = data.get("action_type")
            target_type = data.get("target_type")
            exit_name = data.get("exit_name")
            zone_name = data.get("zone_name")

            if action_type == "wait" or target_type == "current_position":
                return {
                    "action_type": "wait",
                    "target": current_position,
                    "confidence": 0.55,
                    "reasoning": "LLM classified intent as wait",
                }

            if action_type == "move" and target_type == "exit":
                if exit_name == "nearest" or not exit_name:
                    nearest_exit = self._find_nearest_exit(current_position)
                    return {
                        "action_type": "move",
                        "target": nearest_exit["coords"],
                        "confidence": 0.55,
                        "reasoning": f"LLM selected nearest exit ({nearest_exit['name']})",
                    }

                if exit_name in self.exits:
                    return {
                        "action_type": "move",
                        "target": self.exits[exit_name],
                        "confidence": 0.6,
                        "reasoning": f"LLM selected exit {exit_name}",
                    }

            if action_type == "move" and target_type == "zone" and zone_name:
                zone_target = self._find_zone_target(zone_name.lower())
                if zone_target:
                    zone_name, zone_coords = zone_target
                    return {
                        "action_type": "move",
                        "target": zone_coords,
                        "confidence": 0.55,
                        "reasoning": f"LLM selected zone {zone_name}",
                    }

            if action_type in {"help", "follow"}:
                return {
                    "action_type": action_type,
                    "target": None,
                    "confidence": 0.4,
                    "reasoning": f"LLM selected {action_type} intent",
                }

            return None
        except Exception as e:
            logger.warning("LLM action interpreter failed: %s", e)
            return None


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

        Returns:
            Natural language observation string
        """
        observations = []
        if blocked_exits is None:
            blocked_exits = set()

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

        # Nearby agent behaviors
        if nearby_agents:
            behaviors = self._summarize_behaviors(nearby_agents)
            observations.append(behaviors)

            # Exit crowd information (Phase 4.2: helps agents make informed route decisions)
            exit_crowds = self._count_agents_per_exit(nearby_agents)
            if exit_crowds:
                observations.append("People heading toward exits:")
                for exit_name, count in sorted(exit_crowds.items()):
                    observations.append(f"  - {exit_name}: {self._categorize_count(count)}")

        # Recent events
        if events:
            observations.append("Recent events:")
            for event in events[-3:]:  # Last 3 events
                observations.append(f"  - {event}")

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

    def _summarize_behaviors(self, nearby_agents: list[dict[str, Any]]) -> str:
        """Summarize what nearby agents are doing."""
        # Count movement patterns
        moving_count = sum(1 for a in nearby_agents if a.get("is_moving", True))
        waiting_count = len(nearby_agents) - moving_count

        if moving_count > waiting_count:
            return "Most people are moving toward exits."
        elif waiting_count > moving_count:
            return "Many people are waiting or stationary."
        else:
            return "People are mixed between moving and waiting."

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
