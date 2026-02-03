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
from collections.abc import Mapping
from typing import Any

from concordia.language_model import language_model
from concordia.typing import prefab as prefab_lib


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

    def translate(
        self, agent_id: str, action: str, current_position: tuple[float, float]
    ) -> dict[str, Any]:
        """
        Translate a natural language action to a concrete goal.

        Args:
            agent_id: ID of the acting agent
            action: Natural language action from Concordia agent
            current_position: Agent's current (x, y) position

        Returns:
            Dictionary with:
                - action_type: "move", "wait", "help", "follow"
                - target: Target coordinates (x, y) or agent ID
                - confidence: Parsing confidence (0-1)
                - reasoning: Explanation of translation
        """
        action_lower = action.lower()

        def _find_exit_by_number(text: str) -> tuple[str, tuple[float, float]] | None:
            import re

            match = re.search(r"\b(?:entrance|exit)\s*[_-]?(\d+)\b", text)
            if not match:
                return None
            number = match.group(1)
            for exit_name, exit_coords in self.exits.items():
                if number in exit_name:
                    return exit_name, exit_coords
            return None

        # Pattern matching for common actions
        if "evacuate" in action_lower or "exit" in action_lower or "entrance" in action_lower:
            numbered_exit = _find_exit_by_number(action_lower)
            if numbered_exit:
                exit_name, exit_coords = numbered_exit
                return {
                    "action_type": "move",
                    "target": exit_coords,
                    "confidence": 0.85,
                    "reasoning": f"Moving to {exit_name} exit",
                }

            # Extract exit name if mentioned
            for exit_name, exit_coords in self.exits.items():
                if exit_name.lower() in action_lower:
                    return {
                        "action_type": "move",
                        "target": exit_coords,
                        "confidence": 0.9,
                        "reasoning": f"Moving to {exit_name} exit",
                    }

            # Default to nearest exit
            nearest_exit = self._find_nearest_exit(current_position)
            return {
                "action_type": "move",
                "target": nearest_exit["coords"],
                "confidence": 0.7,
                "reasoning": f"Moving to nearest exit ({nearest_exit['name']})",
            }

        elif (
            "wait" in action_lower
            or "stay here" in action_lower
            or "stay put" in action_lower
            or "remain here" in action_lower
            or "stand by" in action_lower
        ):
            return {
                "action_type": "wait",
                "target": current_position,
                "confidence": 1.0,
                "reasoning": "Staying in current position",
            }

        elif "help" in action_lower or "assist" in action_lower:
            # This would require identifying nearby agents needing help
            return {
                "action_type": "help",
                "target": None,  # Will be resolved by simulation
                "confidence": 0.5,
                "reasoning": "Looking for someone to help (requires nearby agent detection)",
            }

        elif "follow" in action_lower:
            return {
                "action_type": "follow",
                "target": None,  # Will be resolved by identifying crowd flow
                "confidence": 0.6,
                "reasoning": "Following the crowd",
            }

        # Movement intent without explicit exit
        elif any(
            phrase in action_lower
            for phrase in (
                "move",
                "walk",
                "head",
                "go",
                "proceed",
                "leave",
            )
        ):
            nearest_exit = self._find_nearest_exit(current_position)
            return {
                "action_type": "move",
                "target": nearest_exit["coords"],
                "confidence": 0.6,
                "reasoning": f"Moving to nearest exit ({nearest_exit['name']})",
            }

        else:
            # Ambiguous action - use LLM to parse if available
            if self.model:
                # TODO: Use LLM to parse ambiguous action
                pass

            # Default fallback: wait
            return {
                "action_type": "wait",
                "target": current_position,
                "confidence": 0.3,
                "reasoning": f"Unclear action, defaulting to wait: {action}",
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
        self.obstacles = station_layout.get("obstacles", [])
        self._agents_with_geometry_intro: set[str] = set()

    def generate_observation(
        self,
        agent_id: str,
        position: tuple[float, float],
        nearby_agents: list[dict[str, Any]],
        events: list[str],
        sim_time: float,
    ) -> str:
        """
        Generate a natural language observation for an agent.

        Args:
            agent_id: ID of the observing agent
            position: Agent's current (x, y) position
            nearby_agents: List of nearby agents with their info
            events: Recent events (announcements, alarms, etc.)
            sim_time: Current simulation time

        Returns:
            Natural language observation string
        """
        observations = []

        if agent_id not in self._agents_with_geometry_intro:
            observations.append(self._describe_geometry())
            self._agents_with_geometry_intro.add(agent_id)

        # Time
        observations.append(f"[Time: {sim_time:.1f}s]")

        # Current location
        zone = self._identify_zone(position)
        observations.append(f"You are in the {zone}.")

        # Crowd density
        num_nearby = len(nearby_agents)
        if num_nearby == 0:
            density = "empty"
        elif num_nearby < 5:
            density = "sparse"
        elif num_nearby < 15:
            density = "moderate"
        else:
            density = "crowded"

        observations.append(f"The area is {density} with {num_nearby} people nearby.")

        # Nearby agent behaviors
        if nearby_agents:
            behaviors = self._summarize_behaviors(nearby_agents)
            observations.append(behaviors)

        # Recent events
        if events:
            observations.append("Recent events:")
            for event in events[-3:]:  # Last 3 events
                observations.append(f"  - {event}")

        # Exit information
        nearest_exit = self._get_nearest_exit_info(position)
        observations.append(f"Nearest exit: {nearest_exit}")

        return " ".join(observations)

    def _identify_zone(self, position: tuple[float, float]) -> str:
        """Identify which zone a position is in."""
        if self.zones_polygons:
            try:
                from shapely.geometry import Point

                point = Point(position)
                for zone_name, polygon in self.zones_polygons.items():
                    if polygon.contains(point):
                        return zone_name
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

        return f"{nearest_name} ({min_dist:.1f}m away)"
