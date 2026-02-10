"""
Translates natural language actions from Concordia agents into JuPedSim waypoints and goals.

Examples:
    "I will evacuate through the north exit" → waypoint at north exit
    "I will wait here for more information" → stay in current position
    "I will help the person nearby" → move toward nearest agent
"""

import json
from typing import Any

from concordia.language_model import language_model

from scenarios.common.logger import get_logger

logger = get_logger(__name__)


class ActionTranslator:
    """
    Translates natural language actions from Concordia agents into
    JuPedSim waypoints and goals.
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
            wait_reason = data.get("wait_reason")  # Phase 4.3: Information seeking
            speed = data.get("speed")  # Phase 4.3: Dynamic speed selection

            # BUG FIX: Check for help/follow BEFORE checking target_type == "current_position"
            # Otherwise "help" with target_type="current_position" gets converted to "wait"
            if action_type in {"help", "follow"}:
                return {
                    "action_type": action_type,
                    "target": current_position,  # Use current position as placeholder
                    "confidence": 0.4,
                    "reasoning": f"LLM selected {action_type} intent",
                }

            if action_type == "wait" or target_type == "current_position":
                # Phase 4.3: Include wait reason if provided
                wait_reason_str = f" ({wait_reason})" if wait_reason else ""
                # Default speed for wait actions: slow_walk for seeking_information, otherwise null (keep current)
                default_speed = "slow_walk" if wait_reason == "seeking_information" else None
                return {
                    "action_type": "wait",
                    "target": current_position,
                    "confidence": 0.9,
                    "reasoning": f"Agent chose to wait at current position{wait_reason_str}",
                    "wait_reason": wait_reason,  # Pass through for tracking
                    "speed": speed or default_speed,  # Use LLM speed if provided, else default
                }

            if action_type == "move" and target_type == "exit":
                if exit_name == "nearest" or not exit_name:
                    nearest_exit = self._find_nearest_exit(current_position)
                    return {
                        "action_type": "move",
                        "target": nearest_exit["coords"],
                        "confidence": 0.9,
                        "reasoning": f"Moving to nearest exit ({nearest_exit['name']})",
                        "speed": speed,  # Phase 4.3: Dynamic speed
                    }

                if exit_name in self.exits:
                    return {
                        "action_type": "move",
                        "target": self.exits[exit_name],
                        "confidence": 0.95,
                        "reasoning": f"Moving to exit {exit_name}",
                        "speed": speed,  # Phase 4.3: Dynamic speed
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
                        "speed": speed,  # Phase 4.3: Dynamic speed
                    }

        except json.JSONDecodeError:
            logger.warning(f"Failed to parse action as JSON, trying LLM fallback: {action[:100]}")

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
        """Find zone coordinates from zone name in text."""
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
