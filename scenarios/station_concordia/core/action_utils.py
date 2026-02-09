"""
Action utilities for parsing and extracting information from agent actions.

This module provides utilities for working with translated actions and
extracting specific information like exit names.
"""

from typing import Any


def extract_exit_name(
    translated_action: dict[str, Any], station_layout: dict[str, Any]
) -> str | None:
    """
    Extract the target exit name from a translated action.

    Args:
        translated_action: Translated action dictionary with action_type and target
        station_layout: Station layout dict with "exits" mapping exit names to coords

    Returns:
        Exit name if action is moving to an exit, None otherwise
    """
    if translated_action["action_type"] != "move":
        return None

    target_coords = translated_action.get("target")
    if not target_coords:
        return None

    # Match coordinates to exit name
    for exit_name, exit_coords in station_layout["exits"].items():
        # Check if coordinates match (within 1m tolerance)
        if (
            abs(target_coords[0] - exit_coords[0]) < 1.0
            and abs(target_coords[1] - exit_coords[1]) < 1.0
        ):
            return exit_name

    return None
