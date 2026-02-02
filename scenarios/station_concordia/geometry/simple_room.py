"""
Simple rectangular room geometry for testing Concordia integration.

This provides a basic 100m x 100m room with clear walkable areas and exits,
making it easy to test and debug the Concordia + JuPedSim integration.
"""

from shapely.geometry import Point, Polygon


def create_simple_room() -> dict:
    """
    Create a simple rectangular room geometry for testing.

    Returns:
        Dictionary with walkable_areas, entrance_areas, platform_areas
    """
    # Main room: 100m x 100m square
    room = Polygon(
        [
            (0, 0),
            (100, 0),
            (100, 100),
            (0, 100),
            (0, 0),
        ]
    )

    # Exits at each corner (5m radius circles)
    exit_north = Point(50, 95).buffer(5.0)
    exit_south = Point(50, 5).buffer(5.0)
    exit_east = Point(95, 50).buffer(5.0)
    exit_west = Point(5, 50).buffer(5.0)

    # Central platform area
    platform = Polygon(
        [
            (40, 40),
            (60, 40),
            (60, 60),
            (40, 60),
            (40, 40),
        ]
    )

    return {
        "walkable_areas": {
            "main_room": room,
        },
        "entrance_areas": {
            "exit_north": exit_north,
            "exit_south": exit_south,
            "exit_east": exit_east,
            "exit_west": exit_west,
        },
        "platform_areas": {
            "central_platform": platform,
        },
    }
