"""
Station network metadata - tracks entrances, exits, and platform access points.
"""

import random
import xml.etree.ElementTree as ET
from typing import Optional

import networkx as nx
from shapely.geometry import Point, Polygon


class StationNetwork:
    """
    Manages station infrastructure metadata including entrances, exits, and platform access.
    Provides methods to query and route between different station locations.
    """

    def __init__(self, stops_file: str, walking_areas_file: Optional[str] = None):
        """
        Initialize station network metadata.

        Args:
            stops_file: Path to osm_stops.add.xml file
        """
        # Entrance/exit edges (hardcoded station boundaries)
        self.entrance_edges = ["258625111", "E8", "1078920102"]
        self.exit_edges = ["258625111", "E8", "1078920102"]

        # Spawn positions for each entrance edge (0.0 = start, -1 = end of edge)
        self.entrance_spawn_positions: dict[str, float] = {
            "258625111": 0.0,
            "E8": -1.0,
            "1078920102": 0.0,
        }

        # Platform access mapping: {busStop_id: access_edge_id}
        self.platform_access: dict[str, str] = {}

        # Routing configuration
        self._init_routing_config()

        # Walkable area polygons for zone lookup
        self.zone_polygons: dict[str, Polygon] = {}

        # Load platform access information from stops file
        self._load_platform_access(stops_file)

        # Load walkable area polygons for zone lookup
        if walking_areas_file:
            self._load_walkable_zones(walking_areas_file)

    def _init_routing_config(self):
        """Initialize routing configuration for footbridge and platform access"""

        # Zone A: Main entrance side
        self.platforms_zone_a = {"317392095", "4492377635", "4267618249"}

        # Zone B: Northern platforms (across footbridge, north exit)
        self.platforms_zone_b = {"4270757351", "4270733515"}

        # Zone C: Southern platforms (across footbridge, south exit)
        self.platforms_zone_c = {"4346939128", "1754217408", "5086156615"}

        # Location zone mapping (for future extension to facilities, etc.)
        self.entrance_zones = {"258625111": "A", "E8": "A", "1078920102": "A"}

        # Build zone routing graph
        self.zone_graph = nx.DiGraph()

        # A → B: Enter footbridge, exit north
        self.zone_graph.add_edge(
            "A", "B", edges=["E10", "540275666#0"], exit_choices=["540275665", "258625791"]
        )

        # A → C: Enter footbridge, continue to south
        self.zone_graph.add_edge(
            "A",
            "C",
            edges=["E10", "540275666#0", "540275666#1"],
            exit_choices=["400897429", "400897430"],
        )

        # B → A: Reverse footbridge path from north to entrance
        self.zone_graph.add_edge(
            "B", "A", edges=["-540275666#0", "-E10"], exit_choices=["258625111", "E8", "1078920102"]
        )

        # C → A: Reverse footbridge path from south to entrance
        self.zone_graph.add_edge(
            "C",
            "A",
            edges=["-540275666#1", "-540275666#0", "-E10"],
            exit_choices=["258625111", "E8", "1078920102"],
        )

        # B → C: Continue on footbridge from north to south
        self.zone_graph.add_edge(
            "B", "C", edges=["540275666#1"], exit_choices=["400897429", "400897430"]
        )

        # C → B: Reverse footbridge from south to north
        self.zone_graph.add_edge(
            "C", "B", edges=["-540275666#1"], exit_choices=["540275665", "258625791"]
        )

    def _load_platform_access(self, stops_file: str):
        """Parse osm_stops.add.xml to extract platform access edges"""
        try:
            tree = ET.parse(stops_file)
            root = tree.getroot()

            # Find all busStops with access elements
            for busstop in root.findall(".//busStop"):
                busstop_id = busstop.get("id")
                access = busstop.find("access")

                if access is not None and busstop_id:
                    access_lane = access.get("lane")
                    if access_lane:
                        # Extract edge from lane (lane format is typically "edge_id_laneIndex")
                        access_edge = access_lane.rsplit("_", 1)[0]
                        self.platform_access[busstop_id] = access_edge

            print(f"Loaded {len(self.platform_access)} platform access mappings")

        except Exception as e:
            print(f"Warning: Could not load platform access data: {e}")

    def _load_walkable_zones(self, walking_areas_file: str):
        """Load walkable area polygons and map them to zones A/B/C."""
        try:
            tree = ET.parse(walking_areas_file)
            root = tree.getroot()

            zone_name_map = {"entrance": "A", "platform_3_to_4": "B", "platform_5_to_7": "C"}

            for poly in root.findall(".//poly"):
                name = poly.get("name")
                poly_type = poly.get("type")
                shape = poly.get("shape")

                if poly_type != "jupedsim.walkable_area" or not name or not shape:
                    continue

                if name not in zone_name_map:
                    continue

                coords = []
                for pair in shape.split():
                    x_str, y_str = pair.split(",")
                    coords.append((float(x_str), float(y_str)))

                zone = zone_name_map[name]
                self.zone_polygons[zone] = Polygon(coords)

            print(f"Loaded zone polygons: {list(self.zone_polygons.keys())}")

        except Exception as e:
            print(f"Warning: Could not load walking area zones: {e}")

    def get_random_entrance_edge(self) -> str:
        """Get a random entrance edge"""
        return random.choice(self.entrance_edges)

    def get_entrance_spawn_position(self, edge_id: str) -> float:
        """Get the spawn position for an entrance edge (0.0=start, -1=end)"""
        return self.entrance_spawn_positions.get(edge_id, -1.0)

    def get_random_exit_edge(self) -> str:
        """Get a random exit edge"""
        return random.choice(self.exit_edges)

    def get_platform_access_edge(self, busstop_id: str) -> Optional[str]:
        """
        Get the access edge for a specific platform busStop.

        Args:
            busstop_id: The busStop ID

        Returns:
            The access edge ID, or None if not found
        """
        return self.platform_access.get(busstop_id)

    def get_all_platforms(self) -> list[str]:
        """Get list of all platform busStop IDs"""
        return list(self.platform_access.keys())

    def get_random_platform(self) -> str:
        """Get a random platform busStop ID"""
        return random.choice(list(self.platform_access.keys()))

    def get_location_side(self, location_id: str) -> Optional[str]:
        """
        Determine which zone of the station a location is in.

        Args:
            location_id: Platform busStop ID, entrance edge ID, or other location ID

        Returns:
            'A', 'B', or 'C' for the zone, None if unknown
        """
        # Check entrances
        if location_id in self.entrance_zones:
            return self.entrance_zones[location_id]

        # Check platforms
        if location_id in self.platforms_zone_a:
            return "A"
        elif location_id in self.platforms_zone_b:
            return "B"
        elif location_id in self.platforms_zone_c:
            return "C"

        return None

    def get_zone_for_xy(self, x: float, y: float) -> Optional[str]:
        """
        Determine zone (A/B/C) from x,y coordinates using walking area polygons.

        Args:
            x: X coordinate
            y: Y coordinate

        Returns:
            'A', 'B', or 'C' for the zone, None if unknown
        """
        if not self.zone_polygons:
            return None

        point = Point(x, y)
        for zone, polygon in self.zone_polygons.items():
            if polygon.contains(point):
                return zone

        return None

    def get_route_from_xy_to_entrance(self, x: float, y: float) -> list[str]:
        """
        Build a zone-based route from arbitrary x,y coordinates to an entrance edge.

        Args:
            x: X coordinate
            y: Y coordinate

        Returns:
            List of edge IDs to reach an entrance edge
        """
        from_zone = self.get_zone_for_xy(x, y)
        entrance_edge = self.get_random_entrance_edge()

        if from_zone is None:
            return [entrance_edge]

        # Already in zone A - head directly to an entrance edge
        if from_zone == "A":
            return [entrance_edge]

        route: list[str] = []

        try:
            zone_path = nx.shortest_path(self.zone_graph, from_zone, "A")

            for i in range(len(zone_path) - 1):
                current_zone = zone_path[i]
                next_zone = zone_path[i + 1]

                edge_data = self.zone_graph.get_edge_data(current_zone, next_zone)
                if edge_data:
                    route.extend(edge_data["edges"])
                    if "exit_choices" in edge_data:
                        route.append(random.choice(edge_data["exit_choices"]))

        except nx.NetworkXNoPath:
            pass

        route.append(entrance_edge)
        return route

    def requires_footbridge(self, from_location: str, to_location: str) -> bool:
        """
        Check if footbridge crossing is required between two locations.
        Footbridge connects zones: A ↔ B ↔ C

        Args:
            from_location: Origin location ID (entrance, platform, facility, etc.)
            to_location: Destination location ID

        Returns:
            True if footbridge crossing needed, False otherwise
        """
        from_zone = self.get_location_side(from_location)
        to_zone = self.get_location_side(to_location)

        # If we can't determine zones, assume bridge not needed
        if from_zone is None or to_zone is None:
            return False

        # Bridge needed if zones differ (A↔B, A↔C, B↔C all require bridge)
        return from_zone != to_zone

    def get_route(self, from_location: str, to_location: str) -> list[str]:
        """
        Compute the edge sequence between two locations using zone graph.

        Args:
            from_location: Starting location ID (entrance edge, platform ID, etc.)
            to_location: Destination location ID (typically platform busStop ID)

        Returns:
            List of edge IDs to traverse
        """
        route = [from_location]

        from_zone = self.get_location_side(from_location)
        to_zone = self.get_location_side(to_location)

        # Same zone - direct route
        if from_zone == to_zone:
            access_edge = self.get_platform_access_edge(to_location)
            if access_edge and access_edge != from_location:
                route.append(access_edge)
            return route

        # Different zones - find path through zone graph
        if from_zone is None or to_zone is None:
            print(f"Warning: Cannot determine zones for {from_location} → {to_location}")
            return route

        try:
            # Find zone path using networkx
            zone_path = nx.shortest_path(self.zone_graph, from_zone, to_zone)

            # Build edge sequence by following zone path
            for i in range(len(zone_path) - 1):
                current_zone = zone_path[i]
                next_zone = zone_path[i + 1]

                # Get edge data for this zone transition
                edge_data = self.zone_graph.get_edge_data(current_zone, next_zone)
                if edge_data:
                    # Add footbridge edges
                    route.extend(edge_data["edges"])

                    # Add exit choice (random selection from options)
                    if "exit_choices" in edge_data:
                        exit_edge = random.choice(edge_data["exit_choices"])
                        route.append(exit_edge)

            # Add final platform access edge
            access_edge = self.get_platform_access_edge(to_location)
            if access_edge and access_edge != route[-1]:
                route.append(access_edge)

        except nx.NetworkXNoPath:
            print(f"Warning: No path from zone {from_zone} to {to_zone}")
            # Fallback: just add access edge
            access_edge = self.get_platform_access_edge(to_location)
            if access_edge:
                route.append(access_edge)

        return route

    def __repr__(self):
        return (
            f"StationNetwork(entrances={len(self.entrance_edges)}, "
            f"exits={len(self.exit_edges)}, "
            f"platforms={len(self.platform_access)})"
        )
