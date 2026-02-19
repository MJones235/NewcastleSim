"""
Level transfer manager for multi-level station simulations.

Handles agent teleportation between levels via escalators/stairs.
Automatically builds transfer mappings from geometry file naming conventions.

Naming convention:
  L{level}_esc_{id}_{direction}
  Examples: L0_esc_a_down, L-1_esc_a_down, L0_esc_b_up

Transfer logic:
  - Agents entering escalator zone on one level
  - Teleport to matching escalator zone on other level
  - Maintain relative position within zone
  - Apply optional traversal delay
"""

import re
import xml.etree.ElementTree as ET
from pathlib import Path

from shapely.geometry import Point, Polygon

from scenarios.common.logger import get_logger

logger = get_logger(__name__)


class LevelTransferManager:
    """Manages escalator/stair transfers between levels."""

    def __init__(self, network_path: Path, levels: list[str]):
        """
        Initialize transfer manager.

        Args:
            network_path: Path to network geometry folder (contains level_*.xml files)
            levels: List of level IDs to load (e.g., ["0", "-1"])
        """
        self.network_path = Path(network_path)
        self.levels = levels
        self.escalator_zones: dict[str, Polygon] = {}  # zone_name -> polygon
        self.transfer_mappings: dict[str, tuple[str, str]] = (
            {}
        )  # source_zone -> (target_zone, target_level)
        self.transfer_history: dict[str, float] = (
            {}
        )  # agent_id -> last_transfer_time (for cooldown)

        self._load_escalator_zones()
        self._build_transfer_mappings()

    def _load_escalator_zones(self):
        """
        Load all escalator zones from geometry files.

        Searches for polygons matching pattern: L{level}_esc_{id}_{direction}
        """
        escalator_pattern = re.compile(r"^L([^_]+)_esc_([a-f])_(up|down)$")

        for level in self.levels:
            level_file = self.network_path / f"level_{level}.xml"

            if not level_file.exists():
                logger.warning(f"Level file not found: {level_file}")
                continue

            try:
                tree = ET.parse(level_file)
                root = tree.getroot()

                for poly in root.findall(".//poly"):
                    zone_name = poly.get("name")
                    if not zone_name:
                        continue

                    # Check if this is an escalator zone
                    match = escalator_pattern.match(zone_name)
                    if not match:
                        continue

                    # Parse shape
                    shape_str = poly.get("shape")
                    if not shape_str:
                        continue

                    coords = self._parse_shape_string(shape_str)
                    polygon = Polygon(coords)

                    self.escalator_zones[zone_name] = polygon
                    logger.debug(f"  Loaded escalator zone: {zone_name}")

            except Exception as e:
                logger.error(f"Error loading escalator zones from {level_file}: {e}")

        logger.info(f"Loaded {len(self.escalator_zones)} escalator zones")

    def _build_transfer_mappings(self):
        """
        Build automatic transfer mappings from escalator zones.

        Groups zones by escalator ID (letter) and creates bidirectional transfers.
        Example:
          L0_esc_a_down + L-1_esc_a_down → bidirectional transfer
          L0_esc_b_up + L-1_esc_b_up → bidirectional transfer
        """
        escalator_pattern = re.compile(r"^L([^_]+)_esc_([a-f])_(up|down)$")
        escalators_by_id: dict[str, dict[str, str]] = {}  # esc_id -> {level -> zone_name}

        # Group zones by escalator ID
        for zone_name in self.escalator_zones.keys():
            match = escalator_pattern.match(zone_name)
            if not match:
                continue

            level, esc_id, direction = match.groups()
            if esc_id not in escalators_by_id:
                escalators_by_id[esc_id] = {}
            escalators_by_id[esc_id][level] = zone_name

        # Create bidirectional transfers for each escalator
        for esc_id, levels_dict in escalators_by_id.items():
            if len(levels_dict) < 2:
                logger.warning(
                    f"Escalator {esc_id} found on only {len(levels_dict)} level(s). "
                    f"Need at least 2 levels for transfer. Zones: {list(levels_dict.values())}"
                )
                continue

            # Create transfers between all level pairs
            level_list = sorted(levels_dict.keys(), key=lambda x: float(x))
            for i in range(len(level_list) - 1):
                from_level = level_list[i]
                to_level = level_list[i + 1]
                from_zone = levels_dict[from_level]
                to_zone = levels_dict[to_level]

                # Bidirectional
                self.transfer_mappings[from_zone] = (to_zone, to_level)
                self.transfer_mappings[to_zone] = (from_zone, from_level)

                logger.info(f"  Created transfer: {from_zone} ↔ {to_zone}")

        logger.info(f"Built {len(self.transfer_mappings)} transfer mappings")

    def check_transfer(
        self,
        agent_id: str,
        position: tuple[float, float],
        current_zone: str,
        current_level: str,
        current_time: float,
    ) -> tuple[str, str, tuple[float, float]] | None:
        """
        Check if agent should transfer to another level.

        Args:
            agent_id: Agent ID
            position: (x, y) position in current level
            current_zone: Current zone name
            current_level: Current level ID
            current_time: Simulation time

        Returns:
            (target_zone, target_level, new_position) if transfer should occur, else None
        """
        # Check cooldown (prevent oscillation)
        cooldown_time = 2.0  # seconds
        if agent_id in self.transfer_history:
            if current_time - self.transfer_history[agent_id] < cooldown_time:
                return None

        # Check if current zone is in transfer mappings
        if current_zone not in self.transfer_mappings:
            return None

        # Get target zone and level
        target_zone, target_level = self.transfer_mappings[current_zone]

        # Get source and target polygons
        source_poly = self.escalator_zones[current_zone]
        target_poly = self.escalator_zones[target_zone]

        # Check if agent is in source polygon
        point = Point(position)
        if not source_poly.contains(point):
            return None

        # Map relative position from source to target polygon
        new_position = self._map_position(source_poly, target_poly, position)

        # Record transfer for cooldown
        self.transfer_history[agent_id] = current_time

        logger.debug(
            f"Transfer {agent_id}: {current_zone} (L{current_level}) → "
            f"{target_zone} (L{target_level}) at pos {position} → {new_position}"
        )

        return target_zone, target_level, new_position

    def _map_position(
        self, source_poly: Polygon, target_poly: Polygon, source_pos: tuple[float, float]
    ) -> tuple[float, float]:
        """
        Map agent position from source polygon to target polygon.

        Uses normalized coordinates within polygon bounding boxes to preserve
        relative position across different geometry sizes.

        Args:
            source_poly: Source polygon
            target_poly: Target polygon
            source_pos: Position within source polygon

        Returns:
            Corresponding position in target polygon
        """
        # Get bounding boxes
        source_bounds = source_poly.bounds  # (minx, miny, maxx, maxy)
        target_bounds = target_poly.bounds

        # Normalize position within source bounds
        sx_min, sy_min, sx_max, sy_max = source_bounds
        tx_min, ty_min, tx_max, ty_max = target_bounds

        # Handle zero-width/height cases
        sx_range = sx_max - sx_min if sx_max > sx_min else 1.0
        sy_range = sy_max - sy_min if sy_max > sy_min else 1.0
        tx_range = tx_max - tx_min if tx_max > tx_min else 1.0
        ty_range = ty_max - ty_min if ty_max > ty_min else 1.0

        # Normalized position (0.0 to 1.0)
        norm_x = (source_pos[0] - sx_min) / sx_range
        norm_y = (source_pos[1] - sy_min) / sy_range

        # Map to target polygon
        target_x = tx_min + norm_x * tx_range
        target_y = ty_min + norm_y * ty_range

        return (target_x, target_y)

    def _parse_shape_string(self, shape_str: str) -> list[tuple[float, float]]:
        """Parse SUMO shape string into coordinate list."""
        coords = []
        for point_str in shape_str.strip().split():
            x, y = point_str.split(",")
            coords.append((float(x), float(y)))
        return coords

    def get_transfer_info(self) -> dict:
        """Get debugging info about transfers."""
        return {
            "escalator_zones_count": len(self.escalator_zones),
            "transfer_mappings_count": len(self.transfer_mappings),
            "zones": list(self.escalator_zones.keys()),
            "transfers": {k: f"{v[0]} (L{v[1]})" for k, v in self.transfer_mappings.items()},
        }
