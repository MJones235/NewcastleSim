#!/usr/bin/env python3
"""
Spatial viewer for Station Concordia simulation.

Shows agent positions on station map in real-time alongside their decisions.
Requires matplotlib for visualization.

Usage:
    python tools/view_concordia_spatial.py --output-file PATH --geometry PATH
    python tools/view_concordia_spatial.py --output-file PATH --network-path PATH
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Setup path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.patches import Polygon as MPLPolygon

    matplotlib.use("TkAgg")
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Install with: pip install matplotlib")


class SpatialConcordiaViewer:
    """Real-time spatial viewer for Concordia simulation."""

    def __init__(
        self,
        output_file: Path,
        geometry_file: Path | None = None,
        network_path: Path | None = None,
    ):
        """Initialize spatial viewer."""
        self.output_file = output_file
        self.geometry_file = geometry_file
        self.network_path = network_path
        self.agent_positions = {}
        self.agent_decisions = {}
        self.last_update = 0
        self.blocked_exits = []  # Phase 4.2: Track blocked exits for visualization

        # Load geometry if provided
        self.geometry = None
        if geometry_file and geometry_file.exists():
            self.geometry = self._load_geometry(geometry_file)
        elif self.network_path and self.network_path.exists():
            self.geometry = self._load_geometry_from_network(self.network_path)

        # Setup matplotlib figure
        self.fig, (self.ax_map, self.ax_decisions) = plt.subplots(
            1, 2, figsize=(16, 8), gridspec_kw={"width_ratios": [2, 1]}
        )

        self.title_text = self.fig.suptitle(
            "Concordia Station Evacuation - Real-Time View | Time: 0.0s", fontsize=14
        )
        self.current_time = 0.0

        self._setup_map_axes()
        self._setup_decision_axes()

        # Agent visual elements
        self.agent_dots = {}
        self.agent_labels = {}
        self.decision_texts = []
        self.blocked_exit_markers = []  # Phase 4.2: Visual markers for blocked exits

    def _load_geometry(self, geometry_file: Path) -> dict:
        """Load station geometry from file."""
        try:
            with open(geometry_file) as f:
                return json.load(f)
        except Exception as e:
            print(f"Failed to load geometry: {e}")
            return None

    def _load_geometry_from_network(self, network_path: Path) -> dict | None:
        """Load station geometry from SUMO walking_areas.add.xml files."""
        try:
            from scenarios.station_jupedsim.geometry import (
                load_entrance_areas,
                load_obstacles,
                load_platform_areas,
                load_walkable_areas,
            )

            walking_areas_file = network_path / "walking_areas.add.xml"
            if not walking_areas_file.exists():
                print(f"Geometry file not found: {walking_areas_file}")
                return None

            walkable_areas = load_walkable_areas(str(walking_areas_file))
            entrance_areas = load_entrance_areas(str(walking_areas_file))
            platform_areas = load_platform_areas(str(walking_areas_file))
            obstacles = load_obstacles(str(walking_areas_file))

            def poly_to_coords(poly):
                return list(poly.exterior.coords)

            geometry = {
                "walkable_areas": {
                    name: poly_to_coords(poly) for name, poly in walkable_areas.items()
                },
                "entrance_areas": {
                    name: poly_to_coords(poly) for name, poly in entrance_areas.items()
                },
                "platform_areas": {
                    name: poly_to_coords(poly) for name, poly in platform_areas.items()
                },
                "obstacles": [poly_to_coords(poly) for poly in obstacles],
            }

            print(
                f"Loaded geometry from {walking_areas_file}: "
                f"{len(walkable_areas)} walkable, "
                f"{len(entrance_areas)} entrances, "
                f"{len(platform_areas)} platforms, "
                f"{len(obstacles)} obstacles"
            )

            return geometry
        except Exception as e:
            print(f"Failed to load geometry from network: {e}")
            return None

    def _setup_map_axes(self):
        """Setup the map visualization axes."""
        self.ax_map.set_title("Agent Positions")
        self.ax_map.set_xlabel("X Position (m)")
        self.ax_map.set_ylabel("Y Position (m)")
        self.ax_map.grid(True, alpha=0.3)
        self.ax_map.set_aspect("equal")

        # Draw geometry if available
        if self.geometry:
            self._draw_geometry()
            self._set_fixed_limits_from_geometry()
        else:
            # Default to simple 100x100 room if no geometry provided
            self._set_fixed_limits(0.0, 100.0, 0.0, 100.0)

    def _setup_decision_axes(self):
        """Setup the decision log axes."""
        self.ax_decisions.set_title("Recent Decisions")
        self.ax_decisions.axis("off")
        self.ax_decisions.set_xlim(0, 1)
        self.ax_decisions.set_ylim(0, 1)

    def _draw_geometry(self):
        """Draw station geometry on map."""
        if not self.geometry:
            return

        # Draw walkable areas
        if "walkable_areas" in self.geometry:
            for _, coords in self.geometry["walkable_areas"].items():
                if coords:
                    polygon = MPLPolygon(
                        coords, fill=True, alpha=0.2, color="gray", label="Walkable"
                    )
                    self.ax_map.add_patch(polygon)

        # Draw entrances/exits
        if "entrance_areas" in self.geometry:
            for _, coords in self.geometry["entrance_areas"].items():
                if coords:
                    polygon = MPLPolygon(coords, fill=True, alpha=0.3, color="green", label="Exit")
                    self.ax_map.add_patch(polygon)

        # Draw platforms
        if "platform_areas" in self.geometry:
            for _, coords in self.geometry["platform_areas"].items():
                if coords:
                    polygon = MPLPolygon(
                        coords, fill=True, alpha=0.3, color="blue", label="Platform"
                    )
                    self.ax_map.add_patch(polygon)

        # Draw obstacles
        if "obstacles" in self.geometry:
            for coords in self.geometry["obstacles"]:
                if coords:
                    polygon = MPLPolygon(
                        coords, fill=True, alpha=0.4, color="black", label="Obstacle"
                    )
                    self.ax_map.add_patch(polygon)

    def _set_fixed_limits(self, x_min: float, x_max: float, y_min: float, y_max: float):
        """Set fixed axis limits with small padding."""
        pad_x = (x_max - x_min) * 0.05 if x_max > x_min else 5.0
        pad_y = (y_max - y_min) * 0.05 if y_max > y_min else 5.0
        self.ax_map.set_xlim(x_min - pad_x, x_max + pad_x)
        self.ax_map.set_ylim(y_min - pad_y, y_max + pad_y)

    def _set_fixed_limits_from_geometry(self):
        """Compute geometry bounds and lock axis limits."""
        coords_list = []
        for key in ("walkable_areas", "entrance_areas", "platform_areas", "obstacles"):
            areas = self.geometry.get(key, {}) if self.geometry else {}
            if isinstance(areas, dict):
                for coords in areas.values():
                    if coords:
                        coords_list.extend(coords)
            elif isinstance(areas, list):
                for coords in areas:
                    if coords:
                        coords_list.extend(coords)

        if coords_list:
            xs = [c[0] for c in coords_list]
            ys = [c[1] for c in coords_list]
            self._set_fixed_limits(min(xs), max(xs), min(ys), max(ys))

    def _update_data(self):
        """Load latest data from output file."""
        if not self.output_file.exists():
            return False

        try:
            with open(self.output_file) as f:
                data = json.load(f)

            # Update agent positions
            if "agent_positions" in data:
                self.agent_positions = data["agent_positions"]

            # Update agent decisions
            if "agent_decisions" in data:
                self.agent_decisions = data["agent_decisions"]

            # Get current simulation time (use current_time from incremental saves, or final_time from final save)
            if "current_time" in data:
                self.current_time = data["current_time"]
            elif "final_time" in data:
                self.current_time = data["final_time"]

            # Update title with time
            self.title_text.set_text(
                f"Concordia Station Evacuation - Real-Time View | Time: {self.current_time:.1f}s"
            )

            # Phase 4.2: Update blocked exits list
            if "blocked_exits" in data:
                self.blocked_exits = data["blocked_exits"]

            self.last_update = time.time()
            return True

        except (OSError, json.JSONDecodeError):
            return False

    def _update_visualization(self, frame):
        """Update visualization with latest data."""
        # Load new data
        if not self._update_data():
            return

        # Update agent positions on map
        self._update_agent_positions()

        # Update blocked exit markers (Phase 4.2)
        self._update_blocked_exits()

        # Update decision log
        self._update_decision_log()

    def _update_blocked_exits(self):
        """Draw visual markers for blocked exits (Phase 4.2)."""
        # Remove old markers
        for marker in self.blocked_exit_markers:
            marker.remove()
        self.blocked_exit_markers = []

        if not self.blocked_exits or not self.geometry:
            return

        # Get entrance areas from geometry
        entrance_areas = self.geometry.get("entrance_areas", {})

        for exit_name in self.blocked_exits:
            if exit_name in entrance_areas:
                # Get exit polygon coordinates
                coords = entrance_areas[exit_name]
                if coords:
                    # Calculate centroid
                    xs = [c[0] for c in coords]
                    ys = [c[1] for c in coords]
                    center_x = sum(xs) / len(xs)
                    center_y = sum(ys) / len(ys)

                    # Draw red X over the exit
                    size = 8
                    marker1 = self.ax_map.plot(
                        [center_x - size, center_x + size],
                        [center_y - size, center_y + size],
                        "r-",
                        linewidth=4,
                    )[0]
                    marker2 = self.ax_map.plot(
                        [center_x - size, center_x + size],
                        [center_y + size, center_y - size],
                        "r-",
                        linewidth=4,
                    )[0]
                    label = self.ax_map.text(
                        center_x,
                        center_y - size - 3,
                        "🚧 BLOCKED",
                        ha="center",
                        fontsize=10,
                        color="red",
                        weight="bold",
                    )

                    self.blocked_exit_markers.extend([marker1, marker2, label])

    def _update_agent_positions(self):
        """Update agent position markers."""
        # Remove old dots
        for dot in self.agent_dots.values():
            dot.remove()
        for label in self.agent_labels.values():
            label.remove()

        self.agent_dots = {}
        self.agent_labels = {}

        # Draw new positions
        if not self.agent_positions:
            return  # No positions yet

        for agent_id, pos in self.agent_positions.items():
            if pos and len(pos) >= 2:
                x, y = pos[0], pos[1]  # Handle both list and tuple from JSON
                dot = self.ax_map.plot(x, y, "ro", markersize=8)[0]
                label = self.ax_map.text(x, y + 1, agent_id, ha="center", fontsize=8)

                self.agent_dots[agent_id] = dot
                self.agent_labels[agent_id] = label

        # Keep fixed axis limits (do not autoscale)

    def _update_decision_log(self):
        """Update the decision log display."""
        # Clear old text
        for text in self.decision_texts:
            text.remove()
        self.decision_texts = []

        # Get recent decisions (last 5)
        recent_decisions = []
        if isinstance(self.agent_decisions, dict):
            for agent_id, agent_data in self.agent_decisions.items():
                # agent_data is a dict with keys like "decisions", "observations"
                if isinstance(agent_data, dict) and "decisions" in agent_data:
                    # decisions is a list of decision dicts
                    decisions_list = agent_data["decisions"]
                    for decision in decisions_list[-3:]:  # Last 3 per agent
                        recent_decisions.append((agent_id, decision))
                elif isinstance(agent_data, list):
                    # Legacy format: agent_data is directly a list of decisions
                    for decision in agent_data[-3:]:
                        recent_decisions.append((agent_id, decision))

        # Sort by timestamp (using "time" key from hybrid_simulation.py)
        recent_decisions.sort(
            key=lambda x: x[1].get("time", x[1].get("timestamp", 0)), reverse=True
        )
        recent_decisions = recent_decisions[:5]  # Top 5 most recent

        # Display decisions
        y_pos = 0.95
        for agent_id, decision in recent_decisions:
            time_val = decision.get("time", decision.get("timestamp", 0))
            action = decision.get("action", "Unknown")

            text = self.ax_decisions.text(
                0.05,
                y_pos,
                f"[{time_val:.1f}s] {agent_id}:\n  → {action[:50]}...",
                fontsize=9,
                verticalalignment="top",
                family="monospace",
            )
            self.decision_texts.append(text)
            y_pos -= 0.18

    def run(self):
        """Run the viewer with animation."""
        if not MATPLOTLIB_AVAILABLE:
            print("matplotlib not available - cannot run spatial viewer")
            return

        print(f"Monitoring: {self.output_file}")
        print("Close window to exit...")

        # Setup animation - update every 500ms to match file write frequency
        self.ani = FuncAnimation(
            self.fig, self._update_visualization, interval=500, cache_frame_data=False
        )

        plt.tight_layout()
        plt.show()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Spatial viewer for Concordia simulation")
    parser.add_argument(
        "--output-file", type=str, required=True, help="Path to agent decisions JSON file"
    )
    parser.add_argument(
        "--geometry", type=str, default=None, help="Path to geometry JSON file (optional)"
    )
    parser.add_argument(
        "--network-path",
        type=str,
        default=None,
        help="Path to station_sim network directory (optional)",
    )
    args = parser.parse_args()

    output_file = Path(args.output_file)
    geometry_file = Path(args.geometry) if args.geometry else None
    network_path = Path(args.network_path) if args.network_path else None

    viewer = SpatialConcordiaViewer(output_file, geometry_file, network_path)
    viewer.run()


if __name__ == "__main__":
    main()
