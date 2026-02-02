#!/usr/bin/env python3
"""
Spatial viewer for Station Concordia simulation.

Shows agent positions on station map in real-time alongside their decisions.
Requires matplotlib for visualization.

Usage:
    python tools/view_concordia_spatial.py --output-file PATH --geometry PATH
"""

import argparse
import json
import time
from pathlib import Path

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

    def __init__(self, output_file: Path, geometry_file: Path | None = None):
        """Initialize spatial viewer."""
        self.output_file = output_file
        self.geometry_file = geometry_file
        self.agent_positions = {}
        self.agent_decisions = {}
        self.last_update = 0

        # Load geometry if provided
        self.geometry = None
        if geometry_file and geometry_file.exists():
            self.geometry = self._load_geometry(geometry_file)

        # Setup matplotlib figure
        self.fig, (self.ax_map, self.ax_decisions) = plt.subplots(
            1, 2, figsize=(16, 8), gridspec_kw={"width_ratios": [2, 1]}
        )

        self.fig.suptitle("Concordia Station Evacuation - Real-Time View", fontsize=14)

        self._setup_map_axes()
        self._setup_decision_axes()

        # Agent visual elements
        self.agent_dots = {}
        self.agent_labels = {}
        self.decision_texts = []

    def _load_geometry(self, geometry_file: Path) -> dict:
        """Load station geometry from file."""
        try:
            with open(geometry_file) as f:
                return json.load(f)
        except Exception as e:
            print(f"Failed to load geometry: {e}")
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

    def _set_fixed_limits(self, x_min: float, x_max: float, y_min: float, y_max: float):
        """Set fixed axis limits with small padding."""
        pad_x = (x_max - x_min) * 0.05 if x_max > x_min else 5.0
        pad_y = (y_max - y_min) * 0.05 if y_max > y_min else 5.0
        self.ax_map.set_xlim(x_min - pad_x, x_max + pad_x)
        self.ax_map.set_ylim(y_min - pad_y, y_max + pad_y)

    def _set_fixed_limits_from_geometry(self):
        """Compute geometry bounds and lock axis limits."""
        coords_list = []
        for key in ("walkable_areas", "entrance_areas", "platform_areas"):
            areas = self.geometry.get(key, {}) if self.geometry else {}
            for coords in areas.values():
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

        # Update decision log
        self._update_decision_log()

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
    args = parser.parse_args()

    output_file = Path(args.output_file)
    geometry_file = Path(args.geometry) if args.geometry else None

    viewer = SpatialConcordiaViewer(output_file, geometry_file)
    viewer.run()


if __name__ == "__main__":
    main()
