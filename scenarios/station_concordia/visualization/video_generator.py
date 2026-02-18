"""
Video generator for Station Concordia simulations.

This module creates MP4 videos from simulation output data.
Videos show agent positions and decisions at regular time intervals,
without delays for LLM responses.
"""

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon as MPLPolygon

from scenarios.common.logger import get_logger

logger = get_logger(__name__)

matplotlib.use("Agg")  # Non-interactive backend for video generation


class VideoGenerator:
    """Generates MP4 videos from simulation output data."""

    def __init__(
        self,
        output_file: Path,
        geometry: dict | None = None,
        fps: int = 20,
        speedup: float = 1.0,
    ):
        """
        Initialize video generator.

        Args:
            output_file: Path to agent decisions JSON file
            geometry: Station geometry dict (or None to load from data)
            fps: Frames per second for output video
            speedup: Speed multiplier (1.0 = real-time, 2.0 = 2x speed)
        """
        self.output_file = output_file
        self.geometry = geometry
        self.fps = fps
        self.speedup = speedup

        # Load simulation data
        self.data = self._load_data()
        if not self.data:
            raise ValueError(f"Could not load data from {output_file}")

        # Extract time series data
        self.time_series = self._extract_time_series()
        if not self.time_series:
            raise ValueError("No position data found in output file")

        logger.info(
            f"Loaded {len(self.time_series)} time steps "
            f"from {self.data.get('current_time', 0):.1f}s simulation"
        )

    def _load_data(self) -> dict:
        """Load simulation output data."""
        try:
            with open(self.output_file) as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            return {}

    def _extract_time_series(self) -> list[dict]:
        """
        Extract time series of agent positions and decisions.

        Returns:
            List of dicts with keys: time, positions, decisions, blocked_exits
        """
        time_series = []

        # Check if we have position history (saved separately for video generation)
        if "position_history" in self.data and self.data["position_history"]:
            logger.info(f"Using position history with {len(self.data['position_history'])} frames")
            # Use saved position history - already in correct format
            for frame in self.data["position_history"]:
                time_series.append(
                    {
                        "time": frame["time"],
                        "positions": frame["positions"],
                        "decisions": self.data.get("agent_decisions", {}),
                        "blocked_exits": frame.get("blocked_exits", []),
                        "agent_states": frame.get("agent_states", {}),
                    }
                )
        else:
            # Fallback: use final state only (single frame)
            logger.warning(
                "No position history found - video will show final state only. "
                "Enable video generation during simulation for full animation."
            )
            agent_positions = self.data.get("agent_positions", {})
            agent_decisions = self.data.get("agent_decisions", {})
            blocked_exits = self.data.get("blocked_exits", [])
            final_time = self.data.get("current_time", self.data.get("final_time", 0))

            time_series.append(
                {
                    "time": final_time,
                    "positions": agent_positions,
                    "decisions": agent_decisions,
                    "blocked_exits": blocked_exits,
                    "agent_states": {},
                }
            )

        return time_series

    def _setup_figure(self) -> tuple:
        """
        Setup matplotlib figure and axes.

        Returns:
            (fig, ax_map, ax_decisions)
        """
        fig, (ax_map, ax_decisions) = plt.subplots(
            1, 2, figsize=(16, 8), gridspec_kw={"width_ratios": [2, 1]}
        )

        title_text = fig.suptitle("Concordia Station Evacuation | Time: 0.0s", fontsize=14)

        # Setup map axes
        ax_map.set_title("Agent Positions")
        ax_map.set_xlabel("X Position (m)")
        ax_map.set_ylabel("Y Position (m)")
        ax_map.grid(True, alpha=0.3)
        ax_map.set_aspect("equal")

        # Draw geometry
        if self.geometry:
            self._draw_geometry(ax_map)
            self._set_limits_from_geometry(ax_map)

        # Setup decision axes
        ax_decisions.set_title("Recent Decisions")
        ax_decisions.axis("off")
        ax_decisions.set_xlim(0, 1)
        ax_decisions.set_ylim(0, 1)

        return fig, ax_map, ax_decisions, title_text

    def _draw_geometry(self, ax):
        """Draw station geometry on axes."""
        if not self.geometry:
            return

        # Draw walkable areas
        if "walkable_areas" in self.geometry:
            for _, coords in self.geometry["walkable_areas"].items():
                if coords:
                    polygon = MPLPolygon(coords, fill=True, alpha=0.2, color="gray")
                    ax.add_patch(polygon)

        # Draw entrances/exits
        if "entrance_areas" in self.geometry:
            for _, coords in self.geometry["entrance_areas"].items():
                if coords:
                    polygon = MPLPolygon(coords, fill=True, alpha=0.3, color="green")
                    ax.add_patch(polygon)

        # Draw platforms
        if "platform_areas" in self.geometry:
            for _, coords in self.geometry["platform_areas"].items():
                if coords:
                    polygon = MPLPolygon(coords, fill=True, alpha=0.3, color="blue")
                    ax.add_patch(polygon)

        # Draw obstacles
        if "obstacles" in self.geometry:
            for coords in self.geometry["obstacles"]:
                if coords:
                    polygon = MPLPolygon(coords, fill=True, alpha=0.4, color="black")
                    ax.add_patch(polygon)

    def _set_limits_from_geometry(self, ax):
        """Set axis limits from geometry."""
        coords_list = []
        for key in ("walkable_areas", "entrance_areas", "platform_areas", "obstacles"):
            areas = self.geometry.get(key, {})
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
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)

            pad_x = (x_max - x_min) * 0.05 if x_max > x_min else 5.0
            pad_y = (y_max - y_min) * 0.05 if y_max > y_min else 5.0

            ax.set_xlim(x_min - pad_x, x_max + pad_x)
            ax.set_ylim(y_min - pad_y, y_max + pad_y)

    def _draw_frame(self, ax_map, ax_decisions, frame_data, title_text):
        """
        Draw a single frame of the video.

        Args:
            ax_map: Map axes
            ax_decisions: Decision log axes
            frame_data: Dict with time, positions, decisions, etc.
            title_text: Title text object
        """
        # Clear previous frame (keep geometry)
        for artist in ax_map.get_children():
            if hasattr(artist, "get_label") and artist.get_label() == "_agent":
                artist.remove()

        ax_decisions.clear()
        ax_decisions.axis("off")
        ax_decisions.set_xlim(0, 1)
        ax_decisions.set_ylim(0, 1)

        # Update title
        time_val = frame_data["time"]
        title_text.set_text(f"Concordia Station Evacuation | Time: {time_val:.1f}s")

        # Draw blocked exits
        blocked_exits = frame_data.get("blocked_exits", [])
        if blocked_exits and self.geometry:
            entrance_areas = self.geometry.get("entrance_areas", {})
            for exit_name in blocked_exits:
                if exit_name in entrance_areas:
                    coords = entrance_areas[exit_name]
                    if coords:
                        xs = [c[0] for c in coords]
                        ys = [c[1] for c in coords]
                        center_x = sum(xs) / len(xs)
                        center_y = sum(ys) / len(ys)

                        size = 8
                        ax_map.plot(
                            [center_x - size, center_x + size],
                            [center_y - size, center_y + size],
                            "r-",
                            linewidth=4,
                            label="_agent",
                        )
                        ax_map.plot(
                            [center_x - size, center_x + size],
                            [center_y + size, center_y - size],
                            "r-",
                            linewidth=4,
                            label="_agent",
                        )
                        ax_map.text(
                            center_x,
                            center_y - size - 3,
                            "🚧 BLOCKED",
                            ha="center",
                            fontsize=10,
                            color="red",
                            weight="bold",
                            label="_agent",
                        )

        # Determine agent states
        waiting_agents = {}

        # Use agent_states if available (from position history)
        agent_states = frame_data.get("agent_states", {})
        if agent_states:
            for agent_id, state in agent_states.items():
                action_type = state.get("action_type", "")
                if action_type == "wait":
                    wait_reason = state.get("wait_reason", "unknown")
                    waiting_agents[agent_id] = wait_reason

        # Fallback: extract from decisions if agent_states not available
        if not agent_states:
            decisions = frame_data.get("decisions", {})
            if isinstance(decisions, dict):
                for agent_id, agent_data in decisions.items():
                    if isinstance(agent_data, dict) and "decisions" in agent_data:
                        decisions_list = agent_data["decisions"]
                        if decisions_list:
                            latest = decisions_list[-1]
                            if isinstance(latest, dict):
                                translated = latest.get("translated", {})
                                if isinstance(translated, dict):
                                    action_type = translated.get("action_type", "")
                                    if action_type == "wait":
                                        wait_reason = translated.get("wait_reason", "unknown")
                                        waiting_agents[agent_id] = wait_reason

        # Draw agent positions
        positions = frame_data.get("positions", {})
        for agent_id, pos in positions.items():
            if pos and len(pos) >= 2:
                x, y = pos[0], pos[1]

                # Determine color and size
                if agent_id in waiting_agents:
                    wait_reason = waiting_agents[agent_id]
                    color = {
                        "seeking_information": "purple",
                        "waiting_for_help": "gold",
                        "observing_others": "cyan",
                        "assessing_situation": "magenta",
                    }.get(wait_reason, "gray")
                    size = 9
                else:
                    color = "red"
                    size = 8

                ax_map.plot(x, y, "o", color=color, markersize=size, label="_agent")
                ax_map.text(x, y + 1, agent_id, ha="center", fontsize=8, label="_agent")

        # Add legend
        legend_elements = [
            Line2D(
                [0], [0], marker="o", color="w", markerfacecolor="red", markersize=8, label="Moving"
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="purple",
                markersize=9,
                label="Wait: Seeking Info",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="gold",
                markersize=9,
                label="Wait: For Help",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="cyan",
                markersize=9,
                label="Wait: Observing",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="magenta",
                markersize=9,
                label="Wait: Assessing",
            ),
        ]
        ax_map.legend(handles=legend_elements, loc="upper right", fontsize=8)

        # Draw decision log (only show decisions up to current frame time)
        recent_decisions = []
        decisions = frame_data.get("decisions", {})
        current_time = frame_data["time"]

        if isinstance(decisions, dict):
            for agent_id, agent_data in decisions.items():
                if isinstance(agent_data, dict) and "decisions" in agent_data:
                    decisions_list = agent_data["decisions"]
                    # Filter decisions that occurred before or at current frame time
                    for decision in decisions_list:
                        dec_time = decision.get("time", decision.get("timestamp", 0))
                        if dec_time <= current_time:
                            recent_decisions.append((agent_id, decision))

        # Sort by timestamp and get most recent
        recent_decisions.sort(
            key=lambda x: x[1].get("time", x[1].get("timestamp", 0)), reverse=True
        )
        recent_decisions = recent_decisions[:5]

        y_pos = 0.95
        for agent_id, decision in recent_decisions:
            dec_time = decision.get("time", decision.get("timestamp", 0))
            action = decision.get("action", "Unknown")

            ax_decisions.text(
                0.05,
                y_pos,
                f"[{dec_time:.1f}s] {agent_id}:\n  → {action[:50]}...",
                fontsize=9,
                verticalalignment="top",
                family="monospace",
            )
            y_pos -= 0.18

    def generate(self, output_path: Path, dpi: int = 100) -> bool:
        """
        Generate video file.

        Args:
            output_path: Path for output MP4 file
            dpi: Resolution (dots per inch)

        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Generating video: {output_path}")
        logger.info(f"Video settings: {self.fps} fps, {self.speedup}x speed, {dpi} dpi")

        try:
            # Setup figure
            fig, ax_map, ax_decisions, title_text = self._setup_figure()

            # Setup video writer
            writer = FFMpegWriter(fps=self.fps, metadata={"artist": "NewcastleSim"})

            with writer.saving(fig, str(output_path), dpi=dpi):
                # For now, we only have one frame (final state)
                # In a proper implementation, we'd iterate through time series
                for frame_data in self.time_series:
                    self._draw_frame(ax_map, ax_decisions, frame_data, title_text)
                    writer.grab_frame()

            plt.close(fig)
            logger.info(f"Video saved: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to generate video: {e}", exc_info=True)
            return False


def generate_video_from_output(
    output_file: Path,
    video_path: Path | None = None,
    geometry: dict | None = None,
    fps: int = 20,
    speedup: float = 1.0,
    dpi: int = 100,
) -> bool:
    """
    Generate video from simulation output file.

    Args:
        output_file: Path to agent decisions JSON file
        video_path: Output video path (default: same dir as output_file)
        geometry: Station geometry dict
        fps: Frames per second
        speedup: Speed multiplier
        dpi: Resolution

    Returns:
        True if successful
    """
    if video_path is None:
        video_path = output_file.parent / f"{output_file.stem}_video.mp4"

    generator = VideoGenerator(output_file, geometry, fps, speedup)
    return generator.generate(video_path, dpi)
