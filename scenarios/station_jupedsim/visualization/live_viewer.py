"""
Real-time visualization for JuPedSim simulation.

Displays agent positions, walkable areas, obstacles, and platforms during simulation.
"""

from typing import Any

import matplotlib.pyplot as plt
from shapely.geometry import Polygon

from .viewer_common import draw_geometry, set_axis_limits


class LiveViewer:
    """Real-time visualization of JuPedSim simulation."""

    def __init__(
        self,
        walkable_areas: dict[str, Polygon],
        obstacles: list[Polygon],
        platform_areas: dict[str, Polygon],
        update_interval: float = 1.0,
    ):
        """
        Initialize the live viewer.

        Args:
            walkable_areas: Dictionary of walkable area name -> Polygon
            obstacles: List of obstacle Polygons
            platform_areas: Dictionary of platform name -> Polygon
            update_interval: Time between visual updates in seconds (default 1.0s)
        """
        self.walkable_areas = walkable_areas
        self.obstacles = obstacles
        self.platform_areas = platform_areas
        self.update_interval = update_interval

        # Create figure and axis
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        self.ax.set_aspect("equal")
        self.ax.set_title("JuPedSim Station Simulation - Live View")
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")

        # Agent scatter plot (will be updated)
        self.agent_scatter: Any = None

        # Event message popup (text annotation)
        self.event_popup: Any = None
        self.event_popup_time: float | None = None
        self.event_popup_duration = 5.0  # Show popup for 5 seconds

        # Setup geometry using shared function
        draw_geometry(self.ax, self.walkable_areas, self.obstacles, self.platform_areas)

        # Set axis limits with some padding using shared function
        set_axis_limits(self.ax, self.walkable_areas, self.obstacles, self.platform_areas)

        # Interactive mode
        plt.ion()
        plt.show(block=False)

    def update(
        self,
        agent_positions: list[tuple],
        sim_time: float,
        agent_count: int,
        event_message: str | None = None,
    ):
        """
        Update the visualization with current agent positions.

        Args:
            agent_positions: List of (x, y) tuples for agent positions
            sim_time: Current simulation time
            agent_count: Number of active agents
            event_message: Optional event message to display as popup
        """
        # Remove old scatter plot
        if self.agent_scatter:
            self.agent_scatter.remove()

        # Draw agents as scatter points
        if agent_positions:
            x_coords = [pos[0] for pos in agent_positions]
            y_coords = [pos[1] for pos in agent_positions]
            self.agent_scatter = self.ax.scatter(
                x_coords, y_coords, c="green", s=30, alpha=0.7, zorder=10
            )

        # Handle event popup
        if event_message:
            # New event message - create popup
            self.show_event_popup(event_message, sim_time)
        elif self.event_popup and self.event_popup_time:
            # Check if popup should be removed
            if sim_time - self.event_popup_time > self.event_popup_duration:
                self.remove_event_popup()

        # Update title with current time and agent count
        self.ax.set_title(f"JuPedSim Station Simulation - t={sim_time:.1f}s, Agents={agent_count}")

        # Redraw
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def show_event_popup(self, message: str, sim_time: float):
        """
        Display an event message popup.

        Args:
            message: Event message to display
            sim_time: Current simulation time
        """
        # Remove existing popup if any
        self.remove_event_popup()

        # Get axis limits for positioning
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        x_center = (xlim[0] + xlim[1]) / 2
        y_top = ylim[1] - (ylim[1] - ylim[0]) * 0.05  # 5% from top

        # Create text box with message
        self.event_popup = self.ax.text(
            x_center,
            y_top,
            f"🔔 EVENT: {message}",
            fontsize=12,
            fontweight="bold",
            color="red",
            bbox={
                "boxstyle": "round,pad=0.8",
                "facecolor": "yellow",
                "edgecolor": "red",
                "linewidth": 2,
                "alpha": 0.9,
            },
            ha="center",
            va="top",
            zorder=100,
            wrap=True,
        )
        self.event_popup_time = sim_time

    def remove_event_popup(self):
        """Remove the event popup from display."""
        if self.event_popup:
            self.event_popup.remove()
            self.event_popup = None
            self.event_popup_time = None

        # Redraw
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def close(self):
        """Close the visualization window."""
        plt.close(self.fig)


if __name__ == "__main__":
    print("Live viewer module ready")
