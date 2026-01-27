"""
Visualization tools for JuPedSim station simulation.
Uses the same style as the live viewer for consistency.
"""

import json
import sqlite3
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from scenarios.station_jupedsim.geometry import (
    load_obstacles,
    load_platform_areas,
    load_walkable_areas,
)
from scenarios.station_jupedsim.visualization.viewer_common import set_axis_limits


def load_trajectory_data(db_path):
    """
    Load trajectory data from SQLite database.

    Args:
        db_path: Path to trajectory database

    Returns:
        Dictionary with frame numbers as keys, each containing agent positions
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get all trajectory data
    cursor.execute(
        """
        SELECT frame, id, pos_x, pos_y
        FROM trajectory_data
        ORDER BY frame, id
    """
    )

    data = cursor.fetchall()
    conn.close()

    if not data:
        return None

    # Organize data by frame
    frames: dict[int, list] = {}
    for frame, _agent_id, x, y in data:
        if frame not in frames:
            frames[frame] = []
        frames[frame].append((x, y))

    return frames


def visualize_simulation(trajectory_db: str, network_path: str):
    """
    Create animated visualization of the simulation using live viewer style.

    Args:
        trajectory_db: Path to trajectory SQLite database
        network_path: Path to network directory with walking_areas.add.xml
    """
    # Load geometry
    walking_areas_file = Path(network_path) / "walking_areas.add.xml"
    walkable_areas = load_walkable_areas(str(walking_areas_file))
    obstacles = load_obstacles(str(walking_areas_file))
    platform_areas = load_platform_areas(str(walking_areas_file))

    # Load trajectory data
    print("Loading trajectory data...")
    frames_data = load_trajectory_data(trajectory_db)

    if not frames_data:
        print("No trajectory data found!")
        return

    print(f"Loaded {len(frames_data)} frames")

    # Load triggered events if available
    events_file = Path(trajectory_db).parent / "triggered_events.json"
    events_data = []
    if events_file.exists():
        with open(events_file) as f:
            events_data = json.load(f)
        print(f"Loaded {len(events_data)} events")

    # Setup figure using live viewer style
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")

    # Draw geometry - same style as live viewer
    # Draw walkable areas (light blue)
    for name, polygon in walkable_areas.items():
        x, y = polygon.exterior.xy
        ax.fill(
            x,
            y,
            alpha=0.2,
            fc="lightblue",
            ec="blue",
            linewidth=1,
            label="Walkable" if name == list(walkable_areas.keys())[0] else "",
        )

    # Draw obstacles (red)
    for i, polygon in enumerate(obstacles):
        x, y = polygon.exterior.xy
        ax.fill(
            x,
            y,
            alpha=0.4,
            fc="lightcoral",
            ec="red",
            linewidth=1,
            label="Obstacle" if i == 0 else "",
        )

    # Draw platforms (yellow)
    for name, polygon in platform_areas.items():
        x, y = polygon.exterior.xy
        ax.fill(
            x,
            y,
            alpha=0.3,
            fc="yellow",
            ec="orange",
            linewidth=1.5,
            label="Platform" if name == list(platform_areas.keys())[0] else "",
        )

    # Add legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles, strict=False))
    ax.legend(by_label.values(), by_label.keys(), loc="upper right")

    # Set axis limits using shared function
    set_axis_limits(ax, walkable_areas, obstacles, platform_areas)

    # Initialize agent scatter plot
    scatter = ax.scatter([], [], c="green", s=30, alpha=0.7, zorder=10)

    # Event popup tracking
    event_popup = None

    frame_list = sorted(frames_data.keys())
    dt = 0.05  # JuPedSim timestep
    frame_interval = 4  # Trajectory written every 4th frame
    time_per_saved_frame = dt * frame_interval  # 0.2s per saved frame
    event_popup_duration = 5.0  # Show popup for 5 seconds

    def init():
        scatter.set_offsets(np.empty((0, 2)))
        ax.set_title("JuPedSim Station Simulation - Frame 0, t=0.00s, Agents=0")
        return (scatter,)

    def update(frame_idx):
        nonlocal event_popup

        frame_num = frame_list[frame_idx]
        positions = frames_data[frame_num]

        if positions:
            scatter.set_offsets(positions)
        else:
            scatter.set_offsets(np.empty((0, 2)))

        # Calculate actual simulation time based on frame number and sampling interval
        time_sec = frame_num * time_per_saved_frame

        # Handle event popups (remove previous popup if it exists)
        nonlocal event_popup
        if event_popup:
            event_popup.remove()
            event_popup = None

        # Check if any events should be displayed
        for i in range(len(events_data)):
            event = events_data[i]
            event_time = event["time"]

            # Show popup if event time is within display window
            if event_time <= time_sec < event_time + event_popup_duration:
                xlim = ax.get_xlim()
                ylim = ax.get_ylim()
                x_center = (xlim[0] + xlim[1]) / 2
                y_top = ylim[1] - (ylim[1] - ylim[0]) * 0.05

                event_popup = ax.text(
                    x_center,
                    y_top,
                    f"🔔 EVENT: {event['value']}",
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
                )
                break

        ax.set_title(f"JuPedSim Station Simulation - t={time_sec:.1f}s, Agents={len(positions)}")

        # Return all artists that were modified
        artists: list[Any] = [scatter]
        if event_popup:
            artists.append(event_popup)
        return artists

    # Create animation
    print("Creating animation...")
    anim = FuncAnimation(  # noqa: F841  # Must keep reference to prevent garbage collection
        fig, update, init_func=init, frames=len(frame_list), interval=50, blit=False, repeat=True
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python visualize.py <trajectory_db>")
        print("Example: python visualize.py output/trajectory.db")
        sys.exit(1)

    trajectory_db = sys.argv[1]
    # Network is in station_sim, not station_jupedsim
    network_path = Path(__file__).parent.parent / "station_sim" / "network"

    print("Creating visualization...")
    visualize_simulation(trajectory_db, str(network_path))
