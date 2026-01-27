"""
Real-time visualization for JuPedSim simulation.

Displays agent positions, walkable areas, obstacles, and platforms during simulation.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
from shapely.geometry import Polygon
from typing import Dict, List, Optional
import numpy as np

from .viewer_common import draw_geometry, set_axis_limits


class LiveViewer:
    """Real-time visualization of JuPedSim simulation."""
    
    def __init__(
        self, 
        walkable_areas: Dict[str, Polygon],
        obstacles: List[Polygon],
        platform_areas: Dict[str, Polygon],
        update_interval: float = 1.0
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
        self.ax.set_aspect('equal')
        self.ax.set_title('JuPedSim Station Simulation - Live View')
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        
        # Agent scatter plot (will be updated)
        self.agent_scatter = None
        
        # Setup geometry using shared function
        draw_geometry(self.ax, self.walkable_areas, self.obstacles, self.platform_areas)
        
        # Set axis limits with some padding using shared function
        set_axis_limits(self.ax, self.walkable_areas, self.obstacles, self.platform_areas)
        
        # Interactive mode
        plt.ion()
        plt.show(block=False)
    
    def update(self, agent_positions: List[tuple], sim_time: float, agent_count: int):
        """
        Update the visualization with current agent positions.
        
        Args:
            agent_positions: List of (x, y) tuples for agent positions
            sim_time: Current simulation time
            agent_count: Number of active agents
        """
        # Remove old scatter plot
        if self.agent_scatter:
            self.agent_scatter.remove()
        
        # Draw agents as scatter points
        if agent_positions:
            x_coords = [pos[0] for pos in agent_positions]
            y_coords = [pos[1] for pos in agent_positions]
            self.agent_scatter = self.ax.scatter(x_coords, y_coords, c='green', s=30, alpha=0.7, zorder=10)
        
        # Update title with current time and agent count
        self.ax.set_title(f'JuPedSim Station Simulation - t={sim_time:.1f}s, Agents={agent_count}')
        
        # Redraw
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    def close(self):
        """Close the visualization window."""
        plt.close(self.fig)


if __name__ == "__main__":
    print("Live viewer module ready")
