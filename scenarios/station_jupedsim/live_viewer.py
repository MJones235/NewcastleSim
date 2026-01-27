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
        
        # Setup geometry
        self._draw_geometry()
        
        # Set axis limits with some padding
        self._set_axis_limits()
        
        # Interactive mode
        plt.ion()
        plt.show(block=False)
        
    def _draw_geometry(self):
        """Draw walkable areas, obstacles, and platforms."""
        # Draw walkable areas (light blue)
        for name, polygon in self.walkable_areas.items():
            x, y = polygon.exterior.xy
            self.ax.fill(x, y, alpha=0.2, fc='lightblue', ec='blue', linewidth=1, label='Walkable' if name == list(self.walkable_areas.keys())[0] else '')
            
        # Draw obstacles (red) - obstacles is a list
        for i, polygon in enumerate(self.obstacles):
            x, y = polygon.exterior.xy
            self.ax.fill(x, y, alpha=0.4, fc='lightcoral', ec='red', linewidth=1, label='Obstacle' if i == 0 else '')
        
        # Draw platforms (yellow)
        for name, polygon in self.platform_areas.items():
            x, y = polygon.exterior.xy
            self.ax.fill(x, y, alpha=0.3, fc='yellow', ec='orange', linewidth=1.5, label='Platform' if name == list(self.platform_areas.keys())[0] else '')
        
        # Add legend (only unique labels)
        handles, labels = self.ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        self.ax.legend(by_label.values(), by_label.keys(), loc='upper right')
    
    def _set_axis_limits(self):
        """Set axis limits based on geometry."""
        all_polygons = list(self.walkable_areas.values()) + self.obstacles + list(self.platform_areas.values())
        
        if not all_polygons:
            return
        
        # Find bounds
        min_x = min_y = float('inf')
        max_x = max_y = float('-inf')
        
        for polygon in all_polygons:
            bounds = polygon.bounds  # (minx, miny, maxx, maxy)
            min_x = min(min_x, bounds[0])
            min_y = min(min_y, bounds[1])
            max_x = max(max_x, bounds[2])
            max_y = max(max_y, bounds[3])
        
        # Add 10% padding
        padding_x = (max_x - min_x) * 0.1
        padding_y = (max_y - min_y) * 0.1
        
        self.ax.set_xlim(min_x - padding_x, max_x + padding_x)
        self.ax.set_ylim(min_y - padding_y, max_y + padding_y)
    
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
