"""
Common visualization functions shared between live viewer and post-run visualization.
"""

from typing import Dict, List
from shapely.geometry import Polygon


def draw_geometry(ax, walkable_areas: Dict[str, Polygon], obstacles: List[Polygon], platform_areas: Dict[str, Polygon]):
    """
    Draw station geometry on matplotlib axes.
    
    Args:
        ax: Matplotlib axes object
        walkable_areas: Dictionary of walkable area name -> Polygon
        obstacles: List of obstacle Polygons
        platform_areas: Dictionary of platform name -> Polygon
    """
    # Draw walkable areas (light blue)
    for name, polygon in walkable_areas.items():
        x, y = polygon.exterior.xy
        ax.fill(x, y, alpha=0.2, fc='lightblue', ec='blue', linewidth=1, 
                label='Walkable' if name == list(walkable_areas.keys())[0] else '')
    
    # Draw obstacles (red)
    for i, polygon in enumerate(obstacles):
        x, y = polygon.exterior.xy
        ax.fill(x, y, alpha=0.4, fc='lightcoral', ec='red', linewidth=1, 
                label='Obstacle' if i == 0 else '')
    
    # Draw platforms (yellow)
    for name, polygon in platform_areas.items():
        x, y = polygon.exterior.xy
        ax.fill(x, y, alpha=0.3, fc='yellow', ec='orange', linewidth=1.5, 
                label='Platform' if name == list(platform_areas.keys())[0] else '')
    
    # Add legend (only unique labels)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right')


def set_axis_limits(ax, walkable_areas: Dict[str, Polygon], obstacles: List[Polygon], platform_areas: Dict[str, Polygon]):
    """
    Set axis limits based on geometry with padding.
    
    Args:
        ax: Matplotlib axes object
        walkable_areas: Dictionary of walkable area name -> Polygon
        obstacles: List of obstacle Polygons
        platform_areas: Dictionary of platform name -> Polygon
    """
    all_polygons = list(walkable_areas.values()) + obstacles + list(platform_areas.values())
    
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
    
    ax.set_xlim(min_x - padding_x, max_x + padding_x)
    ax.set_ylim(min_y - padding_y, max_y + padding_y)


if __name__ == "__main__":
    print("Viewer common module ready")
