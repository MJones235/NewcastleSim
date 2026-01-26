"""
Visualization tools for JuPedSim station simulation.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
import sqlite3
import numpy as np
from pathlib import Path
from geometry_loader import load_walkable_areas, load_obstacles


def plot_geometry(ax, walkable_areas, obstacles=None):
    """
    Plot the walkable areas and obstacles.
    
    Args:
        ax: Matplotlib axis
        walkable_areas: Dict of zone name -> Polygon
        obstacles: List of obstacle Polygons (optional)
    """
    colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral']
    
    for idx, (zone_name, polygon) in enumerate(walkable_areas.items()):
        x, y = polygon.exterior.xy
        color = colors[idx % len(colors)]
        ax.fill(x, y, alpha=0.3, fc=color, ec='black', linewidth=1.5)
        
        # Add zone label at centroid
        centroid = polygon.centroid
        ax.text(centroid.x, centroid.y, zone_name, 
                ha='center', va='center', fontsize=8, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    if obstacles:
        for obstacle in obstacles:
            x, y = obstacle.exterior.xy
            ax.fill(x, y, alpha=0.5, fc='red', ec='darkred', linewidth=1)
    
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')


def load_trajectory_data(db_path):
    """
    Load trajectory data from SQLite database.
    
    Args:
        db_path: Path to trajectory database
        
    Returns:
        Dictionary with 'frames', 'agent_ids', 'positions'
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get all trajectory data
    cursor.execute("""
        SELECT frame, id, pos_x, pos_y 
        FROM trajectory_data 
        ORDER BY frame, id
    """)
    
    data = cursor.fetchall()
    conn.close()
    
    if not data:
        return None
    
    # Organize data by frame
    frames = {}
    for frame, agent_id, x, y in data:
        if frame not in frames:
            frames[frame] = {'ids': [], 'positions': []}
        frames[frame]['ids'].append(agent_id)
        frames[frame]['positions'].append((x, y))
    
    return frames


def visualize_simulation(trajectory_db: str, network_path: str):
    """
    Create animated visualization of the simulation.
    
    Args:
        trajectory_db: Path to trajectory SQLite database
        network_path: Path to network directory with walking_areas.add.xml
    """
    # Load geometry
    walking_areas_file = Path(network_path) / "walking_areas.add.xml"
    walkable_areas = load_walkable_areas(str(walking_areas_file))
    obstacles = load_obstacles(str(walking_areas_file))
    
    # Load trajectory data
    print("Loading trajectory data...")
    frames_data = load_trajectory_data(trajectory_db)
    
    if not frames_data:
        print("No trajectory data found!")
        return
    
    print(f"Loaded {len(frames_data)} frames")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Plot geometry with obstacles
    plot_geometry(ax, walkable_areas, obstacles)
    
    # Initialize agent scatter plot
    scatter = ax.scatter([], [], c='red', s=100, alpha=0.8, edgecolors='darkred', linewidths=1.5)
    
    # Frame counter text
    frame_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                         verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    frame_list = sorted(frames_data.keys())
    
    def init():
        scatter.set_offsets(np.empty((0, 2)))
        frame_text.set_text('')
        return scatter, frame_text
    
    def update(frame_idx):
        frame = frame_list[frame_idx]
        data = frames_data[frame]
        
        positions = np.array(data['positions'])
        scatter.set_offsets(positions)
        
        # Update frame counter
        time_sec = frame * 0.05  # Assuming 0.05s timestep
        frame_text.set_text(f'Frame: {frame}\nTime: {time_sec:.2f}s\nAgents: {len(positions)}')
        
        return scatter, frame_text
    
    # Create animation
    anim = FuncAnimation(fig, update, init_func=init,
                        frames=len(frame_list), interval=50, 
                        blit=True, repeat=True)
    
    plt.title('Station Simulation - JuPedSim', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_final_frame(trajectory_db: str, network_path: str):
    """
    Plot the final frame of the simulation (static image).
    
    Args:
        trajectory_db: Path to trajectory SQLite database
        network_path: Path to network directory
    """
    # Load geometry
    walking_areas_file = Path(network_path) / "walking_areas.add.xml"
    walkable_areas = load_walkable_areas(str(walking_areas_file))
    obstacles = load_obstacles(str(walking_areas_file))
    
    # Load trajectory data
    frames_data = load_trajectory_data(trajectory_db)
    
    if not frames_data:
        print("No trajectory data found!")
        return
    
    # Get final frame
    final_frame = max(frames_data.keys())
    final_data = frames_data[final_frame]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Plot geometry with obstacles
    plot_geometry(ax, walkable_areas, obstacles)
    
    # Plot agents
    positions = np.array(final_data['positions'])
    ax.scatter(positions[:, 0], positions[:, 1], 
              c='red', s=100, alpha=0.8, edgecolors='darkred', linewidths=1.5)
    
    plt.title(f'Final Frame (t={final_frame * 0.05:.2f}s) - {len(positions)} agents', 
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python visualize.py <trajectory_db>")
        print("Example: python visualize.py output/trajectory.db")
        sys.exit(1)
    
    trajectory_db = sys.argv[1]
    network_path = Path(__file__).parent / ".." / "station_sim" / "network"
    
    print("Creating visualization...")
    visualize_simulation(trajectory_db, str(network_path))
