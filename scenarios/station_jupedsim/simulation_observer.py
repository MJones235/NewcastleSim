"""
Observer pattern for simulation events.

Allows external components (like GUI) to observe simulation state changes
without tight coupling to the simulation logic.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict, Any


class SimulationObserver(ABC):
    """Abstract base class for simulation observers."""
    
    @abstractmethod
    def on_simulation_step(
        self, 
        sim_time: float, 
        iteration: int,
        agent_count: int,
        agent_positions: List[Tuple[float, float]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Called after each simulation step.
        
        Args:
            sim_time: Current simulation time in seconds
            iteration: Current iteration number
            agent_count: Number of active agents
            agent_positions: List of (x, y) agent positions
            metadata: Optional dictionary with additional info (events, etc.)
        """
        pass
    
    @abstractmethod
    def on_simulation_start(self, total_agents: int) -> None:
        """
        Called when simulation starts.
        
        Args:
            total_agents: Total number of agents to spawn
        """
        pass
    
    @abstractmethod
    def on_simulation_end(self, stats: Dict[str, Any]) -> None:
        """
        Called when simulation completes.
        
        Args:
            stats: Dictionary with simulation statistics
        """
        pass


class GUIObserver(SimulationObserver):
    """
    Observer that updates a GUI viewer.
    
    Acts as an adapter between the observer pattern and the LiveViewer interface.
    """
    
    def __init__(self, viewer, update_interval: float = 1.0):
        """
        Initialize GUI observer.
        
        Args:
            viewer: LiveViewer instance (or any object with update() and close() methods)
            update_interval: Minimum time between GUI updates in seconds
        """
        self.viewer = viewer
        self.update_interval = update_interval
        self.last_update_time = -update_interval  # Allow immediate first update
        
    def on_simulation_step(
        self, 
        sim_time: float, 
        iteration: int,
        agent_count: int,
        agent_positions: List[Tuple[float, float]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Update GUI if enough time has passed."""
        if (sim_time - self.last_update_time) >= self.update_interval:
            event_message = metadata.get('event_message') if metadata else None
            self.viewer.update(agent_positions, sim_time, agent_count, event_message)
            self.last_update_time = sim_time
    
    def on_simulation_start(self, total_agents: int) -> None:
        """Called when simulation starts (no action needed for GUI)."""
        pass
    
    def on_simulation_end(self, stats: Dict[str, Any]) -> None:
        """Close the GUI viewer."""
        self.viewer.close()


class ConsoleObserver(SimulationObserver):
    """
    Observer that prints progress to console.
    
    Provides text-based progress updates during simulation.
    """
    
    def __init__(self, update_interval: int = 100):
        """
        Initialize console observer.
        
        Args:
            update_interval: Print progress every N iterations
        """
        self.update_interval = update_interval
        
    def on_simulation_step(
        self, 
        sim_time: float, 
        iteration: int,
        agent_count: int,
        agent_positions: List[Tuple[float, float]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Print progress at regular intervals."""
        if iteration % self.update_interval == 0:
            spawned = metadata.get('spawned_count', 0) if metadata else 0
            total = metadata.get('total_agents', 0) if metadata else 0
            print(f"t={sim_time:6.2f}s  agents={agent_count:3d}  spawned={spawned:3d}/{total}")
    
    def on_simulation_start(self, total_agents: int) -> None:
        """Print simulation start message."""
        print("\n[5/5] Running simulation...")
        print("Press Ctrl+C to stop\n")
    
    def on_simulation_end(self, stats: Dict[str, Any]) -> None:
        """Print completion message (statistics printed elsewhere)."""
        pass
