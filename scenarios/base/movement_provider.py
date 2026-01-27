"""
Abstract base class for movement providers.
Separates movement/routing logic from agent decision-making logic.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple


class MovementProvider(ABC):
    """
    Abstract interface for movement/routing in different simulators.
    Implementations handle simulator-specific movement logic (SUMO, JuPedSim, etc.)
    """
    
    @abstractmethod
    def spawn_agent(self, agent: Any, spawn_params: Dict[str, Any]) -> bool:
        """
        Spawn an agent in the simulator.
        
        Args:
            agent: The agent to spawn
            spawn_params: Simulator-specific spawn parameters
            
        Returns:
            True if spawn successful, False otherwise
        """
        pass
    
    @abstractmethod
    def update_agent_position(self, agent: Any) -> Tuple[float, float]:
        """
        Update and return agent's current position.
        
        Args:
            agent: The agent to update
            
        Returns:
            (x, y) coordinates
        """
        pass
    
    @abstractmethod
    def set_agent_target(self, agent: Any, target: Any) -> bool:
        """
        Set a new movement target for the agent.
        
        Args:
            agent: The agent to update
            target: Simulator-specific target (stage_id, edge_id, etc.)
            
        Returns:
            True if target set successfully
        """
        pass
    
    @abstractmethod
    def is_agent_active(self, agent: Any) -> bool:
        """
        Check if agent is still active in the simulation.
        
        Args:
            agent: The agent to check
            
        Returns:
            True if agent is still moving/active
        """
        pass
    
    @abstractmethod
    def remove_agent(self, agent: Any):
        """
        Remove agent from the simulator.
        
        Args:
            agent: The agent to remove
        """
        pass
    
    @abstractmethod
    def get_agent_location_info(self, agent: Any) -> Dict[str, Any]:
        """
        Get detailed location information for the agent.
        
        Args:
            agent: The agent to query
            
        Returns:
            Dictionary with simulator-specific location info
        """
        pass
