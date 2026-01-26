"""
Abstract base class for agents across different simulation types.
Defines the minimal common interface shared by all agents.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any


class AgentBase(ABC):
    """
    Abstract base class for all agents in simulations.
    Contains only the minimal common elements across all simulation types.
    Each simulation extends this with domain-specific behavior.
    """
    
    def __init__(self, agent_id: str, demographics: Optional[Dict] = None):
        """
        Initialize base agent.
        
        Args:
            agent_id: Unique identifier for this agent
            demographics: Optional demographic information
        """
        self.id = agent_id
        self.demographics = demographics or {}
        
        # For receiving events/messages
        self.messages: list[str] = []  # String array of messages
        self.needs_replan = False
        
        # For diagnostics (set by simulation manager)
        self.diagnostics = None
    
    def receive_message(self, message: str):
        """Receive a message string"""
        self.messages.append(message)
        self.needs_replan = True
    
    @abstractmethod
    def update(self, sim_time: int):
        """
        Main update logic called each simulation step.
        Must be implemented by each simulation type.
        
        Args:
            sim_time: Current simulation time in seconds
        """
        pass
    
    @abstractmethod
    def get_current_location(self) -> Any:
        """
        Get the agent's current location.
        Return type depends on simulation type.
        """
        pass
    
    def __repr__(self):
        return f"{self.__class__.__name__}(id={self.id})"
