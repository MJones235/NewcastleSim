"""
Abstract base class for agent decision-making modules.
Uses Strategy pattern to allow swapping between different decision-making approaches.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, Optional


class Decision(Enum):
    """Possible decisions an agent can make in response to messages"""
    IGNORE = "ignore"
    EVACUATE = "evacuate"


class DecisionMakerBase(ABC):
    """
    Abstract base class for decision-making modules.
    Subclasses implement different strategies (rule-based, LLM-based, etc.)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize decision maker with optional configuration.
        
        Args:
            config: Configuration dictionary for the decision maker
        """
        self.config = config or {}
    
    @abstractmethod
    def make_decision(self, 
                     message: str, 
                     agent_state: Dict[str, Any],
                     context: Dict[str, Any]) -> Decision:
        """
        Make a decision based on a message and current context.
        
        Args:
            message: The message received by the agent
            agent_state: Current state of the agent (position, destination, etc.)
            context: Simulation context (time, other agents, network state, etc.)
            
        Returns:
            Decision enum indicating what action to take
        """
        pass
    
    @abstractmethod
    def get_decision_reasoning(self) -> str:
        """Get explanation for the last decision made."""
        pass
