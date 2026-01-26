"""
Rule-based decision maker using probabilistic logic.
"""

import random
from typing import Dict, Any
import sys
import os

# Add parent directory to path for base imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from base.decision_maker_base import DecisionMakerBase, Decision


class RuleBasedDecisionMaker(DecisionMakerBase):
    """
    Simple rule-based decision maker using probabilistic logic.
    Matches keywords and decides whether to evacuate or ignore.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        
        # Probability to evacuate when evacuation keywords detected
        self.evacuation_probability = self.config.get('evacuation_probability', 0.5)
        
        # Keywords that trigger evacuation consideration
        self.evacuation_keywords = ['evacuate', 'emergency', 'leave', 'exit', 'danger', 'fire', 'threat']
        
        # Store last decision reasoning
        self.last_reasoning = ""
    
    def make_decision(self, 
                     message: str, 
                     agent_state: Dict[str, Any],
                     context: Dict[str, Any]) -> Decision:
        """
        Make a decision based on keyword matching and probability.
        
        Args:
            message: The message received
            agent_state: Current agent state
            context: Simulation context
            
        Returns:
            Decision.EVACUATE or Decision.IGNORE
        """
        message_lower = message.lower()
        
        # Check for evacuation keywords
        if any(keyword in message_lower for keyword in self.evacuation_keywords):
            return Decision.EVACUATE
            #if random.random() < self.evacuation_probability:
            #    self.last_reasoning = f"Evacuation keyword detected. Evacuating (p={self.evacuation_probability})"
            #    return Decision.EVACUATE
            #else:
            #    self.last_reasoning = f"Evacuation keyword detected. Ignoring (p={1-self.evacuation_probability})"
            #    return Decision.IGNORE
        
        # Default: ignore messages without evacuation keywords
        self.last_reasoning = "No evacuation keywords. Ignoring."
        return Decision.IGNORE
    
    def get_decision_reasoning(self) -> str:
        """Return the reasoning for the last decision"""
        return self.last_reasoning
