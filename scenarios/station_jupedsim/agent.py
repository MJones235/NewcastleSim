"""
Agent management for JuPedSim station simulation.
"""

import numpy as np
from typing import Optional, List
from pathlib import Path
import sys

# Import decision maker from base
sys.path.append(str(Path(__file__).parent.parent))
from base.decision_maker_base import DecisionMakerBase, Decision


class StationAgent:
    """Represents an agent in the JuPedSim station simulation."""
    
    def __init__(
        self,
        agent_id: int,
        jps_agent_id: int,
        walking_speed: float,
        decision_maker: DecisionMakerBase,
        initial_zone: str
    ):
        """
        Initialize a station agent.
        
        Args:
            agent_id: Sequential agent ID for tracking
            jps_agent_id: JuPedSim internal agent ID
            walking_speed: Preferred walking speed in m/s
            decision_maker: Decision making module
            initial_zone: Starting zone name
        """
        self.agent_id = agent_id
        self.jps_agent_id = jps_agent_id
        self.walking_speed = walking_speed
        self.decision_maker = decision_maker
        self.initial_zone = initial_zone
        
        # Message tracking
        self._last_message_index = -1
        
        # State
        self.is_evacuating = False
        self.original_journey_id = None
        self.evacuation_target = None
        
    def process_messages(self, messages: List[str]):
        """
        Process new messages and make decisions.
        
        Args:
            messages: List of broadcast messages
        """
        # Only process new messages
        new_messages = messages[self._last_message_index + 1:]
        
        if not new_messages:
            return
            
        # Update message index
        self._last_message_index = len(messages) - 1
        
        # Make decision for most recent message
        latest_message = new_messages[-1]
        decision = self.decision_maker.make_decision(latest_message)
        
        if decision == Decision.EVACUATE and not self.is_evacuating:
            print(f"Agent {self.agent_id} deciding to evacuate")
            self.is_evacuating = True
            # Evacuation will be handled by simulation manager


def sample_walking_speed(mean: float = 1.34, std: float = 0.37, 
                         min_speed: float = 0.5, max_speed: float = 2.5) -> float:
    """
    Sample walking speed from normal distribution.
    
    Args:
        mean: Mean walking speed (m/s)
        std: Standard deviation (m/s)
        min_speed: Minimum allowed speed
        max_speed: Maximum allowed speed
        
    Returns:
        Walking speed in m/s
    """
    speed = np.random.normal(mean, std)
    return max(min_speed, min(max_speed, speed))


if __name__ == "__main__":
    # Test walking speed distribution
    print("Testing walking speed distribution:")
    speeds = [sample_walking_speed() for _ in range(1000)]
    print(f"  Mean: {np.mean(speeds):.3f} m/s")
    print(f"  Std: {np.std(speeds):.3f} m/s")
    print(f"  Min: {np.min(speeds):.3f} m/s")
    print(f"  Max: {np.max(speeds):.3f} m/s")
