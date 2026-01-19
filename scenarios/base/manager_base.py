"""
Abstract base class for simulation managers.
Coordinates agents and SUMO simulation.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, TYPE_CHECKING
import traci
import sumolib

if TYPE_CHECKING:
    from .agent_base import AgentBase


class SimulationManagerBase(ABC):
    """
    Abstract base class for simulation managers.
    Handles common simulation lifecycle and agent coordination.
    """
    
    def __init__(self, network_file: str):
        """
        Initialize base simulation manager.
        
        Args:
            network_file: Path to SUMO network file
        """
        self.network_file = network_file
        self.network: Optional[sumolib.net.Net] = None
        self.agents: Dict[str, 'AgentBase'] = {}
        self.current_time = 0
            
    def load_network(self):
        """Load the SUMO network for analysis"""
        try:
            self.network = sumolib.net.readNet(self.network_file)
            print(f"Loaded network with {len(self.network.getEdges())} edges")
        except Exception as e:
            print(f"Warning: Could not load network: {e}")
    
    @abstractmethod
    def add_agent(self, agent: 'AgentBase'):
        """
        Register an agent in the simulation.
        Must be implemented by subclasses to handle agent-specific setup.
        
        Args:
            agent: The agent to add
        """
        pass
    
    def step(self):
        """Execute one simulation step"""
        # Advance SUMO
        traci.simulationStep()
        self.current_time += 1
        
        # Update all agents
        for agent in self.agents.values():
            agent.update(self.current_time)
                
    def broadcast_event(self, event: dict):
        """Send an event to all agents"""
        for agent in self.agents.values():
            agent.receive_message(event)
    
    def get_agent_count(self) -> int:
        """Get the number of registered agents"""
        return len(self.agents)
    
    def get_active_agent_count(self) -> int:
        """Get the number of agents that haven't completed their schedule"""
        return sum(1 for agent in self.agents.values() if not agent.is_schedule_complete())
    
    @abstractmethod
    def get_simulation_statistics(self) -> dict:
        """
        Get simulation-specific statistics.
        Must be implemented by subclasses.
        
        Returns:
            Dictionary of simulation metrics
        """
        pass
