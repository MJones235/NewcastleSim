"""
Simulation manager that coordinates agents and SUMO for traffic simulation.
Extends the base simulation manager with traffic-specific functionality.
"""

import sys
import os

# Add parent directory to path for base imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from base.manager_base import SimulationManagerBase
from base.diagnostics import SimulationDiagnostics

from agent import Agent
from population_loader import PopulationLoader


class SimulationManager(SimulationManagerBase):
    """
    Manages the traffic simulation, coordinating agents and SUMO.
    """
    
    def __init__(self, network_file: str):
        super().__init__(network_file)
        self.diagnostics = SimulationDiagnostics()
    
    def add_agent(self, agent: Agent):
        """Register an agent in the simulation"""
        agent.diagnostics = self.diagnostics  # Link diagnostics
        self.agents[agent.id] = agent
        # Reduced logging - only log every 1000th agent
        if len(self.agents) % 1000 == 0:
            print(f"Added {len(self.agents)} agents...")
    
    def load_population(self, population_file: str, max_agents: int = None, use_trips: bool = False):
        """Load agents from population file
        
        Args:
            population_file: Path to CSV file
            max_agents: Optional limit on agents to load (for testing)
            use_trips: If True, load from trip-based CSV instead of demographics
        """
        print(f"Loading population from {population_file}...")
        
        loader = PopulationLoader(self.network_file)
        if use_trips:
            agents = loader.load_from_trip_csv(population_file, max_agents=max_agents)
        else:
            agents = loader.load_from_csv(population_file, max_agents=max_agents)
        
        for agent in agents:
            self.add_agent(agent)
    
    def get_simulation_statistics(self) -> dict:
        """Get traffic simulation statistics"""
        return {
            'total_agents': self.get_agent_count(),
            'active_agents': self.get_active_agent_count(),
            'trip_starts': self.diagnostics.trip_starts,
            'trip_completions': self.diagnostics.trip_completions,
            'teleports': self.diagnostics.teleports,
            'failed_insertions': self.diagnostics.failed_insertions
        }
