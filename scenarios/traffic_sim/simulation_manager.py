"""
Simulation manager that coordinates agents and SUMO.
"""

import traci
import sumolib
from typing import Dict
from agent import Agent, Activity, ActivityType
from population_loader import PopulationLoader
from diagnostics import SimulationDiagnostics


class SimulationManager:
    """
    Manages the overall simulation, coordinating agents and SUMO.
    """
    
    def __init__(self, network_file: str):
        self.network_file = network_file
        self.network = None
        self.agents: Dict[str, Agent] = {}
        self.current_time = 0
        self.diagnostics = SimulationDiagnostics()
        self.screenshot_enabled = False
        self.screenshot_interval = 10
        self.screenshot_prefix = "output/frame_"
        
    def load_network(self):
        """Load the SUMO network for analysis"""
        try:
            self.network = sumolib.net.readNet(self.network_file)
            print(f"Loaded network with {len(self.network.getEdges())} edges")
        except Exception as e:
            print(f"Warning: Could not load network: {e}")
            # #TODO: Handle network loading failures
    
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
    
        
    def step(self):
        """Execute one simulation step"""
        traci.simulationStep()
        self.current_time += 1
        
        # Update all agents
        for agent in self.agents.values():
            agent.update(self.current_time)
        
        # Take screenshot if recording is enabled
        if self.screenshot_enabled and self.current_time % self.screenshot_interval == 0:
            try:
                filename = f"{self.screenshot_prefix}{self.current_time:06d}.png"
                traci.gui.screenshot(traci.gui.DEFAULT_VIEW, filename)
            except:
                pass  # Silently fail if GUI not available
            
    def broadcast_event(self, event: dict):
        """Send an event to all agents"""
        for agent in self.agents.values():
            agent.receive_message(event)
