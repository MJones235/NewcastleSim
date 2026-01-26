"""
Simulation manager for the station simulation.
Extends the base simulation manager with station-specific functionality.
"""

import traci
import sumolib
from typing import Dict
import sys
import os

# Add parent directory to path for base imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from base.manager_base import SimulationManagerBase
from base.diagnostics import SimulationDiagnostics

from agent import StationAgent
from population_loader import PopulationLoader
from station_network import StationNetwork
from message_ui import MessageUI


class StationSimulationManager(SimulationManagerBase):
    """
    Manages the station simulation, coordinating pedestrian agents and SUMO.
    """
    
    def __init__(self, network_file: str, walking_areas_file: str, stops_file: str):
        super().__init__(network_file)
        self.walking_areas_file = walking_areas_file
        self.stops_file = stops_file
        self.diagnostics = SimulationDiagnostics()
        
        # Station network metadata
        self.station_network = None
        
        # Station-specific tracking
        self.pedestrian_edges = []
        self.agents_to_spawn = []  # Queue of agents waiting to spawn
        self.spawn_interval = 1.0  # Seconds between spawns
        self.last_spawn_time = -999  # Time of last spawn (start far in past)
        
        # Message UI
        self.message_ui = MessageUI(on_message_callback=self.broadcast_message)
        self.message_ui.run()
    
    def load_network(self):
        """Load the SUMO network and identify pedestrian infrastructure"""
        super().load_network()
        
        # Initialize station network metadata
        self.station_network = StationNetwork(self.stops_file, self.walking_areas_file)
        print(f"Loaded station network: {self.station_network}")
        
        if self.network:
            # Find all pedestrian-accessible edges
            self.pedestrian_edges = [
                edge for edge in self.network.getEdges() 
                if edge.allows("pedestrian")
            ]
            print(f"Found {len(self.pedestrian_edges)} pedestrian edges")
    
    def add_agent(self, agent: StationAgent):
        """Register an agent in the simulation"""
        agent.diagnostics = self.diagnostics  # Link diagnostics
        self.agents[agent.id] = agent
        
        # Reduced logging - only log every 10th agent for small populations
        if len(self.agents) % 10 == 0:
            print(f"Added {len(self.agents)} agents...")
    
    def load_population(self, num_agents: int = 100, decision_maker_config: dict = None):
        """
        Load pedestrian population placed randomly in walking areas.
        
        Args:
            num_agents: Number of agents to create
            decision_maker_config: Configuration for decision makers (e.g., {'evacuation_probability': 0.5})
        """
        print(f"Loading population of {num_agents} agents with rule-based decision makers...")
        
        loader = PopulationLoader(decision_maker_config=decision_maker_config)
        agents = loader.create_agents(num_agents, self.station_network)
        
        for agent in agents:
            self.add_agent(agent)
        
        print(f"Loaded {len(self.agents)} agents into simulation")
    
    def spawn_agents(self):
        """
        Queue agents for gradual spawning into SUMO.
        Agents will be spawned with minimum interval to avoid JuPedSim distance violations.
        """
        print("Queueing agents for spawning...")
        

        # Queue all unspawned agents
        self.agents_to_spawn = [agent for agent in self.agents.values() if not agent.is_spawned]
        print(f"Queued {len(self.agents_to_spawn)} agents for gradual spawning ({self.spawn_interval}s interval)")
    
    def step(self, sim_time: int):
        """Execute one simulation step"""
        self.current_time = sim_time
        
        # Spawn one agent if interval has passed and any are waiting
        if self.agents_to_spawn and (sim_time - self.last_spawn_time >= self.spawn_interval):
            agent = self.agents_to_spawn.pop(0)
            try:
                agent.spawn_in_sumo(self.network)
                self.last_spawn_time = sim_time
            except Exception as e:
                print(f"Failed to spawn {agent.id}: {e}")
                self.diagnostics.failed_insertions += 1
                
        # Update all agents
        for agent in self.agents.values():
            agent.update(sim_time)
    
    def _find_nearest_pedestrian_edge(self, position: tuple) -> sumolib.net.edge.Edge:
        """Find the nearest pedestrian edge to a given (x, y) position"""
        if not self.pedestrian_edges:
            return None
        
        x, y = position
        min_dist = float('inf')
        nearest_edge = None
        
        for edge in self.pedestrian_edges:
            # Get edge shape (list of coordinates)
            shape = edge.getShape()
            
            # Calculate distance to first point of edge (simplified)
            edge_x, edge_y = shape[0]
            dist = ((x - edge_x) ** 2 + (y - edge_y) ** 2) ** 0.5
            
            if dist < min_dist:
                min_dist = dist
                nearest_edge = edge
        
        return nearest_edge
    
    def broadcast_message(self, message: str):
        """Broadcast a message to all agents"""
        print(f"Broadcasting message to {len(self.agents)} agents: '{message}'")
        for agent in self.agents.values():
            agent.receive_message(message)
    
    def get_simulation_statistics(self) -> dict:
        """Get station simulation statistics"""
        active_count = sum(1 for agent in self.agents.values() if not agent.is_complete())
        completed_count = sum(1 for agent in self.agents.values() if agent.is_complete())
        
        return {
            'total_agents': self.get_agent_count(),
            'active_agents': active_count,
            'completed_agents': completed_count,
            'spawned_agents': self.diagnostics.trip_starts,
            'trip_completions': self.diagnostics.trip_completions,
            'failed_insertions': self.diagnostics.failed_insertions
        }
            
