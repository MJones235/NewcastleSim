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
    
    def load_population(self, population_file: str, use_test: bool = False, max_agents: int = None):
        """Load agents from population file
        
        Args:
            population_file: Path to CSV file
            use_test: If True, use test population instead
            max_agents: Optional limit on agents to load (for testing)
        """
        print(f"Loading population from {population_file}...")
        
        if use_test:
            # Use test population for initial testing
            print("(Using test population)")
            self._create_test_population()
        else:
            # Load from actual CSV file
            loader = PopulationLoader(self.network_file)
            agents = loader.load_from_csv(population_file, max_agents=max_agents)
            
            for agent in agents:
                self.add_agent(agent)
    
    def _create_test_population(self):
        """Create a small test population for initial testing"""
        # Create one simple test agent
        test_agent = Agent(
            agent_id="person_001",
            demographics={"age": 35, "employed": True, "has_car": True},
            home_location="edge_home"  # #TODO: Use actual edge IDs from network
        )
        
        # Simple daily schedule: Home -> Work -> Home
        # Times in seconds from midnight
        test_agent.add_activity(Activity(
            ActivityType.HOME, 
            "edge_home", 
            start_time=0, 
            duration=8 * 3600  # Stay home until 8 AM
        ))
        
        test_agent.add_activity(Activity(
            ActivityType.WORK, 
            "edge_work",  # #TODO: Use actual edge IDs
            start_time=8 * 3600,  # Start at 8 AM
            duration=8 * 3600  # Work for 8 hours
        ))
        
        test_agent.add_activity(Activity(
            ActivityType.HOME, 
            "edge_home", 
            start_time=16 * 3600,  # Return at 4 PM
            duration=8 * 3600  # Evening at home
        ))
        
        self.add_agent(test_agent)
        
        # #TODO: Add more agents with varied schedules
    def step(self):
        """Execute one simulation step"""
        traci.simulationStep()
        self.current_time += 1
        
        # Update all agents
        for agent in self.agents.values():
            agent.update(self.current_time)
        
        # Update GUI time display
        self._update_time_display()
    
    def _update_time_display(self):
        """Update the time display in SUMO GUI"""
        hours = self.current_time // 3600
        minutes = (self.current_time % 3600) // 60
        time_str = f"Simulation Time: {hours:02d}:{minutes:02d}"
        
        try:
            # Set GUI parameter to show time in the top bar
            traci.gui.setSchema(traci.gui.DEFAULT_VIEW, "real world")
            traci.gui.trackVehicle(traci.gui.DEFAULT_VIEW, "")  # Clear tracking
        except:
            pass  # GUI commands may fail in non-GUI mode
    
    def _print_status(self):
        """Print current status of all agents"""
        for agent in self.agents.values():
            activity = agent.get_current_activity()
            if activity:
                status = "traveling" if agent.in_transit else activity.type.value
                print(f"  {agent.id}: {status}")
            else:
                print(f"  {agent.id}: day complete")
    
    def get_statistics(self):
        """Collect simulation statistics"""
        # #TODO: Implement comprehensive statistics collection
        stats = {
            "total_agents": len(self.agents),
            "active_vehicles": len([a for a in self.agents.values() if a.in_transit]),
            "simulation_time": self.current_time
        }
        return stats
    
    def broadcast_event(self, event: dict):
        """Send an event to all agents (for future replanning)"""
        # #TODO: Implement event broadcasting and agent filtering
        # This will be used for evacuation scenarios
        for agent in self.agents.values():
            agent.receive_message(event)
