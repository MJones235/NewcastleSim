"""
Agent class for Newcastle station simulation.
Each agent represents a pedestrian moving through the station.
Extends the base agent class with station-specific behavior.
"""

from enum import Enum
from typing import Optional, Tuple, List
import traci
import sys
import os

# Add parent directory to path for base imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from base.agent_base import AgentBase


class AgentState(Enum):
    """States a station agent can be in"""
    ENTERING = "entering"
    WALKING = "walking"
    WAITING = "waiting"
    BOARDING = "boarding"
    EXITING = "exiting"
    COMPLETED = "completed"


class StationAgent(AgentBase):
    """
    Represents a single pedestrian in the station simulation.
    Manages navigation through the station to reach a destination (platform or exit).
    """
    
    def __init__(self, agent_id: str, entrance_edge: str, 
                 destination: str, route: List[str], spawn_position: float = -1.0,
                 destination_type: str = "platform"):
        super().__init__(agent_id)
        
        # Spatial state (station-specific)
        self.entrance_edge = entrance_edge  # Edge to spawn at
        self.spawn_position = spawn_position  # Position on entrance edge (0.0=start, -1=end)
        self.route = route  # Full edge sequence from entrance to platform
        self.destination = destination  # busStop ID
        self.destination_type = destination_type  # "platform", "exit"
        self.position = (0.0, 0.0)  # Will be updated from SUMO
        
        # Movement state
        self.state = AgentState.ENTERING
        self.person_id: Optional[str] = None  # SUMO person ID when spawned
        self.is_spawned = False
    
    def get_current_location(self) -> Tuple[float, float]:
        """Get the agent's current (x, y) position"""
        return self.position
    
    def spawn_in_sumo(self, network):
        """Create this agent as a person in SUMO and give them a walking stage with specific routing"""
        if self.is_spawned:
            return
                
        self.person_id = f"person_{self.id}"
        
        try:
            # Add person at entrance edge at specified position
            traci.person.add(self.person_id, self.entrance_edge, self.spawn_position)
            
            print(f"  Agent {self.id}: route = {' → '.join(self.route)}")
            print(f"  Destination: busStop {self.destination}")
                        
            traci.person.appendWalkingStage(
                self.person_id, 
                self.route,  # Full route including footbridge edges if needed
                0.0,  # Arrival position (0 = busStop location)
                stopID=self.destination
            )
                        
            self.is_spawned = True
            self.state = AgentState.WALKING
            
            if self.diagnostics:
                self.diagnostics.trip_starts += 1
                
        except traci.exceptions.TraCIException as e:
            print(f"Failed to spawn agent {self.id}: {e}")
            if self.diagnostics:
                self.diagnostics.failed_insertions += 1
    
    def update(self, sim_time: int):
        """Main update logic called each simulation step"""
        if not self.is_spawned:
            return
        
        if self.state == AgentState.WALKING:
            self._update_walking(sim_time)
        elif self.state == AgentState.WAITING:
            self._update_waiting(sim_time)
        elif self.state == AgentState.BOARDING:
            self._update_boarding(sim_time)
    
    def _update_walking(self, sim_time: int):
        """Update when agent is walking to destination"""
        # Check if agent has reached destination
        # This will be implemented as we develop the navigation logic
        pass
    
    def _update_waiting(self, sim_time: int):
        """Update when agent is waiting (e.g., for a train)"""
        pass
    
    def _update_boarding(self, sim_time: int):
        """Update when agent is boarding a train"""
        # Remove agent from simulation once boarded
        self._remove_from_simulation()
        self.state = AgentState.COMPLETED
    
    def _remove_from_simulation(self):
        """Remove this agent's person from SUMO"""
        if self.person_id and self.is_spawned:
            try:
                if self.person_id in traci.person.getIDList():
                    traci.person.remove(self.person_id)
                    
                if self.diagnostics:
                    self.diagnostics.trip_completions += 1
                    
            except traci.exceptions.TraCIException:
                pass
            
            self.is_spawned = False
    
    def is_complete(self) -> bool:
        """Check if agent has completed their journey"""
        return self.state == AgentState.COMPLETED
    
    def __repr__(self):
        return f"StationAgent(id={self.id}, state={self.state.value}, dest={self.destination})"
