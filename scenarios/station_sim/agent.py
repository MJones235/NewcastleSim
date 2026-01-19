"""
Agent class for Newcastle station simulation.
Each agent represents a pedestrian moving through the station.
Extends the base agent class with station-specific behavior.
"""

from enum import Enum
from typing import Optional, Tuple
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
    
    def __init__(self, agent_id: str, start_position: Tuple[float, float], 
                 destination: str, destination_type: str = "platform"):
        super().__init__(agent_id)
        
        # Spatial state (station-specific)
        self.position = start_position  # (x, y) coordinates
        self.destination = destination  # platform ID, exit name, etc.
        self.destination_type = destination_type  # "platform", "exit"
        
        # Movement state
        self.state = AgentState.ENTERING
        self.person_id: Optional[str] = None  # SUMO person ID when spawned
        self.is_spawned = False
        
        # Train information (if catching a train)
        self.train_id: Optional[str] = None
        self.train_departure_time: Optional[int] = None
    
    def get_current_location(self) -> Tuple[float, float]:
        """Get the agent's current (x, y) position"""
        return self.position
    
    def set_train_info(self, train_id: str, departure_time: int):
        """Set information about the train this agent is catching"""
        self.train_id = train_id
        self.train_departure_time = departure_time
    
    def spawn_in_sumo(self, network):
        """Create this agent as a person in SUMO and give them a walking stage with specific routing"""
        if self.is_spawned:
            return
        
        import sumolib
        import xml.etree.ElementTree as ET
        
        self.person_id = f"person_{self.id}"
        
        # Specific routing path:
        # 1. Start at junction 3608883591 (entering edge 334461310)
        # 2. Walk to end of entrance edge 334461310
        # 3. Go to junction 501986366 (entering edge 540275666#0)
        # 4. Go to junction 2639595885
        # 5. Enter JuPedSim walking area
        # 6. Go to busStop 4270733515 (possibly via access 258625791)
        
        entrance_edge = "334461310"  # Junction 3608883591
        
        try:
            # Add person at entrance edge at the start (position 0)
            print(f"  Adding person '{self.person_id}' to edge '{entrance_edge}' at position 0.0")
            traci.person.add(self.person_id, entrance_edge, 0.0)
            
            # Check if person was added
            person_list = traci.person.getIDList()
            if self.person_id in person_list:
                print(f"  ✓ Person added successfully")
                pos = traci.person.getPosition(self.person_id)
                print(f"  Position: ({pos[0]:.1f}, {pos[1]:.1f})")
            else:
                print(f"  ✗ Person NOT in simulation after add!")
            
            # Create single walking stage through all waypoint edges to busStop
            # This prevents position mismatches between stages
            edges_to_walk = [
                entrance_edge,      # Start edge (already on it)
                "540275666#0",      # Junction 501986366
                "258625791"         # Access to platform (junction 2639595885)
            ]
            
            print(f"  Walking through edges: {' -> '.join(edges_to_walk)}")
            print(f"  Final destination: busStop {self.destination}")
            
            traci.person.appendWalkingStage(
                self.person_id, 
                edges_to_walk, 
                0.0,  # Arrival position (0 = busStop location)
                stopID=self.destination
            )
            
            # Check stages
            stages = traci.person.getRemainingStages(self.person_id)
            print(f"  Remaining stages: {stages}")
            
            self.is_spawned = True
            self.state = AgentState.WALKING
            
            if self.diagnostics:
                self.diagnostics.trip_starts += 1
                
        except traci.exceptions.TraCIException as e:
            print(f"Failed to spawn agent {self.id}: {e}")
            if self.diagnostics:
                self.diagnostics.failed_insertions += 1
    
    def _find_busstop_access_lane(self, busstop_id: str) -> Optional[str]:
        """Find the pedestrian access lane for a busStop by parsing osm_stops.add.xml"""
        import xml.etree.ElementTree as ET
        import os
        
        # Path to stops file
        stops_file = 'scenarios/station_sim/network/osm_stops.add.xml'
        
        if not os.path.exists(stops_file):
            return None
        
        try:
            tree = ET.parse(stops_file)
            root = tree.getroot()
            
            # Find the busStop with matching id
            for busstop in root.findall('.//busStop'):
                if busstop.get('id') == busstop_id:
                    # Look for access child element
                    access = busstop.find('access')
                    if access is not None:
                        access_lane = access.get('lane')
                        return access_lane
            
            return None
        except Exception as e:
            print(f"Warning: Could not parse busStop access: {e}")
            return None
    
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
        if self.train_id and self.train_departure_time:
            # Check if train has arrived
            if sim_time >= self.train_departure_time:
                self.state = AgentState.BOARDING
    
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
