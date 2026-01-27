"""
SUMO-specific movement provider.
Handles spawning, positioning, and routing for SUMO simulations.
"""

import traci
from typing import Any, Dict, Tuple
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from base.movement_provider import MovementProvider


class SUMOMovementProvider(MovementProvider):
    """
    Movement provider for SUMO pedestrian simulations.
    Handles SUMO person API calls and routing.
    """
    
    def __init__(self, network=None):
        """
        Initialize SUMO movement provider.
        
        Args:
            network: Optional SUMO network object for analysis
        """
        self.network = network
    
    def spawn_agent(self, agent: Any, spawn_params: Dict[str, Any]) -> bool:
        """
        Spawn an agent as a SUMO person.
        
        Args:
            agent: StationAgent to spawn
            spawn_params: Must contain:
                - entrance_edge: Edge ID to spawn at
                - route: List of edge IDs to destination
                - destination: busStop ID
                - spawn_position: Position on edge (0.0=start, -1.0=end)
                
        Returns:
            True if spawn successful
        """
        person_id = f"person_{agent.id}"
        
        try:
            # Add person at entrance edge
            entrance_edge = spawn_params['entrance_edge']
            spawn_position = spawn_params.get('spawn_position', -1.0)
            route = spawn_params['route']
            destination = spawn_params['destination']
            
            # Add person at entrance edge
            traci.person.add(person_id, entrance_edge, spawn_position)
            traci.person.setSpeed(person_id, agent.walking_speed)
            
            # Add walking stage to platform
            traci.person.appendWalkingStage(
                person_id,
                route,
                0.0,
                stopID=destination
            )
            
            # Add driving stage (train ride) - always added like in old code
            busstop_edge = traci.busstop.getLaneID(destination).rsplit('_', 1)[0]
            traci.person.appendDrivingStage(
                person_id,
                toEdge=busstop_edge,
                lines="ANY",
                stopID=destination
            )
            
            # Store SUMO person ID in agent
            agent.person_id = person_id
            return True
            
        except traci.exceptions.TraCIException as e:
            print(f"Failed to spawn agent {agent.id}: {e}")
            return False
    
    def update_agent_position(self, agent: Any) -> Tuple[float, float]:
        """
        Get agent's current position from SUMO.
        
        Args:
            agent: StationAgent with person_id attribute
            
        Returns:
            (x, y) coordinates
        """
        if not hasattr(agent, 'person_id') or not agent.person_id:
            return (0.0, 0.0)
        
        try:
            # Check if person still exists before querying position
            if agent.person_id not in traci.person.getIDList():
                return agent.position  # Return last known position
            
            position = traci.person.getPosition(agent.person_id)
            return position
        except traci.exceptions.TraCIException:
            return agent.position  # Return last known position
    
    def set_agent_target(self, agent: Any, target: Any) -> bool:
        """
        Redirect agent to new destination.
        
        Args:
            agent: StationAgent with person_id
            target: Dictionary with 'route' and 'destination' for new target
            
        Returns:
            True if successfully rerouted
        """
        if not hasattr(agent, 'person_id') or not agent.person_id:
            return False
        
        try:
            # Remove remaining stages
            traci.person.removeStages(agent.person_id)
            
            # Add new walking stage to evacuation target
            new_route = target['route']
            new_destination = target['destination']
            
            traci.person.appendWalkingStage(
                agent.person_id,
                new_route,
                0.0,
                stopID=new_destination
            )
            
            return True
            
        except traci.exceptions.TraCIException as e:
            print(f"Failed to reroute agent {agent.id}: {e}")
            return False
    
    def is_agent_active(self, agent: Any) -> bool:
        """
        Check if agent still exists in SUMO simulation AND has remaining stages.
        
        Args:
            agent: StationAgent with person_id
            
        Returns:
            True if person still active and has stages to complete
        """
        if not hasattr(agent, 'person_id') or not agent.person_id:
            return False
        
        try:
            # Check if person exists in SUMO
            if agent.person_id not in traci.person.getIDList():
                return False
            
            # Check if person has remaining stages
            # If they have 0 stages, they've completed their journey
            remaining = traci.person.getRemainingStages(agent.person_id)
            return remaining > 0
            
        except traci.exceptions.TraCIException:
            return False
    
    def remove_agent(self, agent: Any):
        """
        Remove agent from SUMO simulation.
        
        Args:
            agent: StationAgent with person_id
        """
        if hasattr(agent, 'person_id') and agent.person_id:
            try:
                traci.person.remove(agent.person_id)
            except traci.exceptions.TraCIException:
                pass
    
    def get_agent_location_info(self, agent: Any) -> Dict[str, Any]:
        """
        Get detailed location information from SUMO.
        
        Args:
            agent: StationAgent with person_id
            
        Returns:
            Dictionary with edge, position, angle, etc.
        """
        if not hasattr(agent, 'person_id') or not agent.person_id:
            return {'zone': 'unknown'}
        
        try:
            edge = traci.person.getRoadID(agent.person_id)
            position = traci.person.getPosition(agent.person_id)
            angle = traci.person.getAngle(agent.person_id)
            
            return {
                'zone': edge,
                'edge': edge,
                'position': position,
                'angle': angle
            }
        except traci.exceptions.TraCIException:
            return {'zone': 'unknown'}
