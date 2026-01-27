"""
JuPedSim-specific movement provider.
Handles spawning, positioning, and routing for JuPedSim simulations.
"""

import jupedsim as jps
from typing import Any, Dict, Tuple
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from base.movement_provider import MovementProvider


class JuPedSimMovementProvider(MovementProvider):
    """
    Movement provider for JuPedSim pedestrian simulations.
    Handles JuPedSim agent API calls and stage-based routing.
    """
    
    def __init__(self, simulation: jps.Simulation, zones: Dict[str, Any]):
        """
        Initialize JuPedSim movement provider.
        
        Args:
            simulation: JuPedSim simulation object
            zones: Dictionary mapping zone names to polygons
        """
        self.simulation = simulation
        self.zones = zones
    
    def spawn_agent(self, agent: Any, spawn_params: Dict[str, Any]) -> bool:
        """
        Spawn an agent in JuPedSim simulation.
        
        Args:
            agent: StationAgent to spawn
            spawn_params: Must contain:
                - position: (x, y) tuple for spawn location
                - journey_id: JuPedSim journey ID
                
        Returns:
            True if spawn successful
        """
        try:
            position = spawn_params['position']
            journey_id = spawn_params['journey_id']
            
            # Add agent to JuPedSim with journey
            jps_id = self.simulation.add_agent(
                jps.CollisionFreeSpeedModelAgentParameters(
                    position=position,
                    journey_id=journey_id,
                    stage_id=spawn_params.get('stage_id'),
                    v0=agent.walking_speed
                )
            )
            
            # Store JuPedSim agent ID
            agent.jps_agent_id = jps_id
            return True
            
        except Exception as e:
            print(f"Failed to spawn agent {agent.id} in JuPedSim: {e}")
            return False
    
    def update_agent_position(self, agent: Any) -> Tuple[float, float]:
        """
        Get agent's current position from JuPedSim.
        
        Args:
            agent: StationAgent with jps_agent_id attribute
            
        Returns:
            (x, y) coordinates
        """
        if not hasattr(agent, 'jps_agent_id'):
            return (0.0, 0.0)
        
        try:
            position = self.simulation.agent_position(agent.jps_agent_id)
            return (position[0], position[1])
        except:
            return agent.position  # Return last known position
    
    def set_agent_target(self, agent: Any, target: Any) -> bool:
        """
        Set new journey/stage for agent.
        
        Args:
            agent: StationAgent with jps_agent_id
            target: Dictionary with 'journey_id' or 'stage_id' for new target
            
        Returns:
            True if successfully redirected
        """
        if not hasattr(agent, 'jps_agent_id'):
            return False
        
        try:
            # JuPedSim uses journey switching for rerouting
            if 'journey_id' in target:
                self.simulation.switch_agent_journey(
                    agent.jps_agent_id,
                    target['journey_id']
                )
                return True
            return False
            
        except Exception as e:
            print(f"Failed to reroute agent {agent.id}: {e}")
            return False
    
    def is_agent_active(self, agent: Any) -> bool:
        """
        Check if agent still exists in JuPedSim simulation.
        
        Args:
            agent: StationAgent with jps_agent_id
            
        Returns:
            True if agent still in simulation
        """
        if not hasattr(agent, 'jps_agent_id'):
            return False
        
        try:
            # Check if agent ID is in current agent list
            agent_ids = [a.id for a in self.simulation.agents()]
            return agent.jps_agent_id in agent_ids
        except:
            return False
    
    def remove_agent(self, agent: Any):
        """
        Remove agent from JuPedSim simulation.
        
        Args:
            agent: StationAgent with jps_agent_id
        """
        if hasattr(agent, 'jps_agent_id'):
            try:
                self.simulation.mark_agent_for_removal(agent.jps_agent_id)
            except:
                pass
    
    def get_agent_location_info(self, agent: Any) -> Dict[str, Any]:
        """
        Get detailed location information from JuPedSim.
        
        Args:
            agent: StationAgent with jps_agent_id
            
        Returns:
            Dictionary with zone, position, etc.
        """
        if not hasattr(agent, 'jps_agent_id'):
            return {'zone': 'unknown'}
        
        try:
            position = self.simulation.agent_position(agent.jps_agent_id)
            
            # Determine which zone contains this position
            from shapely.geometry import Point
            point = Point(position[0], position[1])
            
            for zone_name, polygon in self.zones.items():
                if polygon.contains(point):
                    return {
                        'zone': zone_name,
                        'position': position
                    }
            
            return {
                'zone': 'unknown',
                'position': position
            }
        except:
            return {'zone': 'unknown'}
