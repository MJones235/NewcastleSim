"""
JuPedSim-specific movement provider.

Handles spawning, positioning, and routing for JuPedSim simulations.
Implements the MovementProvider interface for JuPedSim pedestrian dynamics.

This provider manages:
    - Agent spawning with position and journey assignment
    - Position queries and updates from JuPedSim simulation
    - Journey/stage switching for agent rerouting
    - Agent lifecycle (active checking, removal)
    - Zone-based location queries
"""

import jupedsim as jps
from typing import Any, Dict, Tuple

from scenarios.base.movement_provider import MovementProvider


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
        self.evacuation_journeys = {}  # Will be set by main script
        self.evacuation_exits = {}  # Will be set by main script
    
    def spawn_agent(self, agent: Any, spawn_params: Dict[str, Any]) -> bool:
        """
        Spawn an agent in JuPedSim simulation.
        
        Args:
            agent: StationAgent to spawn
            spawn_params: Must contain:
                - position: (x, y) tuple OR entrance_polygon: Polygon for spawn location
                - platform_polygon: Polygon for platform destination
                - journey_id and stage_id OR create unique waypoint
                
        Returns:
            True if spawn successful
        """
        try:
            # Get or generate spawn position
            if 'position' in spawn_params:
                position = spawn_params['position']
            elif 'entrance_polygon' in spawn_params:
                # Generate a single position from the entrance polygon
                entrance_polygon = spawn_params['entrance_polygon']
                positions = jps.distribute_by_number(
                    polygon=entrance_polygon,
                    number_of_agents=1,
                    distance_to_agents=0.5,
                    distance_to_polygon=0.3,
                    seed=None  # Random seed for variety
                )
                position = positions[0]
            else:
                raise ValueError("spawn_params must contain either 'position' or 'entrance_polygon'")
            
            # Generate unique waypoint position for this agent within their platform
            if 'platform_polygon' in spawn_params:
                platform_polygon = spawn_params['platform_polygon']
                # Generate a random position within the platform for this agent's waypoint
                waypoint_positions = jps.distribute_by_number(
                    polygon=platform_polygon,
                    number_of_agents=1,
                    distance_to_agents=0.3,
                    distance_to_polygon=0.2,
                    seed=None
                )
                waypoint_pos = waypoint_positions[0]
                
                # Create unique waypoint stage for this agent
                stage_id = self.simulation.add_waypoint_stage(waypoint_pos, distance=2.0)
                
                # Create unique journey for this agent
                journey = jps.JourneyDescription([stage_id])
                journey_id = self.simulation.add_journey(journey)
            else:
                # Use provided journey and stage
                journey_id = spawn_params['journey_id']
                stage_id = spawn_params.get('stage_id')
            
            # Add agent to JuPedSim with journey
            jps_id = self.simulation.add_agent(
                jps.CollisionFreeSpeedModelAgentParameters(
                    position=position,
                    journey_id=journey_id,
                    stage_id=stage_id,
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
        Get agent's current position from JuPedSim.
        
        Args:
            agent: StationAgent with jps_agent_id
            
        Returns:
            Dictionary with position (x, y) tuple
        """
        if not hasattr(agent, 'jps_agent_id'):
            return {'position': (0.0, 0.0)}
        
        try:
            position = self.simulation.agent_position(agent.jps_agent_id)
            return {'position': (position[0], position[1])}
        except:
            return {'position': agent.position if hasattr(agent, 'position') else (0.0, 0.0)}
    
    def reroute_to_evacuation_exit(self, agent: Any, exit_name: str) -> bool:
        """
        Reroute agent to a specific evacuation exit.
        
        Switches the agent's journey to one that leads directly to the named exit.
        Requires that evacuation journeys have been configured via the
        evacuation_journeys dictionary.
        
        Args:
            agent: StationAgent to reroute
            exit_name: Name of evacuation exit to route to
            
        Returns:
            True if successfully rerouted, False otherwise
        
        Args:
            agent: StationAgent to reroute
            exit_name: Name of the evacuation exit entrance
            
        Returns:
            True if successfully rerouted
        """
        if not hasattr(agent, 'jps_agent_id'):
            return False
        
        # Get the evacuation journey and exit stage for this exit
        if exit_name not in self.evacuation_journeys:
            print(f"Warning: No evacuation journey found for exit '{exit_name}'")
            return False
        
        journey_id = self.evacuation_journeys[exit_name]
        stage_id = self.evacuation_exits[exit_name]
        
        try:
            # Switch agent to evacuation journey with the exit stage
            self.simulation.switch_agent_journey(
                agent.jps_agent_id,
                journey_id,
                stage_id
            )
            return True
        except Exception as e:
            print(f"Failed to reroute agent {agent.id} to {exit_name}: {e}")
            return False
