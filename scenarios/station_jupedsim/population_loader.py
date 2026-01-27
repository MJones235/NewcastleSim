"""
Population loader for JuPedSim station simulation.
"""

import jupedsim as jps
import numpy as np
from typing import List
from pathlib import Path
import sys

# Import from parent modules
sys.path.append(str(Path(__file__).parent.parent))
from base.decision_maker_base import DecisionMakerBase
from station_sim.rule_based_decision_maker import RuleBasedDecisionMaker
from station_sim import decision_maker_configs

try:
    from .agent import StationAgent, sample_walking_speed
except ImportError:
    from agent import StationAgent, sample_walking_speed


def create_agents_in_zone(
    simulation: jps.Simulation,
    zone_name: str,
    zone_polygon,
    num_agents: int,
    journey_id: int,
    stage_id: int,
    agent_list: List[StationAgent],
    zones_with_obstacles: dict = None
) -> None:
    """
    Create agents in a specific zone.
    
    Args:
        simulation: JuPedSim simulation object
        zone_name: Name of the zone
        zone_polygon: Shapely polygon defining the zone
        num_agents: Number of agents to create
        journey_id: Journey ID for these agents
        stage_id: Initial stage ID
        agent_list: List to append created StationAgent objects to
        zones_with_obstacles: Optional dict of zone name -> polygon with obstacles cut out
    """
    # Distribute agent positions within the zone
    # Use zone polygon with obstacles removed for better distribution
    if zones_with_obstacles and zone_name in zones_with_obstacles:
        distribution_polygon = zones_with_obstacles[zone_name]
    else:
        distribution_polygon = zone_polygon
    
    positions = jps.distribute_by_number(
        polygon=distribution_polygon,
        number_of_agents=num_agents,
        distance_to_agents=0.5,
        distance_to_polygon=0.5,  # Increased to avoid boundary issues with obstacles
        seed=42
    )
    
    # Create each agent
    for pos in positions:
        # Sample walking speed
        walking_speed = sample_walking_speed()
        
        # Create decision maker (random evacuation probability)
        evac_prob = np.random.uniform(0.3, 0.7)
        config = {'evacuation_probability': evac_prob}
        decision_maker = RuleBasedDecisionMaker(config=config)
        
        # Create JuPedSim agent parameters
        agent_params = jps.CollisionFreeSpeedModelAgentParameters(
            journey_id=journey_id,
            stage_id=stage_id,
            position=pos,
            v0=walking_speed,  # Desired speed
            radius=0.3  # Agent radius
        )
        
        # Add to simulation
        jps_agent_id = simulation.add_agent(agent_params)
        
        # Create our wrapper
        agent_id = len(agent_list)
        agent = StationAgent(
            agent_id=agent_id,
            jps_agent_id=jps_agent_id,
            walking_speed=walking_speed,
            decision_maker=decision_maker,
            initial_zone=zone_name
        )
        
        agent_list.append(agent)
        
    print(f"Created {num_agents} agents in zone '{zone_name}'")


if __name__ == "__main__":
    print("Population loader module ready")
