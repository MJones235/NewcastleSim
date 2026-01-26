"""
Load pedestrian population and place them within station walking areas.
"""

import xml.etree.ElementTree as ET
import random
import numpy as np
from typing import List, Tuple, TYPE_CHECKING
from shapely.geometry import Polygon, Point
from agent import StationAgent

if TYPE_CHECKING:
    from station_network import StationNetwork


class PopulationLoader:
        
    def create_agents(self, num_agents: int, station_network: 'StationNetwork') -> List[StationAgent]:
        """
        Create agents at random entrance edges with random platform destinations.
        
        Args:
            num_agents: Number of agents to create
            station_network: StationNetwork instance for querying entrances and platforms
            
        Returns:
            List of StationAgent objects
        """
        agents = []
        
        for i in range(num_agents):
            # Select random entrance and platform
            entrance_edge = station_network.get_random_entrance_edge()
            platform_id = station_network.get_random_platform()
            spawn_position = station_network.get_entrance_spawn_position(entrance_edge)
            
            # Compute full route from entrance to platform (handles footbridge routing)
            route = station_network.get_route(entrance_edge, platform_id)
            
            # Sample preferred walking speed from normal distribution
            # Mean: 1.34 m/s, Std Dev: 0.37 m/s
            walking_speed = np.random.normal(1.34, 0.37)
            # Ensure speed is positive and reasonable (0.5 - 2.5 m/s)
            walking_speed = max(0.5, min(2.5, walking_speed))
            
            agent = StationAgent(
                agent_id=f"agent_{i}",
                entrance_edge=entrance_edge,
                destination=platform_id,
                route=route,
                spawn_position=spawn_position,
                destination_type="platform",
                walking_speed=walking_speed
            )
            
            agents.append(agent)
            print(f"Created agent_{i}: {entrance_edge} → {platform_id} (zone {station_network.get_location_side(entrance_edge)}→{station_network.get_location_side(platform_id)})")
        
        return agents
