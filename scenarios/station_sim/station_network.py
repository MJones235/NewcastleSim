"""
Station network metadata - tracks entrances, exits, and platform access points.
"""

import xml.etree.ElementTree as ET
import random
from typing import Dict, List, Optional


class StationNetwork:
    """
    Manages station infrastructure metadata including entrances, exits, and platform access.
    Provides methods to query and route between different station locations.
    """
    
    def __init__(self, stops_file: str):
        """
        Initialize station network metadata.
        
        Args:
            stops_file: Path to osm_stops.add.xml file
        """
        # Entrance/exit edges (hardcoded station boundaries)
        self.entrance_edges = ['258625111', 'E8', '1078920102']
        self.exit_edges = ['258625111', 'E8', '1078920102'] 
        
        # Spawn positions for each entrance edge (0.0 = start, -1 = end of edge)
        self.entrance_spawn_positions: Dict[str, float] = {
            '258625111': 0.0,
            'E8': -1.0,
            '1078920102': 0.0
        }
        
        # Platform access mapping: {busStop_id: access_edge_id}
        self.platform_access: Dict[str, str] = {}
        
        # Load platform access information from stops file
        self._load_platform_access(stops_file)
    
    def _load_platform_access(self, stops_file: str):
        """Parse osm_stops.add.xml to extract platform access edges"""
        try:
            tree = ET.parse(stops_file)
            root = tree.getroot()
            
            # Find all busStops with access elements
            for busstop in root.findall('.//busStop'):
                busstop_id = busstop.get('id')
                access = busstop.find('access')
                
                if access is not None and busstop_id:
                    access_lane = access.get('lane')
                    if access_lane:
                        # Extract edge from lane (lane format is typically "edge_id_laneIndex")
                        access_edge = access_lane.rsplit('_', 1)[0]
                        self.platform_access[busstop_id] = access_edge
            
            print(f"Loaded {len(self.platform_access)} platform access mappings")
            
        except Exception as e:
            print(f"Warning: Could not load platform access data: {e}")
    
    def get_random_entrance_edge(self) -> str:
        """Get a random entrance edge"""
        return random.choice(self.entrance_edges)    
    def get_entrance_spawn_position(self, edge_id: str) -> float:
        """Get the spawn position for an entrance edge (0.0=start, -1=end)"""
        return self.entrance_spawn_positions.get(edge_id, -1.0)    
    def get_random_exit_edge(self) -> str:
        """Get a random exit edge"""
        return random.choice(self.exit_edges)
    
    def get_platform_access_edge(self, busstop_id: str) -> Optional[str]:
        """
        Get the access edge for a specific platform busStop.
        
        Args:
            busstop_id: The busStop ID
            
        Returns:
            The access edge ID, or None if not found
        """
        return self.platform_access.get(busstop_id)
    
    def get_all_platforms(self) -> List[str]:
        """Get list of all platform busStop IDs"""
        return list(self.platform_access.keys())
    
    def get_random_platform(self) -> str:
        """Get a random platform busStop ID"""
        return random.choice(list(self.platform_access.keys()))
    
    def __repr__(self):
        return (f"StationNetwork(entrances={len(self.entrance_edges)}, "
                f"exits={len(self.exit_edges)}, "
                f"platforms={len(self.platform_access)})")
