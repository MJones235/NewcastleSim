"""
Load pedestrian population and place them within station walking areas.
"""

import xml.etree.ElementTree as ET
import random
from typing import List, Tuple
from shapely.geometry import Polygon, Point
from agent import StationAgent


class PopulationLoader:
    """
    Loads and creates station agents, placing them randomly within walking areas.
    """
    
    def __init__(self, walking_areas_file: str):
        """
        Initialize population loader.
        
        Args:
            walking_areas_file: Path to walking_areas.add.xml file
        """
        self.walking_areas_file = walking_areas_file
        self.walking_areas = []
        self._load_walking_areas()
    
    def _load_walking_areas(self):
        """Parse walking areas from XML file"""
        try:
            tree = ET.parse(self.walking_areas_file)
            root = tree.getroot()
            
            for poly in root.findall('poly'):
                poly_type = poly.get('type')
                if poly_type == 'jupedsim.walkable_area':
                    area_id = poly.get('id')
                    area_name = poly.get('name', area_id)
                    shape_str = poly.get('shape')
                    
                    # Parse shape coordinates and create Polygon
                    coords = self._parse_shape(shape_str)
                    polygon = Polygon(coords)
                    
                    self.walking_areas.append({
                        'id': area_id,
                        'name': area_name,
                        'polygon': polygon
                    })
            
            print(f"Loaded {len(self.walking_areas)} walking areas")
            for area in self.walking_areas:
                print(f"  - {area['name']}: {len(area['polygon'].exterior.coords)} vertices")
                
        except Exception as e:
            print(f"Error loading walking areas: {e}")
    
    def _parse_shape(self, shape_str: str) -> List[Tuple[float, float]]:
        """Parse shape string into list of (x, y) coordinates"""
        coords = []
        points = shape_str.strip().split()
        
        for point_str in points:
            parts = point_str.split(',')
            if len(parts) == 2:
                x = float(parts[0])
                y = float(parts[1])
                coords.append((x, y))
        
        return coords
    
    def _random_point_in_polygon(self, polygon: Polygon) -> Tuple[float, float]:
        """
        Generate a random point inside a polygon using rejection sampling.
        """
        min_x, min_y, max_x, max_y = polygon.bounds
        
        max_attempts = 1000
        for _ in range(max_attempts):
            point = Point(random.uniform(min_x, max_x), random.uniform(min_y, max_y))
            if polygon.contains(point):
                return (point.x, point.y)
        
        # Fallback: return centroid
        return (polygon.centroid.x, polygon.centroid.y)
    
    def create_agents(self, num_agents: int) -> List[StationAgent]:
        """
        Create agents at entrance nodes with platform destinations.
        
        Args:
            num_agents: Number of agents to create (currently only creates 1)
            
        Returns:
            List of StationAgent objects
        """
        agents = []
        
        # Fixed routing: entrance junction 3608883591, destination busStop 4270733515
        entrance = '3608883591'
        destination = '4270733515'
        
        # Use dummy position for now - will be set when spawning
        agent = StationAgent(
            agent_id=f"agent_0",
            start_position=(0.0, 0.0),
            destination=destination,
            destination_type="platform"
        )
        
        # Store entrance node for spawning
        agent.entrance_node = entrance
        
        agents.append(agent)
        print(f"Created 1 agent entering at junction {entrance}, going to busStop {destination}")
        
        return agents
