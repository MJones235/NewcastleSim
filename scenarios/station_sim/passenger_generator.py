#!/usr/bin/env python3
"""
Generate passengers for train station simulation.
Creates person trips for:
- Passengers arriving on trains and exiting the station
- Passengers entering the station and boarding trains
"""

import xml.etree.ElementTree as ET
from typing import List, Tuple
import random


class PassengerGenerator:
    """Generates passenger trips for train station simulation."""
    
    def __init__(self, network_file: str, stops_file: str, pt_routes_file: str):
        """
        Initialize the passenger generator.
        
        Args:
            network_file: Path to net.net.xml
            stops_file: Path to osm_stops.add.xml
            pt_routes_file: Path to osm_pt.rou.xml
        """
        self.network_file = network_file
        self.stops_file = stops_file
        self.pt_routes_file = pt_routes_file
        # Entry/exit edges at station boundary - JuPedSim routes through walkable area
        self.exit_edge = "1078920102"  # Main station exit
        self.entrance_edge = "1078920102"  # Main station entrance
        
        # Parse train stops
        self.train_stops = self._parse_train_stops()
        print(f"Found {len(self.train_stops)} train platforms")
        
        # Parse train schedules
        self.train_schedules = self._parse_train_schedules()
        print(f"Found {len(self.train_schedules)} train services")
    
    def _parse_train_stops(self) -> List[dict]:
        """Parse train platforms from stops file."""
        tree = ET.parse(self.stops_file)
        root = tree.getroot()
        
        train_stops = []
        for stop in root.findall('busStop'):
            name = stop.get('name', '')
            if 'Newcastle' in name:
                lines = stop.get('lines', '')
                # Filter for train stops (not buses or metro)
                if any(prefix in lines for prefix in ['GR', 'NR', 'XC', 'TP']):
                    stop_info = {
                        'id': stop.get('id'),
                        'name': name,
                        'lane': stop.get('lane'),
                        'lines': lines,
                        'access_lanes': []
                    }
                    
                    # Get pedestrian access points
                    for access in stop.findall('access'):
                        stop_info['access_lanes'].append({
                            'lane': access.get('lane'),
                            'pos': float(access.get('pos', 0))
                        })
                    
                    train_stops.append(stop_info)
        
        return train_stops
    
    def _parse_train_schedules(self) -> List[dict]:
        """Parse train schedules from PT routes file."""
        tree = ET.parse(self.pt_routes_file)
        root = tree.getroot()
        
        # First, build a map of route_id -> stops
        route_stops = {}
        for route in root.findall('route'):
            route_id = route.get('id')
            stops = []
            for stop in route.findall('stop'):
                stop_id = stop.get('busStop')
                # Only include Newcastle station stops
                if any(s['id'] == stop_id for s in self.train_stops):
                    stops.append({
                        'id': stop_id,
                        'until': float(stop.get('until', 0)),
                        'duration': float(stop.get('duration', 180))
                    })
            if stops:
                route_stops[route_id] = stops
        
        # Now parse flows and expand them into individual train arrivals
        schedules = []
        for flow in root.findall('flow'):
            flow_type = flow.get('type', '')
            if 'train' in flow_type:
                route_id = flow.get('route')
                if route_id not in route_stops:
                    continue
                
                begin = float(flow.get('begin', 0))
                end = float(flow.get('end', 86400))
                period = float(flow.get('period', 900))
                
                # Generate individual trains from the flow
                current_time = begin
                train_num = 0
                while current_time < end:
                    train_num += 1
                    schedule = {
                        'id': f"{flow.get('id')}_{train_num}",
                        'type': flow_type,
                        'depart': current_time,
                        'stops': []
                    }
                    
                    # Add stops with computed arrival times
                    for stop_info in route_stops[route_id]:
                        schedule['stops'].append({
                            'id': stop_info['id'],
                            'arrival': current_time + stop_info['until'],
                            'until': current_time + stop_info['until'] + stop_info['duration'],
                            'duration': stop_info['duration']
                        })
                    
                    schedules.append(schedule)
                    current_time += period
        
        return schedules
    
    def generate_arriving_passengers(self, 
                                     passengers_per_train: int = 50,
                                     output_file: str = 'passengers_arriving.rou.xml') -> None:
        """
        Generate passengers who arrive on trains and exit the station.
        
        Args:
            passengers_per_train: Number of passengers per train
            output_file: Output file path
        """
        root = ET.Element('routes')
        root.set('xmlns:xsi', 'http://www.w3.org/2001/XMLSchema-instance')
        root.set('xsi:noNamespaceSchemaLocation', 'http://sumo.dlr.de/xsd/routes_file.xsd')
        
        # Define person type
        vtype = ET.SubElement(root, 'vType')
        vtype.set('id', 'passenger')
        vtype.set('vClass', 'pedestrian')
        
        person_id = 0
        
        # For each train arrival, create passengers
        for train in self.train_schedules:
            for stop_info in train['stops']:
                # Get the platform stop
                platform = next((s for s in self.train_stops if s['id'] == stop_info['id']), None)
                if not platform:
                    continue
                
                # Arrival time is when train arrives + random offset for alighting
                arrival_time = stop_info['arrival']
                
                # Extract edge from platform lane (railway edge where train stops)
                platform_edge = platform['lane'].rsplit('_', 1)[0] if '_' in platform['lane'] else platform['lane']
                
                # Create passengers for this train
                for i in range(passengers_per_train):
                    person_id += 1
                    
                    # Person appears after train arrives at the busStop
                    depart_time = arrival_time + random.uniform(10, 60)
                    
                    person = ET.SubElement(root, 'person')
                    person.set('id', f'arriving_{person_id}')
                    person.set('depart', f'{depart_time:.1f}')
                    person.set('type', 'passenger')
                    
                    # Start on a random pedestrian access lane near the platform
                    # This distributes passengers across different access points
                    if platform['access_lanes']:
                        access = random.choice(platform['access_lanes'])
                        start_lane = access['lane']
                        start_edge = start_lane.rsplit('_', 1)[0] if '_' in start_lane else start_lane
                    else:
                        start_edge = self.exit_edge  # Fallback
                    
                    # Walk from access point through JuPedSim area to exit
                    walk = ET.SubElement(person, 'walk')
                    walk.set('from', start_edge)
                    walk.set('to', self.exit_edge)
                    
        # Write XML
        tree = ET.ElementTree(root)
        ET.indent(tree, space='    ')
        tree.write(output_file, encoding='UTF-8', xml_declaration=True)
        
        print(f"Generated {person_id} arriving passengers")
        print(f"Output: {output_file}")
    
    def generate_departing_passengers(self,
                                      passengers_per_train: int = 50,
                                      arrival_window: int = 600,
                                      output_file: str = 'passengers_departing.rou.xml') -> None:
        """
        Generate passengers who enter the station and board trains.
        
        Args:
            passengers_per_train: Number of passengers per train
            arrival_window: Time before train (seconds) passengers arrive at station
            output_file: Output file path
        """
        root = ET.Element('routes')
        root.set('xmlns:xsi', 'http://www.w3.org/2001/XMLSchema-instance')
        root.set('xsi:noNamespaceSchemaLocation', 'http://sumo.dlr.de/xsd/routes_file.xsd')
        
        person_id = 0
        
        # For each train departure, create passengers
        for train in self.train_schedules:
            for stop_info in train['stops']:
                # Get the platform stop
                platform = next((s for s in self.train_stops if s['id'] == stop_info['id']), None)
                if not platform:
                    continue
                
                # Train departure time
                departure_time = stop_info['arrival']
                
                # Extract edge from platform lane
                platform_edge = platform['lane'].rsplit('_', 1)[0] if '_' in platform['lane'] else platform['lane']
                
                # Create passengers for this train
                for i in range(passengers_per_train):
                    person_id += 1
                    
                    # Person arrives at station before train (random within window)
                    arrival_time = departure_time - random.uniform(60, arrival_window)
                    if arrival_time < 0:
                        continue
                    
                    person = ET.SubElement(root, 'person')
                    person.set('id', f'departing_{person_id}')
                    person.set('depart', f'{arrival_time:.1f}')
                    person.set('type', 'passenger')
                    
                    # Walk from entrance through JuPedSim area to random access point for this platform
                    # This distributes passengers across different platform access points
                    if platform['access_lanes']:
                        access = random.choice(platform['access_lanes'])
                        dest_lane = access['lane']
                        dest_edge = dest_lane.rsplit('_', 1)[0] if '_' in dest_lane else dest_lane
                    else:
                        dest_edge = self.entrance_edge  # Fallback
                    
                    walk = ET.SubElement(person, 'walk')
                    walk.set('from', self.entrance_edge)
                    walk.set('to', dest_edge)
        
        # Write XML
        tree = ET.ElementTree(root)
        ET.indent(tree, space='    ')
        tree.write(output_file, encoding='UTF-8', xml_declaration=True)
        
        print(f"Generated {person_id} departing passengers")
        print(f"Output: {output_file}")


if __name__ == "__main__":
    # Example usage
    generator = PassengerGenerator(
        network_file='scenarios/station_sim/network/net.net.xml',
        stops_file='scenarios/station_sim/network/osm_stops.add.xml',
        pt_routes_file='scenarios/station_sim/network/osm_pt.rou.xml'
    )
    
    # Generate both types of passengers
    generator.generate_arriving_passengers(passengers_per_train=50)
    generator.generate_departing_passengers(passengers_per_train=50)
