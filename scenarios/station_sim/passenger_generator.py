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
        self.exit_edge = "1078920102"  # Main pedestrian exit edge leading to station exit
        # Edges in/near the station concourse (known to be in JuPedSim walking area)
        self.outside_exit_edges = ['1112372126#0', '1112372126#1', '540275676#8', '540275676#9', '1078920102']
        self.entry_edges = ['631597724#0']  # Where passengers start before entering station
        
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
                    
                    # Get pedestrian access points if they exist
                    for access in stop.findall('access'):
                        stop_info['access_lanes'].append({
                            'lane': access.get('lane'),
                            'pos': float(access.get('pos', 0)),
                            'length': float(access.get('length', 10))
                        })
                    
                    # Add all train stops, even those without access (we'll use fallback routing)
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
        person_id = 0
        passengers = []  # Collect all passengers first
        
        # For each train arrival, create passengers
        for train in self.train_schedules:
            for stop_info in train['stops']:
                # Get the platform stop
                platform = next((s for s in self.train_stops if s['id'] == stop_info['id']), None)
                if not platform:
                    continue
                
                # Arrival time is when train arrives + random offset for alighting
                arrival_time = stop_info['arrival']
                
                # Create passengers for this train
                for i in range(passengers_per_train):
                    person_id += 1
                    
                    # Person departs slightly after train arrives (random spread)
                    depart_time = arrival_time + random.uniform(10, 60)
                    
                    # Start from an edge in the walking area (not from platform)
                    # Pick a random edge in the concourse area
                    start_edge = random.choice(['540275676#8', '540275676#9', '1112372126#0'])
                    
                    # Walk through walking area to exit
                    outside_dest = random.choice(self.outside_exit_edges)
                    
                    passengers.append({
                        'id': f'arriving_{person_id}',
                        'depart': depart_time,
                        'from': start_edge,  # Start from walking area edge
                        'to': outside_dest
                    })
        
        # Sort passengers by departure time (REQUIRED by SUMO)
        passengers.sort(key=lambda p: p['depart'])
        
        # Build XML
        root = ET.Element('routes')
        root.set('xmlns:xsi', 'http://www.w3.org/2001/XMLSchema-instance')
        root.set('xsi:noNamespaceSchemaLocation', 'http://sumo.dlr.de/xsd/routes_file.xsd')
        
        # Define person type
        vtype = ET.SubElement(root, 'vType')
        vtype.set('id', 'passenger')
        vtype.set('vClass', 'pedestrian')
        
        # Add sorted passengers
        for pax in passengers:
            person = ET.SubElement(root, 'person')
            person.set('id', pax['id'])
            person.set('depart', f"{pax['depart']:.1f}")
            person.set('type', 'passenger')
            
            walk = ET.SubElement(person, 'walk')
            walk.set('from', pax['from'])  # Start from platform edge
            walk.set('to', pax['to'])
                    
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
        person_id = 0
        passengers = []  # Collect all passengers first
        
        # For each train departure, create passengers
        for train in self.train_schedules:
            for stop_info in train['stops']:
                # Get the platform stop
                platform = next((s for s in self.train_stops if s['id'] == stop_info['id']), None)
                if not platform:
                    continue
                
                # Train departure time
                departure_time = stop_info['until']
                
                # Create passengers for this train
                for i in range(passengers_per_train):
                    person_id += 1
                    
                    # Person arrives at station before train (random within window)
                    arrival_time = departure_time - random.uniform(60, arrival_window)
                    if arrival_time < 0:
                        continue
                    
                    # Choose random entry point outside station
                    entry_edge = random.choice(self.entry_edges)
                    
                    passengers.append({
                        'id': f'departing_{person_id}',
                        'depart': arrival_time,
                        'from': entry_edge,
                        'busStop': platform['id']
                    })
        
        # Sort passengers by departure time (REQUIRED by SUMO)
        passengers.sort(key=lambda p: p['depart'])
        
        # Build XML
        root = ET.Element('routes')
        root.set('xmlns:xsi', 'http://www.w3.org/2001/XMLSchema-instance')
        root.set('xsi:noNamespaceSchemaLocation', 'http://sumo.dlr.de/xsd/routes_file.xsd')
        
        # Add sorted passengers
        for pax in passengers:
            person = ET.SubElement(root, 'person')
            person.set('id', pax['id'])
            person.set('depart', f"{pax['depart']:.1f}")
            person.set('type', 'passenger')
            
            # Walk from outside station entrance to platform
            walk = ET.SubElement(person, 'walk')
            walk.set('from', pax['from'])
            walk.set('busStop', pax['busStop'])
            
            # Wait for train at platform busStop
            stop = ET.SubElement(person, 'stop')
            stop.set('busStop', pax['busStop'])
            stop.set('duration', '60')
        
        # Write XML
        tree = ET.ElementTree(root)
        ET.indent(tree, space='    ')
        tree.write(output_file, encoding='UTF-8', xml_declaration=True)
        
        print(f"Generated {person_id} departing passengers")
        print(f"Output: {output_file}")


if __name__ == "__main__":
    # Example usage
    generator = PassengerGenerator(
        network_file='scenarios/station_sim/network/net_with_platforms.net.xml',
        stops_file='scenarios/station_sim/network/osm_stops.add.xml',
        pt_routes_file='scenarios/station_sim/network/osm_pt.rou.xml'
    )
    
    # Generate both types of passengers
    generator.generate_arriving_passengers(passengers_per_train=50)
    generator.generate_departing_passengers(passengers_per_train=50)
