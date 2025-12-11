"""
Load synthetic population and generate agent schedules.
"""

import csv
import json
import os
import random
from typing import Dict, List, Tuple
from collections import defaultdict
import sumolib
import traci
from agent import Agent, Activity, ActivityType, TransportMode


class PopulationLoader:
    """
    Loads synthetic population from CSV and creates agents with schedules.
    """
    
    def __init__(self, network_file: str):
        self.network = sumolib.net.readNet(network_file)
        self.osm_to_edge_cache: Dict[str, str] = {}
        self._fallback_edge = None
        self._route_cache: Dict[Tuple[str, str], bool] = {}
        self._load_edge_cache()
    
    def _load_edge_cache(self):
        """Load edge mapping cache from previous run"""
        cache_file = 'edge_mapping_cache.json'
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    self.osm_to_edge_cache = json.load(f)
                print(f"Loaded {len(self.osm_to_edge_cache)} cached edge mappings")
            except:
                pass
    
    def load_from_csv(self, population_file: str, max_agents: int = None) -> List[Agent]:
        """
        Load population from CSV file.
        Returns list of Agent objects with generated schedules.
        """
        agents = []
        
        try:
            with open(population_file, 'r') as f:
                reader = csv.DictReader(f)
                
                print("Loading population...")
                count = 0
                for row in reader:
                    agent = self._create_agent_from_row(row)
                    agents.append(agent)
                    
                    count += 1
                    if count % 1000 == 0:
                        print(f"  Loaded {count} agents...")
                    
                    if max_agents and count >= max_agents:
                        break
                    
            print(f"Loaded {len(agents)} agents from {population_file}")
            self.save_edge_mapping('edge_mapping_cache.json')
            
        except FileNotFoundError:
            print(f"Warning: Population file {population_file} not found")
            
        return agent
    
    def load_from_trip_csv(self, trip_file: str, max_agents: int = None) -> List[Agent]:
        """
        Load population from trip-based CSV file.
        Each person has multiple trip rows that get converted to activities.
        """
        agents = []
        person_trips = defaultdict(list)
        
        print(f"Loading trips from {trip_file}...")
        
        # Group trips by person
        with open(trip_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Skip trips with missing required fields
                if not row.get('depart_time_hours') or not row.get('depart_time_minutes'):
                    continue
                if not row.get('origin_lat') or not row.get('origin_lon'):
                    continue
                if not row.get('dest_lat') or not row.get('dest_lon'):
                    continue
                if not row.get('transport_mode'):
                    continue
                    
                person_id = f"person_{row['household_id']}_{row['person_id']}"
                person_trips[person_id].append(row)
        
        print(f"Found {len(person_trips)} people with trips")
        
        # Create agents from trips
        count = 0
        for person_id, trips in person_trips.items():
            if max_agents and count >= max_agents:
                break
            
            # Sort trips by sequence
            trips.sort(key=lambda t: int(t['trip_seq']))
            
            # Create agent
            agent = self._create_agent_from_trips(person_id, trips)
            if agent:
                agents.append(agent)
                count += 1
                
                if count % 100 == 0:
                    print(f"Loaded {count} agents...")
        
        print(f"Successfully loaded {len(agents)} agents with trip schedules")
        return agents
    
    def _create_agent_from_trips(self, agent_id: str, trips: List[Dict]) -> Agent:
        """Create an agent from trip records"""
        if not trips:
            return None
        
        # Get home location from first trip's origin
        first_trip = trips[0]
        home_osm_id = first_trip['origin_osm_id']
        home_lon = float(first_trip['origin_lon'])  
        home_lat = float(first_trip['origin_lat'])
        home_edge = self._get_edge_for_location(home_osm_id, home_lon, home_lat)
        
        # Create agent with minimal demographics
        demographics = {
            "household_id": first_trip['household_id'],
            "output_area": first_trip['output_area']
        }
        agent = Agent(agent_id, demographics, home_edge)
        
        # Convert trips to activities
        schedule = self._create_schedule_from_trips(trips, home_edge)
        for activity in schedule:
            agent.add_activity(activity)
        
        return agent
    
    def _create_schedule_from_trips(self, trips: List[Dict], home_edge: str) -> List[Activity]:
        """Convert trip data to activity schedule with route validation"""
        schedule = []
        
        # First activity: at origin from midnight until first trip
        if trips:
            first_trip = trips[0]
            first_depart = int(first_trip['depart_time_hours']) * 3600 + int(first_trip['depart_time_minutes']) * 60
            
            schedule.append(Activity(
                activity_type=self._map_location_type(first_trip['origin_type']),
                location_edge=home_edge,
                start_time=0,
                duration=first_depart,
                transport_mode=TransportMode.CAR  # Not used for first activity
            ))
            
            # Each trip creates an activity at the destination
            for i, trip in enumerate(trips):
                dest_osm_id = trip['dest_osm_id']
                dest_lon = float(trip['dest_lon'])
                dest_lat = float(trip['dest_lat'])
                dest_edge = self._get_edge_for_location(dest_osm_id, dest_lon, dest_lat)
                
                # Get transport mode
                mode_str = trip['transport_mode'].lower()
                transport_mode = self._map_transport_mode_string(mode_str)
                
                # Validate route exists for this mode
                origin_edge = schedule[-1].location if schedule else home_edge
                transport_mode = self._validate_and_adjust_mode(origin_edge, dest_edge, transport_mode)
                
                # Calculate trip start time
                depart_time = int(trip['depart_time_hours']) * 3600 + int(trip['depart_time_minutes']) * 60
                
                # Duration = time until next trip (or end of day)
                if i < len(trips) - 1:
                    next_depart = int(trips[i+1]['depart_time_hours']) * 3600 + int(trips[i+1]['depart_time_minutes']) * 60
                    duration = next_depart - depart_time
                else:
                    duration = 86400 - depart_time  # Until midnight
                
                schedule.append(Activity(
                    activity_type=self._map_location_type(trip['dest_type']),
                    location_edge=dest_edge,
                    start_time=depart_time,
                    duration=duration,
                    transport_mode=transport_mode
                ))
        
        return schedule
    
    def _map_location_type(self, location_type: str) -> ActivityType:
        """Map location type string to ActivityType"""
        mapping = {
            'home': ActivityType.HOME,
            'work': ActivityType.WORK,
            'education': ActivityType.SCHOOL,
            'shopping': ActivityType.SHOPPING,
            'entertainment': ActivityType.LEISURE,
            'leisure': ActivityType.LEISURE
        }
        return mapping.get(location_type.lower(), ActivityType.HOME)
    
    def _map_transport_mode_string(self, mode_str: str) -> TransportMode:
        """Map transport mode string to TransportMode enum"""
        if 'car' in mode_str:
            return TransportMode.CAR
        elif 'bus' in mode_str:
            return TransportMode.BUS
        elif 'bicycle' in mode_str or 'bike' in mode_str:
            return TransportMode.BICYCLE
        elif 'walk' in mode_str:
            return TransportMode.WALK
        elif 'metro' in mode_str or 'rail' in mode_str or 'train' in mode_str:
            return TransportMode.METRO
        else:
            return TransportMode.CAR  # Default
    
    def _validate_and_adjust_mode(self, from_edge: str, to_edge: str, mode: TransportMode) -> TransportMode:
        """Validate route exists for mode, try alternatives if not"""
        if from_edge == to_edge:
            return mode
        
        # Try original mode first (will be validated during simulation)
        # For now, just return the mode - actual validation happens in agent._initiate_trip
        # which will teleport if route doesn't exist
        return mode
    
    def _create_agent_from_row(self, row: Dict) -> Agent:
        """Create an agent from a CSV row"""
        agent_id = f"person_{row['household_id_m']}_{row['resident_id_m']}"
        
        demographics = {
            "age": int(row['assigned_age']),
            "sex": int(row['sex']),
            "household_size": int(row['hh_size_9a']),
            "household_cars": int(row['number_of_cars_5a']),
            "ns_sec": row['ns_sec_10a'],
            "role": row['role']
        }
        
        home_osm_id = row['home_osm_id']
        home_lon = float(row['home_longitude'])
        home_lat = float(row['home_latitude'])
        home_edge = self._get_edge_for_location(home_osm_id, home_lon, home_lat)
        
        agent = Agent(agent_id, demographics, home_edge)
        
        schedule = self._generate_schedule(row, home_edge)
        for activity in schedule:
            agent.add_activity(activity)
        
        return agent
    
    def _get_edge_for_location(self, osm_id: str, lon: float, lat: float) -> str:
        """Map OSM building ID to nearest SUMO edge"""
        if osm_id in self.osm_to_edge_cache:
            return self.osm_to_edge_cache[osm_id]
        
        try:
            x, y = self.network.convertLonLat2XY(lon, lat)
        except:
            edge_id = self._get_fallback_edge()
            self.osm_to_edge_cache[osm_id] = edge_id
            return edge_id
        
        for radius in [50, 100, 200, 500, 1000]:
            edges = self.network.getNeighboringEdges(x, y, r=radius)
            
            if edges:
                best_edge = None
                best_distance = float('inf')
                
                for e, d in edges:
                    if e.allows("passenger") and d < best_distance:
                        if len(e.getOutgoing()) > 0:
                            best_edge = e
                            best_distance = d
                            break
                        elif best_edge is None:
                            best_edge = e
                            best_distance = d
                
                if best_edge:
                    edge_id = best_edge.getID()
                    self.osm_to_edge_cache[osm_id] = edge_id
                    return edge_id
        
        edge_id = self._get_fallback_edge()
        self.osm_to_edge_cache[osm_id] = edge_id
        return edge_id
    
    def _get_fallback_edge(self) -> str:
        """Get a fallback edge"""
        if self._fallback_edge:
            return self._fallback_edge
        
        for edge in self.network.getEdges():
            if edge.allows("passenger") and len(edge.getOutgoing()) > 1:
                self._fallback_edge = edge.getID()
                return self._fallback_edge
        
        for edge in self.network.getEdges():
            if edge.allows("passenger"):
                self._fallback_edge = edge.getID()
                return self._fallback_edge
        
        self._fallback_edge = self.network.getEdges()[0].getID()
        return self._fallback_edge
    
    def _generate_schedule(self, row: Dict, home_edge: str) -> List[Activity]:
        """Generate daily schedule based on person's characteristics"""
        age = int(row['assigned_age'])
        dest_type = row['destination_type']
        
        if row['destination_osm_id'] != 'NA':
            dest_osm_id = row['destination_osm_id']
            dest_lon = float(row['destination_longitude'])
            dest_lat = float(row['destination_latitude'])
            dest_edge = self._get_edge_for_location(dest_osm_id, dest_lon, dest_lat)
            
            if not self._can_route_between(home_edge, dest_edge):
                dest_edge = home_edge
        else:
            dest_edge = home_edge
        
        activity_type = self._map_destination_to_activity(dest_type)
        
        if activity_type == ActivityType.WORK and dest_edge != home_edge:
            return self._create_work_schedule(home_edge, dest_edge, age)
        elif activity_type == ActivityType.SCHOOL and dest_edge != home_edge:
            return self._create_school_schedule(home_edge, dest_edge, age)
        else:
            return [Activity(ActivityType.HOME, home_edge, 0, 24 * 3600)]
    
    def _can_route_between(self, from_edge: str, to_edge: str) -> bool:
        """Check if a route exists between two edges"""
        if from_edge == to_edge:
            return True
        
        cache_key = (from_edge, to_edge)
        if cache_key in self._route_cache:
            return self._route_cache[cache_key]
        
        try:
            from_edge_obj = self.network.getEdge(from_edge)
            to_edge_obj = self.network.getEdge(to_edge)
            
            can_route = (from_edge_obj is not None and 
                        to_edge_obj is not None and
                        len(from_edge_obj.getOutgoing()) > 0)
            
            self._route_cache[cache_key] = can_route
            return can_route
        except:
            self._route_cache[cache_key] = False
            return False
    
    def _map_destination_to_activity(self, dest_type: str) -> ActivityType:
        """Map destination type string to ActivityType enum"""
        mapping = {
            'workplace': ActivityType.WORK,
            'school': ActivityType.SCHOOL,
            'sixth_form': ActivityType.SCHOOL,
            'university': ActivityType.SCHOOL,
            'shopping': ActivityType.SHOPPING,
            'leisure': ActivityType.LEISURE
        }
        return mapping.get(dest_type, ActivityType.HOME)
    
    def _select_transport_mode(self, age: int, distance_km: float = None) -> TransportMode:
        """Select transport mode based on age and trip characteristics"""
        # Age-based probability distributions
        if age < 16:
            # Children: mostly walk or get driven
            return random.choices(
                [TransportMode.WALK, TransportMode.CAR, TransportMode.BUS],
                weights=[0.4, 0.4, 0.2]
            )[0]
        elif age < 18:
            # Teens: more independent, use bus/walk/bike
            return random.choices(
                [TransportMode.WALK, TransportMode.BICYCLE, TransportMode.BUS, TransportMode.CAR],
                weights=[0.3, 0.2, 0.3, 0.2]
            )[0]
        elif age < 65:
            # Working age: mostly car, some bus/bike/walk
            return random.choices(
                [TransportMode.CAR, TransportMode.BUS, TransportMode.BICYCLE, TransportMode.WALK, TransportMode.METRO],
                weights=[0.65, 0.15, 0.08, 0.07, 0.05]
            )[0]
        else:
            # Seniors: mix of car, bus, walk
            return random.choices(
                [TransportMode.CAR, TransportMode.BUS, TransportMode.WALK],
                weights=[0.5, 0.3, 0.2]
            )[0]
    
    def _create_work_schedule(self, home_edge: str, work_edge: str, age: int) -> List[Activity]:
        """Create typical work schedule"""
        # Spread departures more widely: 7:00 AM to 9:30 AM (2.5 hour window)
        base_departure = 8 * 3600
        time_variation = int(random.gauss(0, 2700))  # Std dev of 45 minutes (wider spread)
        departure_time = max(7 * 3600, min(9.5 * 3600, base_departure + time_variation))
        
        work_duration = random.randint(7, 9) * 3600
        return_time = departure_time + work_duration
        
        # Select transport mode for this person
        mode = self._select_transport_mode(age)
        
        return [
            Activity(ActivityType.HOME, home_edge, 0, int(departure_time)),
            Activity(ActivityType.WORK, work_edge, int(departure_time), work_duration, mode),
            Activity(ActivityType.HOME, home_edge, int(return_time), 24 * 3600 - int(return_time), mode)
        ]
    
    def _create_school_schedule(self, home_edge: str, school_edge: str, age: int) -> List[Activity]:
        """Create typical school schedule"""
        if age < 11:
            departure_time = 8.5 * 3600 + random.randint(-300, 300)
        else:
            departure_time = 8.25 * 3600 + random.randint(-300, 300)
        
        school_duration = 6.5 * 3600
        return_time = departure_time + school_duration
        
        # Select transport mode for this child
        mode = self._select_transport_mode(age)
        
        return [
            Activity(ActivityType.HOME, home_edge, 0, int(departure_time)),
            Activity(ActivityType.SCHOOL, school_edge, int(departure_time), int(school_duration), mode),
            Activity(ActivityType.HOME, home_edge, int(return_time), int(24 * 3600 - return_time), mode)
        ]
    
    def save_edge_mapping(self, output_file: str):
        """Save OSM ID to edge mapping for future use"""
        with open(output_file, 'w') as f:
            json.dump(self.osm_to_edge_cache, f, indent=2)


