"""
Setup and configuration for JuPedSim station simulation.
Handles geometry loading, exit creation, and platform stage setup.
"""

import jupedsim as jps
from pathlib import Path
from shapely.geometry import Point
from typing import Dict, Tuple

from simulation import StationSimulation
from geometry import load_entrance_areas, load_platform_areas


def setup_evacuation_exits(
    sim: StationSimulation,
    entrance_areas: Dict[str, any]
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Create evacuation exit stages at entrance locations.
    
    Args:
        sim: StationSimulation instance
        entrance_areas: Dictionary of entrance name -> polygon
        
    Returns:
        Tuple of (evacuation_exits, evacuation_journeys) dictionaries
    """
    print("\n[2/5] Setting up evacuation exits at entrances...")
    evacuation_exits = {}
    evacuation_journeys = {}
    
    for entrance_name, entrance_polygon in entrance_areas.items():
        # JuPedSim exits must be convex polygons
        # Create a circular (convex) exit at the entrance center
        point = entrance_polygon.representative_point()
        position = (point.x, point.y)
        
        # Create circular exit (circles are convex)
        exit_polygon = Point(position).buffer(10.0)  # 10m radius exit area
        
        exit_id = sim.simulation.add_exit_stage(polygon=exit_polygon)
        evacuation_exits[entrance_name] = exit_id
        
        # Create journey to this exit
        journey = jps.JourneyDescription([exit_id])
        journey_id = sim.simulation.add_journey(journey)
        evacuation_journeys[entrance_name] = journey_id
        
        print(f"  Created evacuation exit at '{entrance_name}' (exit_id={exit_id}, journey_id={journey_id})")
    
    return evacuation_exits, evacuation_journeys


def setup_platform_stages(
    sim: StationSimulation,
    platform_areas: Dict[str, any]
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Create waiting stages and journeys for each platform.
    
    Args:
        sim: StationSimulation instance
        platform_areas: Dictionary of platform name -> polygon
        
    Returns:
        Tuple of (platform_stages, platform_journeys) dictionaries
    """
    print("\n[3/5] Setting up platform stages...")
    platform_stages = {}
    platform_journeys = {}
    
    for platform_name, platform_polygon in platform_areas.items():
        # Get representative point (guaranteed to be inside polygon, unlike centroid)
        point = platform_polygon.representative_point()
        position = (point.x, point.y)
        
        # Try to create waiting stage, skip if position is outside walkable area
        try:
            stage_id = sim.stage_manager.create_waiting_stage(
                name=platform_name,
                position=position
            )
            platform_stages[platform_name] = stage_id
            
            # Create journey for this platform (single-stage journey to the waypoint)
            journey = jps.JourneyDescription([stage_id])
            journey_id = sim.simulation.add_journey(journey)
            platform_journeys[platform_name] = journey_id
            
            print(f"  Created waiting stage for platform '{platform_name}' (stage_id={stage_id}, journey_id={journey_id})")
        except Exception as e:
            print(f"  Warning: Skipped platform '{platform_name}' - {e}")
    
    if not platform_stages:
        raise RuntimeError("No valid platform stages created!")
    
    return platform_stages, platform_journeys


def load_geometry(network_path: Path) -> Tuple[Dict, Dict]:
    """
    Load entrance and platform geometry from network files.
    
    Args:
        network_path: Path to network directory
        
    Returns:
        Tuple of (entrance_areas, platform_areas) dictionaries
    """
    walking_areas_file = network_path / "walking_areas.add.xml"
    entrance_areas = load_entrance_areas(str(walking_areas_file))
    platform_areas = load_platform_areas(str(walking_areas_file))
    print(f"Loaded {len(entrance_areas)} entrances, {len(platform_areas)} platforms")
    
    return entrance_areas, platform_areas
