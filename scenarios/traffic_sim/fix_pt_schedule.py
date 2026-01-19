#!/usr/bin/env python3
"""
Fix PT schedule to run throughout the day.
Changes flows to appropriate frequencies to avoid congestion.
- Buses: Longer intervals (realistic headways)
- Trains/Metro: Keep frequent service
"""

import xml.etree.ElementTree as ET

def fix_pt_schedule(input_file, output_file, start_time=19800, end_time=86400):
    """
    Fix PT flows to run during realistic hours with appropriate frequencies.
    
    Args:
        input_file: Path to osm_pt.rou.xml
        output_file: Path to save fixed file
        start_time: Service start time in seconds (default 5:30 AM = 19800s)
        end_time: Service end time in seconds (default 24:00 = 86400s)
    """
    # Parse XML
    tree = ET.parse(input_file)
    root = tree.getroot()
    
    flow_count = 0
    bus_count = 0
    train_count = 0
    
    # Find all flow elements
    for flow in root.findall('flow'):
        vehicle_type = flow.get('type', '')
        current_period = float(flow.get('period', 600))
        
        # Set begin time to start of service
        flow.set('begin', str(start_time))
        
        # Set end time to end of service
        flow.set('end', str(end_time))
        
        # Adjust periods based on vehicle type to reduce congestion
        if 'bus' in vehicle_type:
            # Buses: 30-40 minute headways (1800-2400 seconds)
            # This prevents too many buses stacking up
            new_period = max(1800, current_period * 3)  # At least 30 min between buses
            flow.set('period', str(int(new_period)))
            bus_count += 1
        elif 'train' in vehicle_type or 'rail' in vehicle_type:
            # Trains/Metro: 15-20 minute headways (more realistic for regional rail)
            new_period = max(900, current_period * 1.5)  # At least 15 min between trains
            flow.set('period', str(int(new_period)))
            train_count += 1
        
        flow_count += 1
    # Write modified XML
    tree.write(output_file, encoding='UTF-8', xml_declaration=True)
    
    print(f"Fixed {flow_count} PT flows")
    print(f"  - {bus_count} bus routes (30-40 min headways)")
    print(f"  - {train_count} train/metro routes (15-20 min headways)")
    print(f"Service hours: 05:30 - 24:00")
    print(f"Output written to: {output_file}")

if __name__ == "__main__":
    input_file = "scenarios/traffic_sim/network/osm_pt.rou.xml"
    output_file = "scenarios/traffic_sim/network/osm_pt.rou.xml"
    
    start_time = 5.5 * 3600  # 5:30 AM
    end_time = 24 * 3600     # Midnight
    
    print(f"Fixing PT schedule in {input_file}...")
    fix_pt_schedule(input_file, output_file, start_time=start_time, end_time=end_time)
    print("\nDone! Buses will spawn less frequently to avoid congestion.")
