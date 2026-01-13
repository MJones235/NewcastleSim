#!/usr/bin/env python3
"""
Fix PT schedule for station_sim - morning rush hour only.
Sets trains to run during morning peak (7-9 AM) with realistic frequencies.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from tools.fix_pt_schedule import fix_pt_schedule

if __name__ == "__main__":
    input_file = "scenarios/station_sim/network/osm_pt.rou.xml"
    output_file = "scenarios/station_sim/network/osm_pt.rou.xml"
    
    # Morning rush hour: 7:00 AM - 9:00 AM
    start_time = 7 * 3600  # 7:00 AM
    end_time = 9 * 3600     # 9:00 AM
    
    print(f"Setting up morning rush hour schedule in {input_file}...")
    print("Time window: 07:00 - 09:00")
    fix_pt_schedule(input_file, output_file, start_time=start_time, end_time=end_time)
    print("\nMorning rush hour schedule ready!")
