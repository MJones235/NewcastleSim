#!/usr/bin/env python3
"""Quick test of JuPedSim integration for debugging."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scenarios.station_concordia.core.jupedsim_integration import (  # noqa: E402
    ConcordiaJuPedSimulation,
)

# Test basic integration
network_path = Path("scenarios/station_sim/network")

print("Testing JuPedSim integration...")
print(f"Network path: {network_path}")
print()

try:
    jps_sim = ConcordiaJuPedSimulation(
        network_path=network_path,
        dt=0.05,
        exit_radius=10.0,
    )

    print("\n✓ Simulation created successfully")
    print(f"  - Walkable areas: {len(jps_sim.walkable_areas)}")
    print(f"  - Entrance areas: {len(jps_sim.entrance_areas)}")
    print(f"  - Evacuation exits: {len(jps_sim.evacuation_exits)}")
    print(f"  - Evacuation journeys: {len(jps_sim.evacuation_journeys)}")

    if jps_sim.evacuation_exits:
        print(f"\nExit IDs: {jps_sim.evacuation_exits}")
        print(f"Journey IDs: {jps_sim.evacuation_journeys}")

        # Try adding an agent at a valid position (use centroid of walkable area)
        print("\n� Testing agent addition...")
        main_area = list(jps_sim.walkable_areas.values())[0]
        centroid = main_area.centroid
        test_pos = (centroid.x, centroid.y)
        print(f"Using centroid position: {test_pos}")

        success = jps_sim.add_agent("test_agent", test_pos)
        if success:
            print("✓ Agent added successfully")

            # Run a few steps
            print("\nRunning 10 simulation steps...")
            for i in range(10):
                jps_sim.step()
                pos = jps_sim.get_agent_position("test_agent")
                print(f"  Step {i}: position = {pos}")

        else:
            print("✗ Failed to add agent")
    else:
        print("\n✗ No evacuation exits created!")

except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback

    traceback.print_exc()
