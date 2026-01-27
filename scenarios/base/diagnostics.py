"""
Diagnostic tools for understanding simulation behavior.
Generic enough to be used across different simulation types.
"""

import traci


class SimulationDiagnostics:
    """Track and report simulation metrics"""

    def __init__(self):
        self.trip_starts = 0
        self.trip_completions = 0
        self.teleports = 0
        self.failed_insertions = 0

    def report(self, sim_time: int):
        """Print diagnostic report"""
        try:
            # SUMO's built-in stats
            departed = traci.simulation.getDepartedNumber()
            arrived = traci.simulation.getArrivedNumber()
            running = traci.vehicle.getIDCount()

            hours = sim_time // 3600
            minutes = (sim_time % 3600) // 60

            print(f"\n[{hours:02d}:{minutes:02d}] Simulation Stats:")
            print("  SUMO Stats:")
            print(f"    Departed this step: {departed}")
            print(f"    Arrived this step: {arrived}")
            print(f"    Running now: {running}")
            print("  Agent Actions (cumulative):")
            print(f"    Trip starts: {self.trip_starts}")
            print(f"    Trip completions: {self.trip_completions}")
            print(f"    Teleports: {self.teleports}")
            print(f"    Failed insertions: {self.failed_insertions}")

        except Exception as e:
            print(f"Error getting diagnostics: {e}")
