"""JuPedSim integration layer for Station Concordia.

This package provides the interface and implementation for pedestrian
simulation backends. The PedestrianSimulation protocol can be satisfied
by any simulation engine, not just JuPedSim.
"""

from scenarios.station_concordia.jps_integration.jupedsim_integration import (
    ConcordiaJuPedSimulation,
)
from scenarios.station_concordia.jps_integration.simulation_interface import PedestrianSimulation

__all__ = ["PedestrianSimulation", "ConcordiaJuPedSimulation"]
