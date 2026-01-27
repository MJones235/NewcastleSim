"""
Core simulation components for JuPedSim station simulation.

Contains the main simulation logic, runners, and infrastructure.
"""

from .event_system import EventManager, SimulationEvent
from .movement_jupedsim import JuPedSimMovementProvider
from .population_loader import create_agents_from_entrances
from .simulation import StationSimulation
from .simulation_observer import ConsoleObserver, GUIObserver, SimulationObserver
from .simulation_runner import SimulationRunner
from .simulation_setup import load_geometry, setup_evacuation_exits, setup_platform_stages
from .stage_manager import StageManager

__all__ = [
    "StationSimulation",
    "setup_evacuation_exits",
    "setup_platform_stages",
    "load_geometry",
    "SimulationRunner",
    "SimulationObserver",
    "GUIObserver",
    "ConsoleObserver",
    "StageManager",
    "EventManager",
    "SimulationEvent",
    "JuPedSimMovementProvider",
    "create_agents_from_entrances",
]
