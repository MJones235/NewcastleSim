"""
Core simulation components for JuPedSim station simulation.

Contains the main simulation logic, runners, and infrastructure.
"""

from .simulation import StationSimulation
from .simulation_setup import setup_evacuation_exits, setup_platform_stages, load_geometry
from .simulation_runner import SimulationRunner
from .simulation_observer import SimulationObserver, GUIObserver, ConsoleObserver
from .stage_manager import StageManager
from .event_system import EventManager, SimulationEvent
from .movement_jupedsim import JuPedSimMovementProvider
from .population_loader import create_agents_from_entrances

__all__ = [
    'StationSimulation',
    'setup_evacuation_exits',
    'setup_platform_stages',
    'load_geometry',
    'SimulationRunner',
    'SimulationObserver',
    'GUIObserver',
    'ConsoleObserver',
    'StageManager',
    'EventManager',
    'SimulationEvent',
    'JuPedSimMovementProvider',
    'create_agents_from_entrances',
]
