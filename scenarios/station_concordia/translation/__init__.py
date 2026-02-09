"""
Translation layer between Concordia's natural language interface and JuPedSim's geometric representation.

This package handles bidirectional translation:
- Actions: Natural language → JuPedSim waypoints/goals
- Observations: JuPedSim state → Natural language descriptions
"""

from scenarios.station_concordia.translation.action_translator import ActionTranslator
from scenarios.station_concordia.translation.observation_generator import ObservationGenerator

__all__ = ["ActionTranslator", "ObservationGenerator"]
