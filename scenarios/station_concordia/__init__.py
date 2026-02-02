"""
Station Concordia - Evacuation simulation using Concordia for decision-making.

This scenario demonstrates using Google DeepMind's Concordia library for agent
decision-making in an evacuation simulation, with JuPedSim handling movement.

Architecture:
- Concordia: Agent cognition, memory, decision-making
- JuPedSim: Pedestrian movement simulation
- Translation Layer: Bridges the two systems

Components:
- core/: Core simulation logic
- config/: Configuration files
- agents/: Custom Concordia agent prefabs
- game_master/: Custom Game Master for evacuation scenarios
"""

__version__ = "0.1.0"
