"""
Station Concordia - Evacuation simulation using Concordia for decision-making.

This scenario demonstrates using Google DeepMind's Concordia library for agent
decision-making in an evacuation simulation, with JuPedSim handling movement.

Architecture:
- Concordia: Agent cognition, memory, decision-making
- JuPedSim: Pedestrian movement simulation
- Translation Layer: Bridges the two systems

Directory Structure:
- coordination/: Main simulation orchestration
- jps_integration/: JuPedSim integration layer
- concordia_integration/: Concordia/LLM integration
- decision/: Decision-making and action execution
- systems/: Specialized systems (helping, events, messaging)
- translation/: Concordia ↔ JuPedSim translation
- config/: Configuration files
- setup/: Initialization and factories
- reporting/: Results and analytics
- visualization/: Viewer launching
- utils/: Utilities
"""

__version__ = "0.1.0"
