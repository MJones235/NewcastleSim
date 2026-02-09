"""
Specialized agent behaviors for evacuation scenarios.

This package contains domain-specific behavior coordinators:
- Message system: Agent-to-agent communication with memory and deduplication
- Helping coordinator: Helper-injured agent coordination and synchronization
"""

from scenarios.station_concordia.behaviors.message_system import MessageSystem

__all__ = ["MessageSystem"]
