"""
Helping Relationships Tracker

Tracks social relationships between helpers and those being helped.
Does NOT control agent behavior - provides context for agent decision-making.

Philosophy: Helping is a relationship that agents navigate autonomously,
not a mechanical system that forces behavior.
"""

from typing import Any

from scenarios.common.logger import get_logger

logger = get_logger(__name__)


class HelpingRelationships:
    """
    Tracks which agents are helping others without controlling their behavior.

    Agents decide how to help based on personality and situation.
    This class just maintains relationship state for observations and visualization.
    """

    def __init__(self):
        """Initialize empty relationship tracking."""
        # helper_id -> relationship info
        self.relationships: dict[str, dict[str, Any]] = {}

    def start_helping(
        self,
        helper_id: str,
        helped_id: str,
        help_type: str,
        current_time: float,
    ) -> None:
        """
        Record that helper_id started helping helped_id.

        Args:
            helper_id: Agent providing help
            helped_id: Agent receiving help
            help_type: Style of help (wait_with, guide_to_exit, physical_assist, provide_information)
            current_time: Simulation time when helping started
        """
        self.relationships[helper_id] = {
            "helping": helped_id,
            "started": current_time,
            "help_type": help_type,
        }
        logger.info(
            f"👥 {helper_id} started helping {helped_id} (type: {help_type}, t={current_time:.1f}s)"
        )

    def stop_helping(self, helper_id: str, reason: str = "ended") -> str | None:
        """
        Stop a helping relationship.

        Args:
            helper_id: Helper ending the relationship
            reason: Why the relationship ended (ended, exited, abandoned)

        Returns:
            ID of the agent who was being helped, or None if no relationship existed
        """
        if helper_id not in self.relationships:
            return None

        relationship = self.relationships[helper_id]
        helped_id = relationship["helping"]
        help_type = relationship["help_type"]

        del self.relationships[helper_id]

        logger.info(f"👋 {helper_id} stopped helping {helped_id} ({reason}, type was: {help_type})")
        return helped_id

    def get_helper(self, helped_id: str) -> str | None:
        """
        Find who (if anyone) is helping a specific agent.

        Args:
            helped_id: Agent to check

        Returns:
            Helper's ID if someone is helping, None otherwise
        """
        for helper_id, info in self.relationships.items():
            if info["helping"] == helped_id:
                return helper_id
        return None

    def get_helped(self, helper_id: str) -> str | None:
        """
        Find who (if anyone) a helper is helping.

        Args:
            helper_id: Helper to check

        Returns:
            ID of agent being helped, or None if not helping anyone
        """
        if helper_id in self.relationships:
            return self.relationships[helper_id]["helping"]
        return None

    def is_helping(self, helper_id: str) -> bool:
        """
        Check if agent is currently helping someone.

        Args:
            helper_id: Agent to check

        Returns:
            True if currently helping someone
        """
        return helper_id in self.relationships

    def is_being_helped(self, helped_id: str) -> bool:
        """
        Check if agent is currently being helped.

        Args:
            helped_id: Agent to check

        Returns:
            True if someone is helping them
        """
        return self.get_helper(helped_id) is not None

    def get_relationship_info(self, helper_id: str) -> dict[str, Any] | None:
        """
        Get full relationship information for a helper.

        Args:
            helper_id: Helper to get info for

        Returns:
            Relationship dict with helping, started, help_type, or None
        """
        return self.relationships.get(helper_id)

    def get_all_relationships(self) -> dict[str, dict[str, Any]]:
        """
        Get all active relationships.

        Returns:
            Dict mapping helper_id to relationship info
        """
        return self.relationships.copy()

    def cleanup_exited_agents(self, exited_agents: set[str]) -> None:
        """
        Remove relationships involving agents who have exited.

        Args:
            exited_agents: Set of agent IDs who have exited the simulation
        """
        # Helpers who exited
        for helper_id in list(self.relationships.keys()):
            if helper_id in exited_agents:
                self.stop_helping(helper_id, reason="helper_exited")

        # Those being helped who exited
        for helper_id in list(self.relationships.keys()):
            helped_id = self.relationships[helper_id]["helping"]
            if helped_id in exited_agents:
                self.stop_helping(helper_id, reason="helped_exited")

    def get_statistics(self) -> dict[str, Any]:
        """
        Get statistics about helping relationships.

        Returns:
            Dict with count, help types distribution, etc.
        """
        if not self.relationships:
            return {
                "active_relationships": 0,
                "help_types": {},
            }

        help_types: dict[str, int] = {}
        for relationship in self.relationships.values():
            help_type = relationship["help_type"]
            help_types[help_type] = help_types.get(help_type, 0) + 1

        return {
            "active_relationships": len(self.relationships),
            "help_types": help_types,
        }
