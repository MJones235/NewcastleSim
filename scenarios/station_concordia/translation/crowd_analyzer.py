"""
Crowd behavior analysis for observation generation.

Analyzes agent behaviors, movement patterns, and exit crowd densities.
"""

from typing import Any

from scenarios.common.logger import get_logger

logger = get_logger(__name__)


class CrowdAnalyzer:
    """
    Analyzes crowd behavior and movement patterns.

    Handles:
    - Behavior summarization (injured, helping, moving, waiting)
    - Exit crowd density counting
    - Movement pattern categorization
    - Crowd density classification
    """

    def __init__(self, exits: dict[str, tuple[float, float]]):
        """
        Initialize crowd analyzer.

        Args:
            exits: Dictionary mapping exit names to coordinates
        """
        self.exits = exits

    def summarize_behaviors(
        self,
        nearby_agents: list[dict[str, Any]],
        agent_injured: set[str],
        helping_relationships,
    ) -> str:
        """
        Summarize what nearby agents are doing using three-dimensional model.

        Args:
            nearby_agents: List of nearby agent info dictionaries
            agent_injured: Set of injured agent IDs
            helping_relationships: HelpingRelationships tracker

        Returns:
            Natural language summary of behaviors
        """
        # Detect injured/slow-moving agents
        injured_nearby = []
        helping_nearby = []

        for agent in nearby_agents:
            agent_id = agent.get("id")
            if agent_id:
                distance = agent.get("distance", 999)

                # Check if injured (physical capability dimension)
                if agent_id in agent_injured and distance < 20.0:
                    injured_nearby.append(agent_id)

                # Check if helping (social relationship dimension)
                if (
                    helping_relationships
                    and helping_relationships.is_helping(agent_id)
                    and distance < 20.0
                ):
                    helping_nearby.append(agent_id)

        # Build behavior summary
        parts = []

        # Count movement patterns
        moving_count = sum(1 for a in nearby_agents if a.get("is_moving", True))
        waiting_count = len(nearby_agents) - moving_count

        if moving_count > waiting_count:
            parts.append("Most people are moving toward exits.")
        elif waiting_count > moving_count:
            parts.append("Many people are waiting or stationary.")
        else:
            parts.append("People are mixed between moving and waiting.")

        # Note injured agents nearby
        if injured_nearby:
            if len(injured_nearby) == 1:
                parts.append(
                    f"You notice {injured_nearby[0]} appears injured or moving very slowly."
                )
            else:
                parts.append(
                    f"You notice {len(injured_nearby)} people nearby appear injured or moving very slowly: {', '.join(injured_nearby[:3])}"
                )

        if helping_nearby:
            parts.append("Someone nearby is helping another person.")

        return " ".join(parts)

    def count_agents_per_exit(self, nearby_agents: list[dict[str, Any]]) -> dict[str, int]:
        """
        Count how many nearby agents appear to be heading toward each exit.

        Args:
            nearby_agents: List of nearby agent info dictionaries

        Returns:
            Dict mapping exit name to approximate agent count
        """
        exit_counts: dict[str, int] = {}

        for agent in nearby_agents:
            target_exit = agent.get("target_exit")
            if target_exit and target_exit in self.exits:
                exit_counts[target_exit] = exit_counts.get(target_exit, 0) + 1

        return exit_counts

    @staticmethod
    def categorize_density(num_nearby: int) -> str:
        """
        Categorize crowd density.

        Args:
            num_nearby: Number of nearby agents

        Returns:
            Density category string
        """
        if num_nearby == 0:
            return "empty (no one nearby)"
        elif num_nearby <= 3:
            return "sparse (a few people nearby)"
        elif num_nearby <= 10:
            return "moderate crowd nearby"
        else:
            return "crowded (many people nearby)"

    @staticmethod
    def categorize_count(count: int) -> str:
        """
        Categorize people count to prevent minor changes from triggering LLM.

        Args:
            count: Number of people

        Returns:
            Count category string
        """
        if count == 0:
            return "empty"
        elif count <= 3:
            return "sparse (few people)"
        elif count <= 10:
            return "moderate crowd"
        else:
            return "crowded (many people)"

    @staticmethod
    def analyze_movement_pattern(nearby_agents: list[dict[str, Any]]) -> str:
        """
        Analyze overall movement pattern of nearby agents.

        Args:
            nearby_agents: List of nearby agent info dictionaries

        Returns:
            Movement pattern description
        """
        if not nearby_agents:
            return ""

        moving_count = sum(1 for a in nearby_agents if a.get("is_moving", True))
        moving_pct = (moving_count / len(nearby_agents)) * 100

        if moving_pct > 70:
            return "Most people around you are moving purposefully toward exits."
        elif moving_pct > 40:
            return "The crowd is mixed - some evacuating, others waiting or uncertain."
        else:
            return "Many people around you are waiting or looking for information."
