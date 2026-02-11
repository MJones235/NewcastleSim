"""
Agent builder for creating and initializing Concordia agents.

This module handles the creation of Concordia agents with their memory banks
and initial knowledge state.
"""

from typing import Any

from concordia.associative_memory import basic_associative_memory
from concordia.language_model import language_model
from concordia.typing import entity as entity_lib

from scenarios.common.logger import get_logger
from scenarios.station_concordia.concordia_integration.evacuation_agent import EvacuationAgent

logger = get_logger(__name__)


class AgentBuilder:
    """Builds Concordia agents with configured memory and initial knowledge."""

    def __init__(
        self,
        language_model: language_model.LanguageModel,
        embedder: Any,
        station_layout_description: str,
    ):
        """
        Initialize the agent builder.

        Args:
            language_model: LLM for agent cognition
            embedder: Sentence embedding function
            station_layout_description: Description of station geometry for agent memory
        """
        self.model = language_model
        self.embedder = embedder
        self.station_layout_description = station_layout_description

    def build_agents(
        self, agents_config: list[dict[str, Any]]
    ) -> tuple[dict[str, entity_lib.Entity], set[str]]:
        """
        Build Concordia agents from configurations.

        Args:
            agents_config: List of agent configuration dictionaries

        Returns:
            Tuple of (concordia_agents dict, injured_agents set)
        """
        logger.info(f"Building {len(agents_config)} Concordia agents...")

        concordia_agents: dict[str, entity_lib.Entity] = {}
        injured_agents: set[str] = set()

        for agent_config in agents_config:
            agent_id = agent_config["id"]
            logger.info(f"Building {agent_id}...")

            # Create separate memory bank for each agent
            memory_bank = basic_associative_memory.AssociativeMemoryBank(
                sentence_embedder=self.embedder
            )

            # Create agent prefab
            prefab = EvacuationAgent(params=agent_config)

            # Build agent
            agent = prefab.build(
                model=self.model,
                memory_bank=memory_bank,
            )

            concordia_agents[agent_id] = agent

            # Track if agent is injured
            if agent_config.get("is_injured", False):
                injured_agents.add(agent_id)

            # Add initial memories
            self._initialize_agent_memory(agent, agent_config)

        logger.info(
            f"Built {len(concordia_agents)} Concordia agents ({len(injured_agents)} injured)"
        )
        return concordia_agents, injured_agents

    def _initialize_agent_memory(self, agent: entity_lib.Entity, config: dict[str, Any]) -> None:
        """
        Initialize an agent's memory with background knowledge.

        Args:
            agent: Concordia agent entity
            config: Agent configuration dictionary
        """
        initial_memories = [
            "I am at a train station.",
            f"I am in the {config.get('initial_zone', 'platform')} area.",
            "I am waiting for my train.",
            "I am on my way to my destination.",
            self.station_layout_description,  # Station layout info
            "The station has clear signage for platforms and exits.",
            "I notice other passengers waiting and walking around.",
            "The atmosphere is calm and routine.",
        ]

        # Add injury-specific memories
        if config.get("is_injured", False):
            initial_memories.extend(
                [
                    "I am injured and moving slowly.",
                    "I may need assistance.",
                ]
            )

        for memory in initial_memories:
            agent.observe(memory)
