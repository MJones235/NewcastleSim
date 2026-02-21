"""
Agent builder for creating and initializing Concordia agents.

This module handles the creation of Concordia agents with their memory banks
and initial knowledge state.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
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
        observation_generator,
        jps_sim,
    ):
        """
        Initialize the agent builder.

        Args:
            language_model: LLM for agent cognition
            embedder: Sentence embedding function
            observation_generator: ObservationGenerator for level-aware station descriptions
            jps_sim: JuPedSim simulation instance for level information
        """
        self.model = language_model
        self.embedder = embedder
        self.observation_generator = observation_generator
        self.jps_sim = jps_sim

    async def build_agents(
        self, agents_config: list[dict[str, Any]]
    ) -> tuple[dict[str, entity_lib.Entity], set[str]]:
        """
        Build Concordia agents from configurations in parallel.

        Args:
            agents_config: List of agent configuration dictionaries

        Returns:
            Tuple of (concordia_agents dict, injured_agents set)
        """
        logger.info(f"Building {len(agents_config)} Concordia agents in parallel...")

        # Create a larger thread pool to handle many agents efficiently
        # Use min(32, agent_count) to avoid creating too many threads
        max_workers = min(32, len(agents_config))

        # Build all agents concurrently with custom executor
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            tasks = [
                self._build_single_agent(agent_config, executor) for agent_config in agents_config
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect results
        concordia_agents: dict[str, entity_lib.Entity] = {}
        injured_agents: set[str] = set()

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to build agent {agents_config[i]['id']}: {result}")
                continue

            agent, agent_id, is_injured = result
            concordia_agents[agent_id] = agent
            if is_injured:
                injured_agents.add(agent_id)

        logger.info(
            f"Built {len(concordia_agents)} Concordia agents ({len(injured_agents)} injured)"
        )
        return concordia_agents, injured_agents

    async def _build_single_agent(
        self, agent_config: dict[str, Any], executor: ThreadPoolExecutor
    ) -> tuple[entity_lib.Entity, str, bool]:
        """
        Build a single Concordia agent (async for parallel execution).

        Args:
            agent_config: Agent configuration dictionary
            executor: Thread pool executor to use

        Returns:
            Tuple of (agent, agent_id, is_injured)
        """
        agent_id = agent_config["id"]

        # Create separate memory bank for each agent
        memory_bank = basic_associative_memory.AssociativeMemoryBank(
            sentence_embedder=self.embedder
        )

        # Create agent prefab
        prefab = EvacuationAgent(params=agent_config)

        # Build agent using custom executor
        loop = asyncio.get_event_loop()
        agent = await loop.run_in_executor(
            executor,
            prefab.build,
            self.model,
            memory_bank,
        )

        # Add initial memories using custom executor
        await loop.run_in_executor(
            executor,
            self._initialize_agent_memory,
            agent,
            agent_config,
        )

        is_injured = agent_config.get("is_injured", False)

        return agent, agent_id, is_injured

    def _initialize_agent_memory(self, agent: entity_lib.Entity, config: dict[str, Any]) -> None:
        """
        Initialize an agent's memory with background knowledge.

        Args:
            agent: Concordia agent entity
            config: Agent configuration dictionary
        """
        # Get agent's level for level-specific station description
        agent_level = config.get("level_id", "0")

        # Generate level-specific station layout description
        station_layout_description = self.observation_generator._describe_geometry(agent_level)

        initial_memories = [
            "I am at a train station.",
            f"I am in the {config.get('initial_zone', 'platform')} area.",
            "I am waiting for my train.",
            "I am on my way to my destination.",
            station_layout_description,  # Level-specific station layout info
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
