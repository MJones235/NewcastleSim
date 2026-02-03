"""
Custom Concordia agent prefab for evacuation scenarios.

This prefab extends Concordia's basic Entity with evacuation-specific components:
- Evacuation memory formation
- Risk perception
- Social influence observation
- Exit knowledge and mental mapping
"""

import dataclasses
from collections.abc import Mapping
from typing import Any

from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import basic_associative_memory
from concordia.components import agent as agent_components
from concordia.language_model import language_model
from concordia.typing import prefab as prefab_lib


@dataclasses.dataclass
class EvacuationAgent(prefab_lib.Prefab):
    """
    A Concordia agent prefab specialized for evacuation scenarios.

    Extends the basic reasoning agent with evacuation-specific components:
    - Enhanced memory for evacuation events
    - Risk perception reasoning
    - Social influence observation
    - Exit knowledge
    """

    description: str = (
        "An evacuation agent that reasons about emergency situations using "
        "memory, personality, risk perception, and social influence."
    )

    params: Mapping[str, Any] = dataclasses.field(
        default_factory=lambda: {
            "name": "Agent",
            "personality_type": "ISTJ",
            "age": 30,
            "gender": "neutral",
            "initial_zone": "platform",
            "destination": "exit",
            "risk_tolerance": "moderate",  # low, moderate, high
        }
    )

    def build(
        self,
        model: language_model.LanguageModel,
        memory_bank: basic_associative_memory.AssociativeMemoryBank,
    ) -> entity_agent_with_logging.EntityAgentWithLogging:
        """
        Build an evacuation agent with Concordia components.

        Args:
            model: The language model to use for reasoning
            memory_bank: The associative memory bank for storing experiences

        Returns:
            A configured EntityAgentWithLogging instance
        """
        # Extract parameters
        name = self.params.get("name", "Agent")
        personality = self.params.get("personality_type", "ISTJ")
        age = self.params.get("age", 30)
        gender = self.params.get("gender", "neutral")
        risk_tolerance = self.params.get("risk_tolerance", "moderate")

        # Define personality descriptions
        personality_descriptions = {
            "ISTJ": "Practical and fact-minded. Values order, procedures, and reliability. Prefers clear instructions.",
            "ISFJ": "Caring and loyal. Values harmony and helping others. May prioritize group safety.",
            "INFJ": "Idealistic and principled. Seeks meaning and authenticity. May consider ethical implications.",
            "INTJ": "Strategic and analytical. Values competence and logic. Plans multiple steps ahead.",
            "ISTP": "Practical and adaptable. Observant problem-solver. Handles crisis calmly.",
            "ISFP": "Gentle and sensitive. Values personal freedom. May be spontaneous in reactions.",
            "INFP": "Idealistic and empathetic. Guided by personal values. May help others first.",
            "INTP": "Analytical and curious. Seeks logical understanding. May analyze before acting.",
            "ESTP": "Energetic and pragmatic. Thrives on action. Quick to respond to emergencies.",
            "ESFP": "Enthusiastic and sociable. Lives in the moment. May follow crowd behavior.",
            "ENFP": "Enthusiastic and creative. Sees possibilities. May encourage others.",
            "ENTP": "Inventive and analytical. Enjoys challenges. May find unconventional solutions.",
            "ESTJ": "Organized and decisive. Values order and tradition. Takes charge in crisis.",
            "ESFJ": "Warm and cooperative. Values harmony. May organize group evacuation.",
            "ENFJ": "Charismatic and empathetic. Focused on helping others. Natural leader in crisis.",
            "ENTJ": "Bold and strategic. Values efficiency. Commands authority in emergencies.",
        }

        personality_desc = personality_descriptions.get(personality, "Unknown personality")

        # Core components
        memory_key = agent_components.memory.DEFAULT_MEMORY_COMPONENT_KEY
        memory = agent_components.memory.AssociativeMemory(memory_bank=memory_bank)

        # Instructions component with evacuation context
        instructions_key = "Instructions"
        instructions = agent_components.instructions.Instructions(
            agent_name=name,
            pre_act_label="\nAgent Profile",
        )

        # Observation to memory - automatically stores observations
        observation_to_memory_key = "ObservationToMemory"
        observation_to_memory = agent_components.observation.ObservationToMemory()

        # Recent observations
        observation_key = agent_components.observation.DEFAULT_OBSERVATION_COMPONENT_KEY
        observation = agent_components.observation.LastNObservations(
            history_length=20,
            pre_act_label="\nRecent Events (newest last)",
        )

        # Self perception with personality
        self_perception_key = "SelfPerception"
        self_perception = agent_components.question_of_recent_memories.SelfPerception(
            model=model,
            pre_act_label=f"\nCharacter Profile - {name}",
        )

        # Situation perception - understand current situation
        situation_perception_key = "SituationPerception"
        situation_perception = agent_components.question_of_recent_memories.SituationPerception(
            model=model,
            pre_act_label=f"\nCurrent Situation - {name}",
        )

        # Risk perception - custom reasoning about danger level
        risk_perception_key = "RiskPerception"
        risk_perception = agent_components.question_of_recent_memories.QuestionOfRecentMemories(
            model=model,
            pre_act_label=f"\nThreat Assessment - {name}",
            question=(
                f"Evaluate the danger level based on recent observations and events. "
                f"Assess any immediate threats or risks. "
                f"Note: {name} has {risk_tolerance} risk tolerance."
            ),
            answer_prefix=f"{name} assesses: ",
            add_to_memory=True,
            memory_tag="[risk assessment]",
        )

        # Social influence - observe what others are doing
        social_influence_key = "SocialInfluence"
        social_influence = agent_components.question_of_recent_memories.QuestionOfRecentMemories(
            model=model,
            pre_act_label=f"\nSocial Context - {name}",
            question=(
                f"Describe observed behaviors of nearby people based on recent observations. "
                f"Note their emotional state (calm, urgent, orderly) and actions (waiting, moving). "
                f"Consider how this social context may inform {name}'s decision."
            ),
            answer_prefix=f"{name} notes: ",
            add_to_memory=False,
        )

        # Person by situation - core decision reasoning
        person_by_situation_key = "PersonBySituation"
        person_by_situation = agent_components.question_of_recent_memories.PersonBySituation(
            model=model,
            components=[
                self_perception_key,
                situation_perception_key,
                risk_perception_key,
                social_influence_key,
            ],
            pre_act_label=(f"\nSimulated Decision Process - {name}"),
        )

        # Relevant memories
        relevant_memories_key = "RelevantMemories"
        relevant_memories = agent_components.all_similar_memories.AllSimilarMemories(
            model=model,
            components=[situation_perception_key],
            num_memories_to_retrieve=10,
            pre_act_label="\nRelevant Memories",
        )

        # Goal component - everyday objective (no emergency framing)
        goal_key = "Goal"
        evacuation_goal = agent_components.constant.Constant(
            state="Continue your planned trip (e.g., wait for your train or head to your destination).",
            pre_act_label="\nGoal",
        )

        # Personality component
        personality_key = "Personality"
        personality_component = agent_components.constant.Constant(
            state=(
                f"{name} is a {age}-year-old {gender}. "
                f"Personality type: {personality} - {personality_desc}"
            ),
            pre_act_label="\nPersonality",
        )

        # Assemble all components
        components_of_agent = {
            instructions_key: instructions,
            personality_key: personality_component,
            goal_key: evacuation_goal,
            observation_to_memory_key: observation_to_memory,
            relevant_memories_key: relevant_memories,
            self_perception_key: self_perception,
            situation_perception_key: situation_perception,
            risk_perception_key: risk_perception,
            social_influence_key: social_influence,
            person_by_situation_key: person_by_situation,
            observation_key: observation,
            memory_key: memory,
        }

        # Component order for prompt construction
        component_order = [
            instructions_key,
            personality_key,
            goal_key,
            relevant_memories_key,
            self_perception_key,
            situation_perception_key,
            risk_perception_key,
            social_influence_key,
            person_by_situation_key,
            observation_key,
        ]

        # Action component - generates final action
        act_component = agent_components.concat_act_component.ConcatActComponent(
            model=model,
            component_order=component_order,
            randomize_choices=False,  # Deterministic for evacuation
            prefix_entity_name=True,
        )

        # Create agent
        agent = entity_agent_with_logging.EntityAgentWithLogging(
            agent_name=name,
            act_component=act_component,
            context_components=components_of_agent,
        )

        return agent
