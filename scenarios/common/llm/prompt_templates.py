"""Prompt templates for LLM decision-making."""

from typing import Any


class EvacuationPromptBuilder:
    """
    Builds prompts for evacuation decision-making.

    Creates contextual prompts that include:
    - Current agent state and location
    - Recent messages received
    - Time since message
    - Agent's personality/risk tolerance
    """

    @staticmethod
    def build_evacuation_prompt(agent: Any, message: dict | str) -> str:
        """
        Build prompt for evacuation decision.

        Args:
            agent: The agent making the decision
            message: The message that triggered the decision (dict or str)

        Returns:
            Formatted prompt string for LLM
        """
        # Extract message text
        if isinstance(message, dict):
            message_text = message.get("text", str(message))
        else:
            message_text = str(message)

        # Get agent context
        position = getattr(agent, "position", (0, 0))
        location = getattr(agent, "current_zone", "unknown")

        # Build prompt
        prompt = f"""You are a person at a train station who has just received the following announcement:

"{message_text}"

Your current situation:
- Location: {location}
- Position: approximately {position[0]:.1f}, {position[1]:.1f}
- You were waiting for your train

Based on this announcement, should you:
1. EVACUATE - Leave the station immediately via the nearest exit
2. STAY - Continue waiting for your train

Consider:
- The urgency and credibility of the message
- Your current location and safety
- Typical human behavior in such situations

Respond with your decision and brief reasoning."""

        return prompt

    @staticmethod
    def build_batch_prompts(agents: list[Any], messages: dict[int, dict | str]) -> list[str]:
        """
        Build prompts for multiple agents.

        Args:
            agents: List of agents needing decisions
            messages: Dict mapping agent IDs to their messages

        Returns:
            List of prompts (one per agent)
        """
        prompts = []
        for agent in agents:
            message = messages.get(agent.id, {})
            prompt = EvacuationPromptBuilder.build_evacuation_prompt(agent, message)
            prompts.append(prompt)
        return prompts
