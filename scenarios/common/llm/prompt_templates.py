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
    def build_evacuation_prompt(
        agent: Any, message: dict | str, message_history: list[dict] = None
    ) -> str:
        """
        Build prompt for evacuation decision.

        Args:
            agent: The agent making the decision
            message: The message that triggered the decision (dict or str)
            message_history: Optional list of previous messages with metadata

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

        # Get current zone/location from movement provider
        try:
            location_info = agent.movement_provider.get_agent_location_info(agent)
            location = location_info.get("zone", "unknown")
            # Use the location info position if available (more accurate)
            if "position" in location_info:
                position = location_info["position"]

            # Convert technical zone names to readable location names
            if location != "unknown":
                # Convert "jps.entrance_1" -> "Entrance 1"
                # Convert "jps.platform_2" -> "Platform 2"
                # Convert "jps.concourse" -> "Concourse"
                location = location.replace("jps.", "").replace("_", " ").title()
        except:
            location = "unknown"

        # Get agent demographics for personality-driven responses
        age = getattr(agent, "age", 30)
        gender = getattr(agent, "gender", "unknown")
        personality_type = getattr(agent, "personality_type", "ISTJ")
        personality_desc = getattr(agent, "personality_description", "")

        # Build message history section if provided
        history_text = ""
        if message_history and len(message_history) > 0:
            history_text = "\n\nMessage History (most recent first):"
            # Show last 5 messages
            recent_messages = list(reversed(message_history[-5:]))
            for msg in recent_messages:
                time_str = f"t={msg.get('time', 0):.1f}s"
                sender = msg.get("from", "unknown")
                content = msg.get("message", "")
                history_text += f'\n- [{time_str}] FROM {sender}: "{content}"'

        # Build prompt
        prompt = f"""You are a {age}-year-old {gender} at a train station who has just received the following announcement:

"{message_text}"

Your personality:
- Type: {personality_type} - {personality_desc}
- This shapes how you perceive risk, make decisions, and communicate with others

Your current situation:
- Location: {location}
- Position: approximately {position[0]:.1f}, {position[1]:.1f}
- You were waiting for your train{history_text}

Based on this announcement{' and previous messages' if history_text else ''}, you need to decide:

1. EVACUATION DECISION - Should you evacuate or stay?
   - EVACUATE: Leave the station immediately via the nearest exit
   - STAY: Continue waiting for your train

2. COMMUNICATION (optional) - Should you warn others nearby?
   - Only broadcast if you have URGENT NEW information others likely don't have
   - Most people can hear system announcements themselves - don't repeat what was just announced
   - Your message reaches people within ~2 meters (immediate vicinity only)
   - Shouting too often reduces credibility and causes confusion
   - Consider: Is your message adding value, or just echoing what everyone already heard?

Consider:
- The urgency and credibility of the message
- Your current location and safety
- Typical human behavior in such situations
- Whether warning others might help or cause panic
- Messages from other people can influence your decision (they may have information you don't)
- If multiple people are warning about danger, it increases credibility
- Don't broadcast just because others are - only if you have something new to add

Respond in JSON format with these fields:
{{
  "decision": "evacuate" or "stay",
  "reasoning": "Your explanation for the decision",
  "confidence": 0.0 to 1.0,
  "broadcast_message": "Optional: Only if you have urgent new info (or null)"
}}

Example with communication (rare - only when you have NEW urgent info):
{{
  "decision": "evacuate",
  "reasoning": "I can see smoke that others might not have noticed yet",
  "confidence": 0.9,
  "broadcast_message": "I see smoke - evacuate now!"
}}

Example without communication (most common - announcement was clear):
{{
  "decision": "evacuate",
  "reasoning": "The announcement was clear and everyone heard it",
  "confidence": 0.85,
  "broadcast_message": null,
  "broadcast_radius": null
}}"""

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
        # Build prompts using list comprehension for efficiency
        prompts = [
            EvacuationPromptBuilder.build_evacuation_prompt(
                agent,
                messages.get(agent.id, {}),
                message_history=(
                    agent.decision_maker.message_history
                    if hasattr(agent, "decision_maker")
                    and hasattr(agent.decision_maker, "message_history")
                    else None
                ),
            )
            for agent in agents
        ]
        return prompts
