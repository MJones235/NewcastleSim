"""
Natural language observation formatting.

Formats simulation data into natural language observations for agents.
"""

from typing import Any

from scenarios.common.logger import get_logger

logger = get_logger(__name__)


class ObservationFormatter:
    """
    Formats observations into natural language.

    Handles:
    - Message display formatting
    - Conversation history formatting
    - Event formatting
    - Nearby agent list formatting
    """

    @staticmethod
    def format_received_messages(received_messages: list[dict[str, Any]]) -> list[str]:
        """
        Format received messages for display.

        Args:
            received_messages: List of message dictionaries

        Returns:
            List of formatted message strings
        """
        # Show recent unique messages (last 5)
        unique_messages = []
        seen_texts = set()
        for msg in reversed(received_messages):
            msg_key = msg["text"][:30].lower()
            if msg_key not in seen_texts:
                unique_messages.append(msg)
                seen_texts.add(msg_key)
            if len(unique_messages) >= 5:
                break

        if not unique_messages:
            return []

        lines = ["What people just said to you:"]
        for msg in reversed(unique_messages):
            sender_name = msg["from"].replace("agent_", "Person ")
            msg_type = msg.get("message_type", "")
            type_indicator = {
                "directed": " (to you)",
                "quiet": " (quietly)",
                "shout": " (shouting)",
            }.get(msg_type, "")
            lines.append(f'  - {sender_name}{type_indicator}: "{msg["text"]}"')

        return lines

    @staticmethod
    def format_conversation_history(
        agent_id: str,
        conversation_history: dict[str, list[dict]],
        nearby_agents: list[dict[str, Any]],
    ) -> list[str]:
        """
        Format conversation history for active conversations.

        Args:
            agent_id: ID of the observing agent
            conversation_history: Dict mapping other_agent_id to conversation messages
            nearby_agents: List of nearby agent info

        Returns:
            List of formatted conversation strings
        """
        # Only show conversations with nearby people who have exchanged multiple messages
        active_conversations = []
        nearby_ids = {a.get("id") for a in nearby_agents}

        for other_agent_id, messages in conversation_history.items():
            if other_agent_id in nearby_ids and len(messages) >= 2:
                # Get last 3 messages in this conversation
                recent = messages[-3:]
                convo_summary = []
                for m in recent:
                    direction = (
                        "You"
                        if m["from"] == agent_id
                        else other_agent_id.replace("agent_", "Person ")
                    )
                    convo_summary.append(f'{direction}: "{m["text"]}"')

                active_conversations.append(
                    {
                        "other": other_agent_id.replace("agent_", "Person "),
                        "summary": " → ".join(convo_summary),
                    }
                )

        if not active_conversations:
            return []

        lines = ["Recent conversation context:"]
        for convo in active_conversations[:2]:  # Max 2 to keep it concise
            lines.append(f"  - With {convo['other']}: {convo['summary']}")

        return lines

    @staticmethod
    def format_nearby_agent_ids(nearby_agents: list[dict[str, Any]]) -> list[str]:
        """
        Format nearby agent IDs for targeting messages.

        Args:
            nearby_agents: List of nearby agent info

        Returns:
            List with single formatted string, or empty list
        """
        # Only list IDs when there are a few people (not in crowds)
        if len(nearby_agents) > 0 and len(nearby_agents) <= 5:
            nearby_ids = [a.get("id") for a in nearby_agents[:5] if a.get("id")]
            if nearby_ids:
                return [f"Nearby: {', '.join(nearby_ids)}"]
        return []

    @staticmethod
    def format_exit_crowds(exit_crowds: dict[str, int], categorize_func) -> list[str]:
        """
        Format exit crowd information.

        Args:
            exit_crowds: Dict mapping exit name to agent count
            categorize_func: Function to categorize count into string

        Returns:
            List of formatted exit crowd strings
        """
        if not exit_crowds:
            return []

        lines = ["People heading toward exits:"]
        for exit_name, count in sorted(exit_crowds.items()):
            lines.append(f"  - {exit_name}: {categorize_func(count)}")

        return lines

    @staticmethod
    def format_events(events: list[str]) -> list[str]:
        """
        Format recent events.

        Args:
            events: List of event strings

        Returns:
            List of formatted event strings
        """
        if not events:
            return []

        lines = ["Recent events:"]
        for event in events[-3:]:  # Last 3 events
            lines.append(f"  - {event}")

        return lines

    @staticmethod
    def format_blocked_exits(visible_blocked: list[dict[str, Any]]) -> list[str]:
        """
        Format visible blocked exits.

        Args:
            visible_blocked: List of blocked exit info dicts

        Returns:
            List of formatted blocked exit strings
        """
        if not visible_blocked:
            return []

        lines = ["⚠️ Visual observations:"]
        for blocked in visible_blocked:
            lines.append(
                f"  - The {blocked['name']} appears blocked/obstructed "
                f"({blocked['distance']} away)"
            )

        return lines

    @staticmethod
    def format_own_status(
        agent_id: str,
        agent_injured: set[str],
        agent_action: dict[str, str],
        helping_relationships,
        state_queries=None,
    ) -> list[str]:
        """
        Format agent's own status from three-dimensional model.

        Args:
            agent_id: ID of the agent
            agent_injured: Set of injured agent IDs
            agent_action: Dict of agent_id -> action ("moving"|"waiting")
            helping_relationships: HelpingRelationships tracker
            state_queries: SimulationStateQueries for position lookups (optional)

        Returns:
            List of formatted status strings (may be empty)
        """
        lines = []

        # Physical capability dimension
        if agent_id in agent_injured:
            lines.append("You are injured and moving slowly.")

            # Check if someone is helping them
            if helping_relationships:
                helper_id = helping_relationships.get_helper(agent_id)
                if helper_id and state_queries:
                    try:
                        helper_pos = state_queries.get_agent_position(helper_id)
                        helped_pos = state_queries.get_agent_position(agent_id)
                        if helper_pos and helped_pos:
                            distance = (
                                (helper_pos[0] - helped_pos[0]) ** 2
                                + (helped_pos[1] - helper_pos[1]) ** 2
                            ) ** 0.5

                            if distance < 3.0:
                                lines.append(f"{helper_id} has reached you and is with you now.")
                            elif distance < 10.0:
                                lines.append(
                                    f"{helper_id} is approaching to help ({distance:.1f}m away)."
                                )
                    except Exception:
                        pass

        # Social relationship dimension - with distance context
        if helping_relationships and helping_relationships.is_helping(agent_id):
            helped_id = helping_relationships.get_helped(agent_id)
            if helped_id:
                # Calculate distance if state_queries available
                if state_queries:
                    try:
                        helper_pos = state_queries.get_agent_position(agent_id)
                        helped_pos = state_queries.get_agent_position(helped_id)
                        if helper_pos and helped_pos:
                            distance = (
                                (helper_pos[0] - helped_pos[0]) ** 2
                                + (helped_pos[1] - helper_pos[1]) ** 2
                            ) ** 0.5

                            if distance < 3.0:
                                # Close enough to have reached them
                                lines.append(
                                    f"You have reached {helped_id} and are now with them. "
                                    f"You could wait with them, guide them to an exit, or continue alone."
                                )
                            elif distance < 10.0:
                                lines.append(
                                    f"You are approaching {helped_id} ({distance:.1f}m away)."
                                )
                            else:
                                lines.append(f"You are moving toward {helped_id}.")
                        else:
                            lines.append(f"You are currently helping {helped_id}.")
                    except Exception:
                        lines.append(f"You are currently helping {helped_id}.")
                else:
                    lines.append(f"You are currently helping {helped_id}.")
            else:
                lines.append("You are currently helping another person.")

        # Action dimension (waiting for assistance is special case)
        action = agent_action.get(agent_id, "moving")
        if action == "waiting":
            # Only mention waiting if they're not already mentioned as injured/helping
            if agent_id not in agent_injured and not (
                helping_relationships and helping_relationships.is_helping(agent_id)
            ):
                lines.append("You are waiting.")

        return lines
