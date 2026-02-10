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
    def format_own_status(agent_id: str, agent_status: dict[str, str]) -> list[str]:
        """
        Format agent's own status.

        Args:
            agent_id: ID of the agent
            agent_status: Dict mapping agent_id to status

        Returns:
            List of formatted status strings (may be empty)
        """
        lines = []
        own_status = agent_status.get(agent_id, "EVACUATING")

        if own_status == "HELPING":
            # Find who they're helping
            for helped_id, helper_id in agent_status.items():
                if helped_id.startswith("helped_by_") and helper_id == agent_id:
                    lines.append("You are currently helping another person.")
                    break
        elif own_status == "INJURED":
            lines.append("You are injured and moving slowly.")
        elif own_status == "WAITING":
            lines.append("You are waiting for assistance.")

        return lines
