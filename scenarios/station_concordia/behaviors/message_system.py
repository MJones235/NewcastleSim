"""
Message system for agent-to-agent communication in evacuation scenarios.

Handles:
- Message extraction from agent actions
- Message delivery to nearby agents based on type (directed, shout, quiet)
- Message memory and deduplication to prevent repetition
- Conversation tracking between agents
"""

import json
from typing import Any

from scenarios.common.logger import get_logger

logger = get_logger(__name__)


class MessageSystem:
    """
    Manages agent-to-agent messaging with memory and deduplication.

    Features:
    - Spatial message delivery (radius-based)
    - Message types: directed (to specific agent), shout (wide radius), quiet (narrow radius)
    - Repetition prevention (agents don't repeat recently sent messages)
    - Deduplication (agents don't hear the same message twice)
    - Conversation tracking (maintains dialogue history between agent pairs)
    """

    def __init__(
        self,
        default_radius: float = 10.0,
        memory_window: float = 60.0,
    ):
        """
        Initialize the message system.

        Args:
            default_radius: Default radius for message delivery (meters)
            memory_window: How long to remember sent messages (seconds)
        """
        self.default_radius = default_radius
        self.memory_window = memory_window

        # Message state
        self.agent_messages: dict[str, list[dict[str, Any]]] = {}  # agent_id -> received messages
        self.message_history: list[dict[str, Any]] = []  # All messages sent
        self.agent_sent_messages: dict[str, list[dict[str, Any]]] = (
            {}
        )  # agent_id -> recent sent messages
        self.agent_heard_messages: dict[str, set[str]] = (
            {}
        )  # agent_id -> heard message content
        self.agent_conversations: dict[str, dict[str, list[dict]]] = (
            {}
        )  # agent_id -> {other_agent_id -> conversation}

    def extract_and_deliver_message(
        self,
        sender_id: str,
        action: str,
        sender_position: tuple[float, float],
        current_sim_time: float,
        state_queries: Any,  # SimulationStateQueries instance
        agent_status: dict[str, str],
        exited_agents: set[str],
    ) -> dict[str, Any] | None:
        """
        Extract message from action JSON and deliver to nearby agents.

        Args:
            sender_id: ID of the agent sending the message
            action: JSON action string from agent
            sender_position: Current position of sender
            current_sim_time: Current simulation time
            state_queries: SimulationStateQueries instance for finding nearby agents
            agent_status: Dict mapping agent_id to status (for finding injured agents)
            exited_agents: Set of agent IDs who have exited

        Returns:
            Message info dict if message was sent, None otherwise
        """
        try:
            # Parse action JSON to extract message
            json_start = action.find("{")
            if json_start > 0:
                action = action[json_start:]

            data = json.loads(action)
            message_text = data.get("message")
            message_type = data.get("message_type")  # directed, shout, quiet
            target_agent = data.get("target_agent")  # agent_id, nearest_injured, or null

            if not message_text or message_text == "null":
                return None

            # Check message memory - prevent repetition
            if self._is_repeat_message(sender_id, message_text, current_sim_time):
                return None

            # Determine message radius based on type
            radius = self._get_message_radius(message_type)

            # Find nearby agents
            nearby_agents = state_queries.get_nearby_agents(sender_id, radius)

            # Filter recipients based on target_agent
            recipient_ids = self._find_recipients(
                target_agent, nearby_agents, agent_status, exited_agents, sender_id
            )

            if not recipient_ids:
                logger.debug(f"📢 {sender_id} sent message but no valid recipients nearby")
                return None

            # Create message record
            message_record = {
                "time": current_sim_time,
                "sender": sender_id,
                "position": sender_position,
                "text": message_text,
                "message_type": message_type or "broadcast",
                "target_agent": target_agent,
                "recipients": recipient_ids,
                "num_recipients": len(recipient_ids),
            }

            # Deliver message to each recipient
            self._deliver_to_recipients(
                sender_id, message_text, message_type, recipient_ids, current_sim_time
            )

            # Store in message history
            self.message_history.append(message_record)

            # Track this message in sender's sent history
            self._record_sent_message(sender_id, message_text, current_sim_time)

            # Log message with type indicator
            type_emoji = {"directed": "💬", "shout": "📢", "quiet": "🤫"}.get(message_type, "📣")
            logger.info(f"{type_emoji} {sender_id} → {len(recipient_ids)} people: '{message_text}'")
            return message_record

        except json.JSONDecodeError:
            return None
        except Exception as e:
            logger.warning(f"Error extracting message from {sender_id}: {e}")
            return None

    def get_received_messages(self, agent_id: str) -> list[dict[str, Any]]:
        """Get messages received by an agent and clear them."""
        messages = self.agent_messages.get(agent_id, [])
        self.agent_messages[agent_id] = []  # Clear after retrieval
        return messages

    def get_conversation_history(self, agent_id: str) -> dict[str, list[dict]]:
        """Get conversation history for an agent."""
        return self.agent_conversations.get(agent_id, {})

    def _is_repeat_message(self, sender_id: str, message_text: str, current_time: float) -> bool:
        """Check if this message was recently sent by the same agent."""
        if sender_id not in self.agent_sent_messages:
            return False

        # Clean old messages outside memory window
        recent_cutoff = current_time - self.memory_window
        self.agent_sent_messages[sender_id] = [
            msg for msg in self.agent_sent_messages[sender_id] if msg["time"] >= recent_cutoff
        ]

        # Check if similar message was recently sent
        for recent_msg in self.agent_sent_messages[sender_id]:
            if recent_msg["text"].lower() == message_text.lower():
                logger.debug(
                    f"{sender_id} suppressed repeat message: '{message_text}' "
                    f"(last sent {current_time - recent_msg['time']:.0f}s ago)"
                )
                return True

        return False

    def _get_message_radius(self, message_type: str | None) -> float:
        """Determine message radius based on type."""
        if message_type == "quiet":
            return 3.0  # Only very close people
        elif message_type == "shout":
            return 15.0  # Wider range for warnings
        else:
            return self.default_radius  # Default 10m

    def _find_recipients(
        self,
        target_agent: str | None,
        nearby_agents: list[dict[str, Any]],
        agent_status: dict[str, str],
        exited_agents: set[str],
        sender_id: str,
    ) -> list[str]:
        """Find recipient agent IDs based on targeting."""
        recipient_ids = []

        if target_agent and target_agent != "null":
            if target_agent == "nearest_injured":
                # Find nearest injured agent
                injured = [
                    a
                    for a in nearby_agents
                    if agent_status.get(a["id"], "").startswith("INJURED")
                ]
                if injured:
                    recipient_ids = [injured[0]["id"]]
            elif target_agent.startswith("agent_"):
                # Specific agent targeted
                if any(a["id"] == target_agent for a in nearby_agents):
                    recipient_ids = [target_agent]
        else:
            # Broadcast to all nearby (but filter out exited and self)
            recipient_ids = [
                agent["id"]
                for agent in nearby_agents
                if agent["id"] != sender_id and agent["id"] not in exited_agents
            ]

        return recipient_ids

    def _deliver_to_recipients(
        self,
        sender_id: str,
        message_text: str,
        message_type: str | None,
        recipient_ids: list[str],
        current_time: float,
    ):
        """Deliver message to all recipients with deduplication."""
        for recipient_id in recipient_ids:
            # Skip if recipient has heard this exact message recently
            if recipient_id not in self.agent_heard_messages:
                self.agent_heard_messages[recipient_id] = set()

            # Clean old heard messages
            if len(self.agent_heard_messages[recipient_id]) > 50:
                self.agent_heard_messages[recipient_id].clear()

            # Check if already heard
            msg_key = f"{message_text.lower()[:30]}"  # First 30 chars normalized
            if msg_key in self.agent_heard_messages[recipient_id]:
                continue  # Skip delivering duplicate

            self.agent_heard_messages[recipient_id].add(msg_key)

            # Deliver to recipient
            if recipient_id not in self.agent_messages:
                self.agent_messages[recipient_id] = []

            self.agent_messages[recipient_id].append(
                {
                    "time": current_time,
                    "from": sender_id,
                    "text": message_text,
                    "message_type": message_type or "broadcast",
                }
            )

            # Track conversation history between sender and recipient
            self._track_conversation(sender_id, recipient_id, message_text, current_time)

    def _track_conversation(
        self, sender_id: str, recipient_id: str, message_text: str, current_time: float
    ):
        """Track conversation history between two agents."""
        # Track on sender's side
        if sender_id not in self.agent_conversations:
            self.agent_conversations[sender_id] = {}
        if recipient_id not in self.agent_conversations[sender_id]:
            self.agent_conversations[sender_id][recipient_id] = []

        self.agent_conversations[sender_id][recipient_id].append(
            {
                "time": current_time,
                "from": sender_id,
                "to": recipient_id,
                "text": message_text,
            }
        )

        # Also track on recipient's side
        if recipient_id not in self.agent_conversations:
            self.agent_conversations[recipient_id] = {}
        if sender_id not in self.agent_conversations[recipient_id]:
            self.agent_conversations[recipient_id][sender_id] = []

        self.agent_conversations[recipient_id][sender_id].append(
            {
                "time": current_time,
                "from": sender_id,
                "to": recipient_id,
                "text": message_text,
            }
        )

    def _record_sent_message(self, sender_id: str, message_text: str, current_time: float):
        """Record a message in sender's sent history."""
        if sender_id not in self.agent_sent_messages:
            self.agent_sent_messages[sender_id] = []
        self.agent_sent_messages[sender_id].append({"time": current_time, "text": message_text})
