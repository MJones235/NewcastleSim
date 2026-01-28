"""
Structured logging for individual agent decisions and actions.
Allows post-simulation analysis of agent behavior.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class AgentActionLog:
    """Single action/decision log entry for an agent."""

    timestamp: float  # Simulation time in seconds
    action_type: str  # decision, message_sent, message_received, state_change, spawn, exit
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp,
            "action_type": self.action_type,
            "details": self.details,
        }


@dataclass
class AgentTimeline:
    """Complete timeline of an agent's actions during simulation."""

    agent_id: str
    demographics: dict[str, Any]
    spawn_time: float | None = None
    exit_time: float | None = None
    actions: list[AgentActionLog] = field(default_factory=list)

    def add_action(self, timestamp: float, action_type: str, **details):
        """Add an action to the timeline."""
        self.actions.append(
            AgentActionLog(timestamp=timestamp, action_type=action_type, details=details)
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "agent_id": self.agent_id,
            "demographics": self.demographics,
            "spawn_time": self.spawn_time,
            "exit_time": self.exit_time,
            "total_actions": len(self.actions),
            "actions": [action.to_dict() for action in self.actions],
        }


class AgentActionLogger:
    """Manages structured logging of agent actions throughout simulation."""

    def __init__(self):
        """Initialize the agent action logger."""
        self.timelines: dict[str, AgentTimeline] = {}

    def register_agent(self, agent):
        """
        Register an agent for logging.

        Args:
            agent: StationAgent instance
        """
        demographics = {
            "age": getattr(agent, "age", None),
            "gender": getattr(agent, "gender", None),
            "personality_type": getattr(agent, "personality_type", None),
            "personality_description": getattr(agent, "personality_description", None),
        }

        self.timelines[agent.id] = AgentTimeline(agent_id=agent.id, demographics=demographics)

    def log_spawn(self, agent_id: str, timestamp: float, location: str):
        """Log agent spawn."""
        if agent_id in self.timelines:
            self.timelines[agent_id].spawn_time = timestamp
            self.timelines[agent_id].add_action(
                timestamp=timestamp, action_type="spawn", location=location
            )

    def log_decision(
        self,
        agent_id: str,
        timestamp: float,
        decision: str,
        reasoning: str,
        confidence: float,
        prompt: str | None = None,
        full_response: dict | None = None,
    ):
        """Log an LLM decision."""
        if agent_id in self.timelines:
            details = {
                "decision": decision,
                "reasoning": reasoning,
                "confidence": confidence,
            }
            if prompt:
                details["prompt"] = prompt
            if full_response:
                details["full_response"] = full_response

            self.timelines[agent_id].add_action(
                timestamp=timestamp, action_type="decision", **details
            )

    def log_message_sent(
        self,
        agent_id: str,
        timestamp: float,
        message: str,
        recipients_count: int,
        broadcast_radius: float,
    ):
        """Log a broadcast message sent by agent."""
        if agent_id in self.timelines:
            self.timelines[agent_id].add_action(
                timestamp=timestamp,
                action_type="message_sent",
                message=message,
                recipients_count=recipients_count,
                broadcast_radius=broadcast_radius,
            )

    def log_message_received(self, agent_id: str, timestamp: float, message: str, from_agent: str):
        """Log a message received from another agent or system."""
        if agent_id in self.timelines:
            self.timelines[agent_id].add_action(
                timestamp=timestamp,
                action_type="message_received",
                message=message,
                from_agent=from_agent,
            )

    def log_state_change(
        self,
        agent_id: str,
        timestamp: float,
        old_state: str,
        new_state: str,
        reason: str | None = None,
    ):
        """Log a state change (e.g., start evacuating)."""
        if agent_id in self.timelines:
            details = {"old_state": old_state, "new_state": new_state}
            if reason:
                details["reason"] = reason

            self.timelines[agent_id].add_action(
                timestamp=timestamp, action_type="state_change", **details
            )

    def log_exit(self, agent_id: str, timestamp: float, exit_location: str | None = None):
        """Log agent exiting the simulation."""
        if agent_id in self.timelines:
            self.timelines[agent_id].exit_time = timestamp
            self.timelines[agent_id].add_action(
                timestamp=timestamp, action_type="exit", exit_location=exit_location
            )

    def save_to_file(self, output_path: Path):
        """
        Save all agent timelines to JSON file.

        Args:
            output_path: Path to save the JSON file
        """
        output_data = {
            "total_agents": len(self.timelines),
            "agents": [timeline.to_dict() for timeline in self.timelines.values()],
        }

        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)

    def get_agent_timeline(self, agent_id: str) -> AgentTimeline | None:
        """Get timeline for a specific agent."""
        return self.timelines.get(agent_id)

    def get_summary_stats(self) -> dict:
        """Get summary statistics across all agents."""
        total_agents = len(self.timelines)
        total_decisions = sum(
            len([a for a in t.actions if a.action_type == "decision"])
            for t in self.timelines.values()
        )
        total_messages_sent = sum(
            len([a for a in t.actions if a.action_type == "message_sent"])
            for t in self.timelines.values()
        )
        total_messages_received = sum(
            len([a for a in t.actions if a.action_type == "message_received"])
            for t in self.timelines.values()
        )

        evacuated = sum(
            1
            for t in self.timelines.values()
            if any(
                a.details.get("decision") == "evacuate"
                for a in t.actions
                if a.action_type == "decision"
            )
        )

        return {
            "total_agents": total_agents,
            "total_decisions": total_decisions,
            "total_messages_sent": total_messages_sent,
            "total_messages_received": total_messages_received,
            "agents_evacuated": evacuated,
            "agents_stayed": total_agents - evacuated,
        }
