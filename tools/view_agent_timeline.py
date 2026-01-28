#!/usr/bin/env python3
"""
View structured agent timeline from simulation output.
"""

import json
import sys
from pathlib import Path


def format_demographics(demographics):
    """Format demographics for display."""
    age = demographics.get("age", "?")
    gender = demographics.get("gender", "?")
    ptype = demographics.get("personality_type", "?")
    return f"{age}yo {gender}, {ptype}"


def view_agent_timeline(timeline_file, agent_id=None):
    """
    View agent timeline from JSON file.

    Args:
        timeline_file: Path to agent_timelines.json
        agent_id: Optional specific agent ID to view (default: show all)
    """
    with open(timeline_file) as f:
        data = json.load(f)

    print("=" * 80)
    print("AGENT TIMELINE VIEWER")
    print("=" * 80)
    print(f"Total agents: {data['total_agents']}")
    print()

    if agent_id:
        # View specific agent
        agent_data = next((a for a in data["agents"] if a["agent_id"] == agent_id), None)
        if not agent_data:
            print(f"Agent {agent_id} not found!")
            return

        display_agent(agent_data)
    else:
        # Show summary of all agents
        print("AGENT SUMMARY:")
        print("-" * 80)
        for agent_data in data["agents"]:
            aid = agent_data["agent_id"]
            demo = format_demographics(agent_data["demographics"])
            actions = agent_data["total_actions"]

            # Count decision types
            decisions = [a for a in agent_data["actions"] if a["action_type"] == "decision"]
            evacuate_decisions = sum(
                1 for d in decisions if d["details"].get("decision") == "evacuate"
            )

            print(
                f"{aid:12} | {demo:40} | {actions:3} actions | evacuated: {evacuate_decisions > 0}"
            )

        print()
        print("Use: python view_agent_timeline.py <timeline_file> <agent_id> to see details")


def display_agent(agent_data):
    """Display detailed timeline for one agent."""
    aid = agent_data["agent_id"]
    demo = agent_data["demographics"]

    print(f"AGENT: {aid}")
    print(f"Age: {demo.get('age', '?')}")
    print(f"Gender: {demo.get('gender', '?')}")
    print(
        f"Personality: {demo.get('personality_type', '?')} - {demo.get('personality_description', '')}"
    )
    print(f"Spawn time: {agent_data.get('spawn_time', 'N/A')}s")
    print(f"Exit time: {agent_data.get('exit_time', 'still active')}")
    print(f"Total actions: {agent_data['total_actions']}")
    print()
    print("TIMELINE:")
    print("-" * 80)

    for action in agent_data["actions"]:
        timestamp = action["timestamp"]
        atype = action["action_type"]
        details = action["details"]

        if atype == "spawn":
            print(f"[{timestamp:6.1f}s] SPAWN at {details.get('location', '?')}")

        elif atype == "message_received":
            from_who = details.get("from_agent", "?")
            msg = details.get("message", "")[:60]
            print(f'[{timestamp:6.1f}s] ← MESSAGE from {from_who}: "{msg}..."')

        elif atype == "decision":
            decision = details.get("decision", "?")
            confidence = details.get("confidence", 0)
            reasoning = details.get("reasoning", "")[:100]
            print(
                f"[{timestamp:6.1f}s] DECISION: {decision.upper()} (confidence: {confidence:.2f})"
            )
            print(f"             Reasoning: {reasoning}...")

        elif atype == "message_sent":
            msg = details.get("message", "")[:60]
            recipients = details.get("recipients_count", 0)
            radius = details.get("broadcast_radius", 0)
            print(
                f'[{timestamp:6.1f}s] → BROADCAST to {recipients} agents (radius={radius}m): "{msg}..."'
            )

        elif atype == "state_change":
            old = details.get("old_state", "?")
            new = details.get("new_state", "?")
            reason = details.get("reason", "")[:60]
            print(f"[{timestamp:6.1f}s] STATE: {old} → {new}")
            print(f"             Reason: {reason}")

        elif atype == "exit":
            loc = details.get("exit_location", "?")
            print(f"[{timestamp:6.1f}s] EXIT at {loc}")

    print("=" * 80)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python view_agent_timeline.py <timeline_file> [agent_id]")
        print(
            "Example: python view_agent_timeline.py scenarios/station_jupedsim/output/agent_timelines.json"
        )
        print(
            "Example: python view_agent_timeline.py scenarios/station_jupedsim/output/agent_timelines.json agent_5"
        )
        sys.exit(1)

    timeline_file = Path(sys.argv[1])
    if not timeline_file.exists():
        print(f"Error: File not found: {timeline_file}")
        sys.exit(1)

    agent_id = sys.argv[2] if len(sys.argv) > 2 else None
    view_agent_timeline(timeline_file, agent_id)
