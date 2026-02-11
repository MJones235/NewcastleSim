"""
Action Executor

Executes translated agent actions in the JuPedSim simulation, including:
- Speed adjustments
- Movement to exits, waypoints, or toward other agents
- Waiting behavior (standing still or seeking information)
- Organic helping relationship detection (when moving toward injured agents)

Helping emerges naturally from movement + communication rather than special actions.
"""

import math
import random
from typing import Any

from scenarios.common.logger import get_logger
from scenarios.station_concordia.decision.action_utils import extract_exit_name
from scenarios.station_concordia.utils.speed_utils import convert_speed_to_ms

logger = get_logger(__name__)


class ActionExecutor:
    """Applies translated actions to the JuPedSim simulation."""

    def __init__(
        self,
        jps_sim,
        state_queries,
        event_manager,
        station_layout: dict[str, Any],
        agent_injured: set[str],
        agent_action: dict[str, str],
        agent_destinations: dict[str, str],
        helping_relationships,  # HelpingRelationships instance
        help_events: list[dict[str, Any]],
        wait_events: list[dict[str, Any]],
        agent_configs: list[dict[str, Any]],
        test_scenarios: dict[str, Any],
    ):
        """
        Initialize action executor.

        Args:
            jps_sim: JuPedSim simulation instance
            state_queries: Simulation state query interface
            event_manager: Event manager for blocked exits
            station_layout: Station geometry and exit information
            agent_injured: Set of agent IDs who are injured/slow
            agent_action: agent_id -> "moving"|"waiting"
            agent_destinations: agent_id -> current exit name
            helping_relationships: HelpingRelationships instance for tracking help
            help_events: List tracking all help interactions
            wait_events: List tracking all wait decisions with reasons
            agent_configs: List of agent configuration dictionaries
            test_scenarios: Test scenario configuration
        """
        self.jps_sim = jps_sim
        self.state_queries = state_queries
        self.event_manager = event_manager
        self.station_layout = station_layout
        self.agent_injured = agent_injured
        self.agent_action = agent_action
        self.agent_destinations = agent_destinations
        self.helping_relationships = helping_relationships
        self.help_events = help_events
        self.wait_events = wait_events
        self.agent_configs = agent_configs
        self.test_scenarios = test_scenarios

    def execute_action(
        self, agent_id: str, translated_action: dict[str, Any], current_sim_time: float
    ):
        """
        Apply a translated action to the JuPedSim simulation.

        Args:
            agent_id: ID of the agent performing the action
            translated_action: Dict containing action details from ActionTranslator
            current_sim_time: Current simulation time in seconds
        """
        action_type = translated_action["action_type"]
        target = translated_action["target"]

        # Store time in translated_action for downstream use
        translated_action["time"] = current_sim_time

        logger.info(
            f"Agent {agent_id}: {action_type} to {target} "
            f"(confidence: {translated_action['confidence']:.2f}) - {translated_action['reasoning']}"
        )

        try:
            # Apply dynamic speed if specified
            speed_str = translated_action.get("speed")
            if speed_str:
                speed_ms = convert_speed_to_ms(speed_str)
                if speed_ms:
                    self.jps_sim.set_agent_speed(agent_id, speed_ms)
                    logger.debug(f"Set {agent_id} speed to {speed_ms:.2f} m/s ({speed_str})")

            # Resolve target_agent to actual position if specified
            target_agent_id = translated_action.get("target_agent")
            if target_agent_id and action_type == "move":
                target_position = self.state_queries.get_agent_position(target_agent_id)
                if target_position:
                    target = target_position
                    translated_action["target"] = target
                    logger.debug(f"{agent_id} moving toward {target_agent_id} at {target}")

            if action_type == "move" and target:
                self._handle_move_action(agent_id, translated_action, target)
            elif action_type == "wait":
                self._handle_wait_action(agent_id, translated_action, current_sim_time)

        except Exception as e:
            logger.error(f"Failed to apply action for {agent_id}: {e}")

    def _handle_move_action(self, agent_id: str, translated_action: dict[str, Any], target):
        """Handle move action: agent moving to exit, waypoint, or toward another agent."""
        # Update action state
        self.agent_action[agent_id] = "moving"

        # Check if moving toward another agent (helping behavior detection)
        target_agent = translated_action.get("target_agent")

        if target_agent:
            # Agent is moving toward another agent
            target_is_injured = target_agent in self.agent_injured

            if target_is_injured and not self.helping_relationships.is_being_helped(target_agent):
                # Establish helping relationship organically
                self.helping_relationships.start_helping(
                    helper_id=agent_id,
                    helped_id=target_agent,
                    help_type="approach_and_assist",
                    current_time=translated_action.get("time", 0.0),
                )

                # Record help event
                helper_config = next((c for c in self.agent_configs if c["id"] == agent_id), {})
                position = self.state_queries.get_agent_position(agent_id)

                self.help_events.append(
                    {
                        "time": translated_action.get("time", 0.0),
                        "helper": agent_id,
                        "helped": target_agent,
                        "helper_personality": helper_config.get("personality_type", "UNKNOWN"),
                        "help_type": "emergent_from_movement",
                        "location": position,
                    }
                )

                logger.info(
                    f"🤝 {agent_id} moving toward injured {target_agent} (helping relationship established)"
                )
            else:
                # Just moving toward another agent (following, coordinating, etc.)
                logger.debug(f"{agent_id} moving toward {target_agent}")
        else:
            # Regular movement - check if they were helping and moving away
            if self.helping_relationships.is_helping(agent_id):
                # Moving without target_agent - they're moving away from who they were helping
                helped_id = self.helping_relationships.get_helped(agent_id)
                if helped_id:
                    # Check if they're still near the helped person
                    helper_pos = self.state_queries.get_agent_position(agent_id)
                    helped_pos = self.state_queries.get_agent_position(helped_id)
                    if helper_pos and helped_pos:
                        distance = (
                            (helper_pos[0] - helped_pos[0]) ** 2
                            + (helper_pos[1] - helped_pos[1]) ** 2
                        ) ** 0.5
                        if distance > 5.0:  # More than 5 meters away
                            self.helping_relationships.stop_helping(agent_id, reason="moved_away")
                            logger.info(f"👋 {agent_id} moved away from {helped_id}, ending help")

        # Extract the NEW exit name from this action (if moving to an exit)
        new_exit_name = extract_exit_name(translated_action, self.station_layout)

        if new_exit_name:
            # Agent is moving to an exit - update destination tracking
            self.agent_destinations[agent_id] = new_exit_name

            # Check if agent is trying to switch to a blocked exit
            if new_exit_name in self.event_manager.blocked_exits:
                logger.debug(
                    f"⚠️ {agent_id} tried to switch to blocked exit {new_exit_name} - "
                    f"keeping waypoint only"
                )
                # Only set waypoint, don't switch journey (would let them evacuate through blocked exit)
                self.jps_sim.set_agent_target(agent_id, target)
            else:
                # Switch the agent's evacuation journey to this exit
                if hasattr(self.jps_sim, "set_agent_evacuation_exit"):
                    self.jps_sim.set_agent_evacuation_exit(agent_id, new_exit_name)
                    logger.debug(f"Switched {agent_id} to journey for {new_exit_name}")
                else:
                    self.jps_sim.set_agent_target(agent_id, target)
        else:
            # Not moving to an exit, just a waypoint
            self.jps_sim.set_agent_target(agent_id, target)

    def _handle_wait_action(
        self, agent_id: str, translated_action: dict[str, Any], current_sim_time: float
    ):
        """Handle wait action: agent staying in place or seeking information."""
        # Update action state
        self.agent_action[agent_id] = "waiting"

        wait_reason = translated_action.get("wait_reason", "unspecified")

        # If helping and waiting for non-helping reasons, end the relationship
        if self.helping_relationships.is_helping(agent_id):
            if wait_reason not in ["waiting_with_injured", "helping", "assisting"]:
                # They chose to wait for reasons unrelated to helping
                self.helping_relationships.stop_helping(agent_id, reason="waiting_for_other_reason")
                logger.info(f"👋 {agent_id} stopped helping to wait ({wait_reason})")

        current_position = self.state_queries.get_agent_position(agent_id)

        # Different behavior based on wait reason
        if wait_reason == "seeking_information":
            # Seeking information: move slowly in a small random direction (looking around)
            # Generate a random nearby point within 3-5 meters
            distance = random.uniform(3.0, 5.0)
            angle = random.uniform(0, 2 * math.pi)
            target_x = current_position[0] + distance * math.cos(angle)
            target_y = current_position[1] + distance * math.sin(angle)

            self.jps_sim.set_agent_target(agent_id, (target_x, target_y))
            logger.debug(
                f"{agent_id} seeking information - moving slowly to nearby point "
                f"({distance:.1f}m away)"
            )
        else:
            # All other wait types: stand still at current position
            self.jps_sim.set_agent_target(agent_id, current_position)

        # Record wait event
        agent_config = next((c for c in self.agent_configs if c["id"] == agent_id), {})
        self.wait_events.append(
            {
                "time": current_sim_time,
                "agent": agent_id,
                "personality": agent_config.get("personality_type", "UNKNOWN"),
                "wait_reason": wait_reason if wait_reason else "unspecified",
                "location": current_position,
            }
        )
