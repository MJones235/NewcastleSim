"""
Action Executor

Executes translated agent actions in the JuPedSim simulation, including:
- Speed adjustments
- Movement to exits, waypoints, or toward other agents
- Waiting behavior (standing still or seeking information)
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
        agent_last_decision: dict[str, dict],
        agent_destinations: dict[str, str],
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
            agent_last_decision: agent_id -> last translated_action dict (for memory)
            agent_destinations: agent_id -> current exit name
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
        self.agent_last_decision = agent_last_decision
        self.agent_destinations = agent_destinations
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
            target_type = translated_action.get("target_type", "")

            if target_agent_id and action_type == "move":
                target_position = self.state_queries.get_agent_position(target_agent_id)
                if target_position is not None:
                    target = target_position
                    translated_action["target"] = target

                    # For "following" type, optionally match speed for coordinated movement
                    if target_type == "agent":
                        logger.debug(f"{agent_id} following {target_agent_id} at {target}")

                        # Check if close enough to match speed (within 10m)
                        agent_pos = self.state_queries.get_agent_position(agent_id)
                        if agent_pos is not None:
                            distance = (
                                (agent_pos[0] - target_position[0]) ** 2
                                + (agent_pos[1] - target_position[1]) ** 2
                            ) ** 0.5

                            # Match speed if close enough and no explicit speed set
                            if distance < 10.0 and not speed_str:
                                try:
                                    target_speed = self.jps_sim.get_agent_speed(target_agent_id)
                                    if target_speed:
                                        self.jps_sim.set_agent_speed(agent_id, target_speed)
                                        logger.debug(
                                            f"{agent_id} matching {target_agent_id}'s speed: "
                                            f"{target_speed:.2f} m/s (distance: {distance:.1f}m)"
                                        )
                                except Exception:
                                    pass  # If speed matching fails, continue with default
                    else:
                        # Regular agent approach (not following)
                        logger.debug(f"{agent_id} moving toward {target_agent_id} at {target}")

            if action_type == "move" and target:
                self._handle_move_action(agent_id, translated_action, target)
            elif action_type == "wait":
                self._handle_wait_action(agent_id, translated_action, current_sim_time)

            # Store decision for agent memory (Concordia natural language context)
            # This allows agents to reference their previous commitment in future decisions
            self.agent_last_decision[agent_id] = {
                "time": current_sim_time,
                "action_type": action_type,
                "target_type": translated_action.get("target_type"),
                "target_agent": translated_action.get("target_agent"),
                "target_exit": translated_action.get("target_exit"),
                "zone_name": translated_action.get("zone_name"),
                "speed": translated_action.get("speed"),
                "reasoning": translated_action.get("reasoning"),
                "wait_reason": translated_action.get("wait_reason"),
            }

        except Exception as e:
            logger.error(f"Failed to apply action for {agent_id}: {e}")

    def _handle_move_action(self, agent_id: str, translated_action: dict[str, Any], target):
        """Handle move action: agent moving to exit, waypoint, or toward another agent."""
        # Update action state
        self.agent_action[agent_id] = "moving"

        target_agent = translated_action.get("target_agent")
        target_type = translated_action.get("target_type", "")

        if target_agent and target_type == "agent":
            # Agent is following someone - this is coordinated movement, not helping detection
            logger.debug(f"{agent_id} following {target_agent}")
        elif target_agent:
            # Regular agent approach (not following)
            logger.debug(f"{agent_id} moving toward {target_agent}")

        # Extract the NEW exit name from this action (if moving to an exit)
        new_exit_name = extract_exit_name(translated_action, self.station_layout)

        if new_exit_name:
            # Check if agent is trying to switch to a blocked exit
            if new_exit_name in self.event_manager.blocked_exits:
                logger.warning(
                    f"⚠️ {agent_id} tried to switch to blocked exit '{new_exit_name}' - "
                    f"setting waypoint only, NOT updating journey or destination tracking"
                )
                # Only set waypoint, don't switch journey (would let them evacuate through blocked exit)
                # Do NOT update agent_destinations - they haven't actually changed their route
                self.jps_sim.set_agent_target(agent_id, target)
            else:
                # Valid exit - update destination tracking ONLY for non-blocked exits
                self.agent_destinations[agent_id] = new_exit_name

                # Switch the agent's evacuation journey to this exit
                self.jps_sim.set_agent_evacuation_exit(agent_id, new_exit_name)
                logger.debug(f"Switched {agent_id} to journey for {new_exit_name}")
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
