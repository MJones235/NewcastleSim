"""
Action Executor

Executes translated agent actions in the JuPedSim simulation, including:
- Speed adjustments
- Help behavior initialization (approaching and assisting injured agents)
- Movement to exits or waypoints
- Waiting behavior (standing still or seeking information)

This module handles the complex logic of applying high-level agent decisions
to low-level JuPedSim simulation controls.
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
        agent_status: dict[str, str],
        agents_being_helped: dict[str, str],
        agent_destinations: dict[str, str],
        active_helping_pairs: dict[str, dict[str, Any]],
        agent_original_speeds: dict[str, float],
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
            agent_status: agent_id -> status (EVACUATING|HELPING|WAITING|INJURED)
            agents_being_helped: helped_agent_id -> helper_agent_id
            agent_destinations: agent_id -> current exit name
            active_helping_pairs: helper_id -> {helped, start_time, duration, phase}
            agent_original_speeds: agent_id -> original speed (m/s)
            help_events: List tracking all help interactions
            wait_events: List tracking all wait decisions with reasons
            agent_configs: List of agent configuration dictionaries
            test_scenarios: Test scenario configuration
        """
        self.jps_sim = jps_sim
        self.state_queries = state_queries
        self.event_manager = event_manager
        self.station_layout = station_layout
        self.agent_status = agent_status
        self.agents_being_helped = agents_being_helped
        self.agent_destinations = agent_destinations
        self.active_helping_pairs = active_helping_pairs
        self.agent_original_speeds = agent_original_speeds
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

            if action_type == "help":
                self._handle_help_action(agent_id, translated_action, current_sim_time)
            elif action_type == "move" and target:
                self._handle_move_action(agent_id, translated_action, target)
            elif action_type == "wait":
                self._handle_wait_action(agent_id, translated_action, current_sim_time)

        except Exception as e:
            logger.error(f"Failed to apply action for {agent_id}: {e}")

    def _handle_help_action(
        self, agent_id: str, translated_action: dict[str, Any], current_sim_time: float
    ):
        """Handle help action: agent assisting an injured person."""
        # Agent is helping an injured person
        self.agent_status[agent_id] = "HELPING"

        # Get observation radius from config
        help_config = self.test_scenarios.get("help_behavior", {})
        observation_radius = help_config.get("observation_radius", 20.0)

        # Find nearest injured agent within observation radius
        position = self.state_queries.get_agent_position(agent_id)
        nearby_agents = self.state_queries.get_nearby_agents(agent_id, radius=observation_radius)

        injured_nearby = None
        for agent_info in nearby_agents:
            other_id = agent_info.get("id")
            if other_id and self.agent_status.get(other_id) == "INJURED":
                # Check if this injured agent is already being helped
                if other_id not in self.agents_being_helped:
                    injured_nearby = other_id
                    break

        if injured_nearby:
            # Record help event
            helper_config = next((c for c in self.agent_configs if c["id"] == agent_id), {})

            # Get help configuration from test_scenarios
            help_config = self.test_scenarios.get("help_behavior", {})
            help_duration = help_config.get("help_duration", 15.0)

            # Get injured agent's position
            injured_position = self.state_queries.get_agent_position(injured_nearby)

            self.help_events.append(
                {
                    "time": current_sim_time,
                    "helper": agent_id,
                    "helped": injured_nearby,
                    "helper_personality": helper_config.get("personality_type", "UNKNOWN"),
                    "location": position,
                    "duration": help_duration,
                }
            )

            # Track active helping pair with two phases:
            # 1. "approaching" - helper walks to injured agent (who stops)
            # 2. "traveling" - both travel together at intermediate speed
            self.active_helping_pairs[agent_id] = {
                "helped": injured_nearby,
                "start_time": current_sim_time,
                "duration": help_duration,
                "phase": "approaching",
                "injured_position": injured_position,
            }

            # Update helped agent's status to WAITING (receiving assistance)
            self.agents_being_helped[injured_nearby] = agent_id
            self.agent_status[injured_nearby] = "WAITING"

            # Store original speeds if not already stored
            help_config = self.test_scenarios.get("help_behavior", {})
            if agent_id not in self.agent_original_speeds:
                normal_speed = help_config.get("normal_walking_speed", 1.34)
                self.agent_original_speeds[agent_id] = normal_speed
            if injured_nearby not in self.agent_original_speeds:
                injured_speed = help_config.get("injured_walking_speed", 0.5)
                self.agent_original_speeds[injured_nearby] = injured_speed

            # Phase 1: Approaching
            # - Injured agent stops (speed = 0) so helper can reach them
            # - Helper walks toward injured agent's current position
            self.jps_sim.set_agent_speed(injured_nearby, 0.0)
            self.jps_sim.set_agent_target(agent_id, injured_position)

            distance = (
                (position[0] - injured_position[0]) ** 2 + (position[1] - injured_position[1]) ** 2
            ) ** 0.5

            logger.info(
                f"🤝 {agent_id} is approaching {injured_nearby} to help (distance: {distance:.1f}m)"
            )
        else:
            # Check if there were injured agents but all already being helped
            injured_already_helped = [
                agent_info.get("id")
                for agent_info in nearby_agents
                if agent_info.get("id")
                and self.agent_status.get(agent_info.get("id")) == "INJURED"
                and agent_info.get("id") in self.agents_being_helped
            ]
            if injured_already_helped:
                logger.info(
                    f"ℹ️ {agent_id} wanted to help but all nearby injured agents "
                    f"already being helped: {injured_already_helped}"
                )
            else:
                logger.warning(f"{agent_id} wanted to help but no injured agents nearby")

    def _handle_move_action(self, agent_id: str, translated_action: dict[str, Any], target):
        """Handle move action: agent moving to exit or waypoint."""
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
        current_position = self.state_queries.get_agent_position(agent_id)
        wait_reason = translated_action.get("wait_reason", "unspecified")

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
