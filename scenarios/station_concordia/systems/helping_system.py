"""
Helping System Manager

Manages helping relationships between agents during evacuation, including:
- Two-phase approach: approaching (helper moves to injured agent) and traveling (both move together)
- Speed synchronization between helper and helped agents
- Duration tracking and relationship expiration
- Abandonment detection when helpers change status

This module handles the complex logic of coordinating helper-helped pairs
in the evacuation simulation.
"""

from typing import Any

from scenarios.common.logger import get_logger

logger = get_logger(__name__)


class HelpingSystemManager:
    """Manages active helping relationships and their lifecycle."""

    def __init__(
        self,
        active_helping_pairs: dict[str, dict[str, Any]],
        agents_being_helped: dict[str, str],
        agent_original_speeds: dict[str, float],
        agent_status: dict[str, str],
        agent_destinations: dict[str, str],
        exited_agents: set[str],
        test_scenarios: dict[str, Any],
        jps_sim,
        state_queries,
    ):
        """
        Initialize helping system manager.

        Args:
            active_helping_pairs: helper_id -> {helped, start_time, duration, phase}
            agents_being_helped: helped_agent_id -> helper_agent_id
            agent_original_speeds: agent_id -> original speed (m/s)
            agent_status: agent_id -> status string (EVACUATING|HELPING|WAITING|INJURED)
            agent_destinations: agent_id -> current exit name
            exited_agents: Set of agent IDs who have exited
            test_scenarios: Test scenario configuration dict
            jps_sim: JuPedSim simulation instance
            state_queries: Simulation state query interface
        """
        self.active_helping_pairs = active_helping_pairs
        self.agents_being_helped = agents_being_helped
        self.agent_original_speeds = agent_original_speeds
        self.agent_status = agent_status
        self.agent_destinations = agent_destinations
        self.exited_agents = exited_agents
        self.test_scenarios = test_scenarios
        self.jps_sim = jps_sim
        self.state_queries = state_queries

    def update_helping_relationships(self, current_sim_time: float):
        """Update active helping relationships and release them when duration expires."""
        if not self.active_helping_pairs:
            return

        expired_pairs = []
        for helper_id, pair_info in self.active_helping_pairs.items():
            helped_id = pair_info["helped"]
            start_time = pair_info["start_time"]
            duration = pair_info["duration"]
            phase = pair_info.get("phase", "traveling")

            # Phase 1: Approaching - check if helper has reached injured agent
            if phase == "approaching":
                helper_pos = self.state_queries.get_agent_position(helper_id)
                injured_pos = self.state_queries.get_agent_position(helped_id)

                # Skip if either agent has exited
                if helper_pos is None or injured_pos is None:
                    expired_pairs.append(helper_id)
                    continue

                # Calculate distance between helper and injured agent
                distance = (
                    (helper_pos[0] - injured_pos[0]) ** 2 + (helper_pos[1] - injured_pos[1]) ** 2
                ) ** 0.5

                # Get approach distance threshold from config
                help_config = self.test_scenarios.get("help_behavior", {})
                approach_distance = help_config.get("approach_distance", 1.5)

                # Injured agent should wait (speed = 0) for helper to arrive
                self.jps_sim.set_agent_speed(helped_id, 0.0)

                # Log progress occasionally
                if int(current_sim_time) % 5 == 0:  # Every 5 seconds
                    logger.debug(
                        f"🚶 {helper_id} approaching {helped_id}: distance={distance:.1f}m "
                        f"(threshold={approach_distance}m)"
                    )

                # If within approach distance, transition to traveling phase
                if distance < approach_distance:
                    pair_info["phase"] = "traveling"

                    # Phase 2: Traveling together
                    # Get assisted speed from config
                    help_config = self.test_scenarios.get("help_behavior", {})
                    assisted_speed = help_config.get("assisted_speed", 0.8)
                    self.jps_sim.set_agent_speed(helper_id, assisted_speed)
                    self.jps_sim.set_agent_speed(helped_id, assisted_speed)

                    # Both agents target the same exit (helper's current destination)
                    helper_exit = self.agent_destinations.get(helper_id)
                    if helper_exit:
                        self.jps_sim.set_agent_evacuation_exit(helped_id, helper_exit)
                        logger.debug(f"Set {helped_id} to follow {helper_id} to {helper_exit}")

                    logger.info(
                        f"🚶 {helper_id} reached {helped_id} - "
                        f"now traveling together at {assisted_speed} m/s toward {helper_exit}"
                    )

            # Phase 2: Traveling - continuously update helped agent to follow helper
            if phase == "traveling":
                # Check if helper has exited - if so, terminate helping relationship
                if helper_id in self.exited_agents:
                    logger.info(
                        f"🚪 {helper_id} has exited, releasing {helped_id} to evacuate independently"
                    )
                    expired_pairs.append(helper_id)

                    # Restore helped agent's speed and status
                    if helped_id in self.agent_original_speeds:
                        original_speed = self.agent_original_speeds[helped_id]
                        self.jps_sim.set_agent_speed(helped_id, original_speed)
                        logger.debug(f"Restored {helped_id} speed to {original_speed} m/s")

                    self.agent_status[helped_id] = "INJURED"

                    # CRITICAL: Remove from being helped tracking immediately
                    # This must happen before we continue to ensure visual indicators update
                    if helped_id in self.agents_being_helped:
                        del self.agents_being_helped[helped_id]
                        logger.debug(f"Removed {helped_id} from agents_being_helped")

                    # Skip to next pair (don't try to update positions)
                    continue

                # Check if helper has changed status (e.g., stopped helping, waiting)
                # If helper is no longer in HELPING status, abandon the helped agent
                helper_status = self.agent_status.get(helper_id)
                if helper_status and helper_status != "HELPING":
                    logger.warning(
                        f"⚠️ {helper_id} changed status to {helper_status}, abandoning {helped_id}"
                    )
                    expired_pairs.append(helper_id)

                    # Restore helped agent
                    if helped_id in self.agent_original_speeds:
                        original_speed = self.agent_original_speeds[helped_id]
                        self.jps_sim.set_agent_speed(helped_id, original_speed)

                    self.agent_status[helped_id] = "INJURED"

                    if helped_id in self.agents_being_helped:
                        del self.agents_being_helped[helped_id]

                    continue

                # Ensure helper maintains their evacuation journey and assisted speed
                help_config = self.test_scenarios.get("help_behavior", {})
                assisted_speed = help_config.get("assisted_speed", 0.8)
                self.jps_sim.set_agent_speed(helper_id, assisted_speed)
                self.jps_sim.set_agent_speed(helped_id, assisted_speed)

                # Keep helped agent following helper by setting their target to helper's current position
                helper_pos = self.state_queries.get_agent_position(helper_id)

                # Check if helper has exited
                if helper_pos is None:
                    logger.warning(f"⚠️ {helper_id} has exited, releasing {helped_id}")
                    expired_pairs.append(helper_id)
                    continue

                self.jps_sim.set_agent_target(helped_id, helper_pos)

                # Log occasionally to verify they're staying together
                if int(current_sim_time) % 5 == 0:  # Every 5 seconds
                    helped_pos = self.state_queries.get_agent_position(helped_id)
                    if helped_pos is not None:
                        distance = (
                            (helper_pos[0] - helped_pos[0]) ** 2
                            + (helper_pos[1] - helped_pos[1]) ** 2
                        ) ** 0.5
                        logger.debug(
                            f"👥 {helper_id} and {helped_id} traveling together (distance: {distance:.1f}m)"
                        )

            # Check if help duration has expired (only for traveling phase)
            if phase == "traveling" and current_sim_time >= start_time + duration:
                expired_pairs.append(helper_id)

                # Restore original speeds using JuPedSim agent.model.v0
                if helper_id in self.agent_original_speeds:
                    original_speed = self.agent_original_speeds[helper_id]
                    self.jps_sim.set_agent_speed(helper_id, original_speed)

                if helped_id in self.agent_original_speeds:
                    original_speed = self.agent_original_speeds[helped_id]
                    self.jps_sim.set_agent_speed(helped_id, original_speed)

                # Update statuses back to EVACUATING/INJURED
                self.agent_status[helper_id] = "EVACUATING"
                self.agent_status[helped_id] = "INJURED"  # Still injured but now independent

                # Remove from being helped tracking
                if helped_id in self.agents_being_helped:
                    del self.agents_being_helped[helped_id]

                logger.info(
                    f"👋 {helper_id} finished helping {helped_id} - "
                    f"both resuming independent evacuation"
                )

        # Remove expired pairs
        for helper_id in expired_pairs:
            del self.active_helping_pairs[helper_id]

    def check_abandoned_helps(self):
        """
        Check if any helpers abandoned their helping commitment.

        Helpers can make free decisions, but if they change away from HELPING status,
        they have abandoned their commitment. Release the helped agent.

        Returns:
            List of helper IDs who abandoned their commitment
        """
        abandoned_helps = []
        for helper_id, pair_info in list(self.active_helping_pairs.items()):
            # Check if helper's status is still HELPING
            if self.agent_status.get(helper_id) != "HELPING":
                helped_id = pair_info["helped"]
                logger.warning(
                    f"⚠️ {helper_id} abandoned helping {helped_id} "
                    f"(status changed from HELPING to {self.agent_status.get(helper_id)}) - "
                    f"releasing {helped_id} to independent movement"
                )
                abandoned_helps.append(helper_id)

        # Remove abandoned helping relationships
        for helper_id in abandoned_helps:
            pair_info = self.active_helping_pairs[helper_id]
            helped_id = pair_info["helped"]

            # Remove from tracking
            if helped_id in self.agents_being_helped:
                del self.agents_being_helped[helped_id]

            # Restore helped agent's original status (INJURED)
            self.agent_status[helped_id] = "INJURED"

            # Restore helped agent's speed
            if helped_id in self.agent_original_speeds:
                original_speed = self.agent_original_speeds[helped_id]
                self.jps_sim.set_agent_speed(helped_id, original_speed)

            # Remove the helping pair
            del self.active_helping_pairs[helper_id]

        return abandoned_helps
