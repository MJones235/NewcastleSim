"""
Main simulation runner for JuPedSim station simulation.
Handles the simulation loop, event processing, and observer notifications.
Uses observer pattern to decouple GUI and other output mechanisms.

Agent Decision Processing:
- Agents query LLM when system events occur (e.g., announcements)
- Agents also react to messages from other agents with throttling
- Cooldown period prevents excessive LLM calls from peer communication
- All decisions and messages are logged to agent_timelines.json
"""

import asyncio
import json
import random
import time
from pathlib import Path

from scenarios.common.logger import get_logger
from scenarios.station_jupedsim.core.agent_logger import AgentActionLogger
from scenarios.station_jupedsim.core.event_system import EventManager
from scenarios.station_jupedsim.core.simulation_observer import SimulationObserver

logger = get_logger(__name__)


class SimulationRunner:
    """Manages the main simulation execution loop with observer pattern."""

    def __init__(
        self,
        sim,
        agents: list,
        event_manager: EventManager,
        max_iterations: int = 3600,
        spawn_interval: float = 2.0,
        observers: list[SimulationObserver] | None = None,
    ):
        """
        Initialize simulation runner.

        Args:
            sim: StationSimulation instance
            agents: List of StationAgent objects
            event_manager: EventManager for handling timed events
            max_iterations: Maximum simulation iterations
            spawn_interval: Time between agent spawns in seconds
            observers: Optional list of SimulationObserver instances
        """
        self.sim = sim
        self.agents = agents
        self.event_manager = event_manager
        self.max_iterations = max_iterations
        self.spawn_interval = spawn_interval
        self.observers = observers or []

        # Queue agents for spawning - randomize order so entrances are mixed
        self.agents_to_spawn = list(agents)
        random.shuffle(self.agents_to_spawn)
        logger.debug("Agent spawn order randomized")

        # Set up agent-to-agent communication callback
        for agent in agents:
            agent._get_all_agents_callback = lambda: self.agents

        # Initialize structured agent action logger
        self.agent_logger = AgentActionLogger()
        for agent in agents:
            self.agent_logger.register_agent(agent)
            # Set up message logging callback
            if hasattr(agent.decision_maker, "set_message_callback"):
                agent.decision_maker.set_message_callback(self._log_message_received)

        self.last_spawn_time = -spawn_interval  # Allow first spawn immediately
        self.last_event_message: str | None = None
        self.last_event_time = -100.0

        # Track last LLM processing time for throttling peer message reactions
        self.last_llm_processing_time = -100.0
        self.llm_processing_cooldown = 20.0  # Minimum seconds between peer-triggered LLM calls

    def _log_message_received(self, agent_id: str, timestamp: float, message: str, from_agent: str):
        """Callback for logging received messages."""
        self.agent_logger.log_message_received(agent_id, timestamp, message, from_agent)

    def run(self) -> dict:
        """
        Execute the main simulation loop.

        Returns:
            Dictionary with simulation statistics
        """
        # Notify observers simulation is starting
        for observer in self.observers:
            observer.on_simulation_start(len(self.agents))

        # Start timer for real execution time
        start_time = time.time()

        try:
            while self.sim.iteration < self.max_iterations:
                # Spawn one agent if interval has passed and any are waiting
                if self.agents_to_spawn and (
                    self.sim.get_simulation_time() - self.last_spawn_time >= self.spawn_interval
                ):
                    agent = self.agents_to_spawn.pop(0)
                    self.last_spawn_time = self.sim.get_simulation_time()
                    try:
                        if agent.spawn():
                            self.agent_logger.log_spawn(
                                agent.id, self.sim.get_simulation_time(), agent.initial_zone
                            )
                    except Exception as e:
                        logger.error(f"Failed to spawn {agent.id}: {e}")

                # Step simulation (even if no agents yet)
                if not self.sim.step():
                    # Simulation ended - check if we're done
                    if not self.agents_to_spawn and self.sim.simulation.agent_count() == 0:
                        break  # All agents spawned and completed

                sim_time = self.sim.get_simulation_time()
                agent_count = self.sim.simulation.agent_count()

                # Check and trigger events
                triggered_events = self.event_manager.check_and_trigger_events(
                    sim_time, self.agents
                )

                # Store latest event message for observer notifications
                if triggered_events:
                    self.last_event_message = triggered_events[-1].value
                    self.last_event_time = sim_time

                # Update all spawned agents (this queues messages in decision makers)
                for agent in self.agents:
                    if agent.is_spawned:
                        agent.update(sim_time)

                # Process LLM decisions if:
                # 1. System event was triggered (always process), OR
                # 2. Agents have pending messages from peers AND cooldown elapsed
                should_process_llm = False

                if triggered_events and hasattr(self, "_llm_enabled") and self._llm_enabled:
                    should_process_llm = True
                    logger.debug("Event triggered - will process LLM decisions")
                elif hasattr(self, "_llm_enabled") and self._llm_enabled:
                    # Check if cooldown has elapsed since last peer-triggered processing
                    time_since_last = sim_time - self.last_llm_processing_time
                    if time_since_last >= self.llm_processing_cooldown:
                        # Check if any agents have pending messages (from peer broadcasts)
                        from scenarios.common.decision_makers.llm_decision_maker import (
                            LLMDecisionMaker,
                        )

                        agents_with_messages = [
                            agent
                            for agent in self.agents
                            if agent.is_spawned
                            and hasattr(agent, "decision_maker")
                            and isinstance(agent.decision_maker, LLMDecisionMaker)
                            and agent.decision_maker.pending_messages
                        ]
                        if agents_with_messages:
                            should_process_llm = True
                            logger.debug(
                                f"{len(agents_with_messages)} agents have pending peer messages - will process LLM decisions"
                            )

                if should_process_llm:
                    self.last_llm_processing_time = sim_time

                    # Get or create event loop
                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_closed():
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                    except RuntimeError:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)

                    # Run async LLM processing
                    loop.run_until_complete(self._process_llm_decisions())

                # Notify observers of simulation step
                if self.observers:
                    # Get current agent positions directly from JuPedSim
                    agent_positions = []
                    for agent in self.sim.simulation.agents():
                        pos = agent.position
                        agent_positions.append((pos[0], pos[1]))

                    # Prepare metadata
                    event_message = None
                    if self.last_event_message and (sim_time - self.last_event_time) < 5.0:
                        event_message = self.last_event_message

                    spawned_count = sum(1 for a in self.agents if a.is_spawned)
                    metadata = {
                        "event_message": event_message,
                        "spawned_count": spawned_count,
                        "total_agents": len(self.agents),
                    }

                    for observer in self.observers:
                        observer.on_simulation_step(
                            sim_time, self.sim.iteration, agent_count, agent_positions, metadata
                        )

        except KeyboardInterrupt:
            print("\n\nSimulation interrupted by user")

        # Calculate statistics
        end_time = time.time()
        real_time_elapsed = end_time - start_time
        simulated_time = self.sim.get_simulation_time()

        stats = {
            "iterations": self.sim.iteration,
            "simulated_time": simulated_time,
            "real_time": real_time_elapsed,
            "remaining_agents": self.sim.simulation.agent_count(),
            "total_agents": len(self.agents),
        }

        # Log LLM usage stats if enabled
        if hasattr(self, "_llm_enabled") and self._llm_enabled:
            try:
                from scenarios.common.decision_makers.llm_decision_maker import LLMDecisionMaker

                if LLMDecisionMaker._llm_provider:
                    # Close the Azure client properly
                    try:
                        loop = asyncio.get_event_loop()
                        if not loop.is_closed():
                            loop.run_until_complete(LLMDecisionMaker._llm_provider.close())
                    except Exception as e:
                        logger.warning(f"Failed to close LLM client: {e}")

                    usage_stats = LLMDecisionMaker._llm_provider.get_usage_stats()
                    logger.info("=" * 60)
                    logger.info("LLM Token Usage Summary:")
                    logger.info(f"  Total requests: {usage_stats['total_requests']}")
                    logger.info(f"  Prompt tokens: {usage_stats['prompt_tokens']:,}")
                    logger.info(f"  Completion tokens: {usage_stats['completion_tokens']:,}")
                    logger.info(f"  Total tokens: {usage_stats['total_tokens']:,}")
                    logger.info(f"  Estimated cost: £{usage_stats['estimated_cost_gbp']:.4f}")
                    logger.info(f"    Input: £{usage_stats['input_cost_gbp']:.4f}")
                    logger.info(f"    Output: £{usage_stats['output_cost_gbp']:.4f}")
                    logger.info("=" * 60)

                    # Log message history summary
                    self._log_message_history_summary()

                    # Also print to console
                    print("\n" + "=" * 60)
                    print("LLM Token Usage Summary:")
                    print(f"  Total requests: {usage_stats['total_requests']}")
                    print(f"  Prompt tokens: {usage_stats['prompt_tokens']:,}")
                    print(f"  Completion tokens: {usage_stats['completion_tokens']:,}")
                    print(f"  Total tokens: {usage_stats['total_tokens']:,}")
                    print(f"  Estimated cost: £{usage_stats['estimated_cost_gbp']:.4f}")
                    print(f"    Input: £{usage_stats['input_cost_gbp']:.4f}")
                    print(f"    Output: £{usage_stats['output_cost_gbp']:.4f}")
                    print("=" * 60)
            except Exception as e:
                logger.warning(f"Failed to retrieve LLM usage stats: {e}")

        # Save structured agent timelines
        output_dir = Path("scenarios/station_jupedsim/output")
        timeline_file = output_dir / "agent_timelines.json"
        self.agent_logger.save_to_file(timeline_file)
        logger.info(f"Saved agent timelines to {timeline_file}")

        # Log summary stats
        summary = self.agent_logger.get_summary_stats()
        logger.info(f"Agent Action Summary: {summary}")

        # Notify observers simulation ended
        for observer in self.observers:
            observer.on_simulation_end(stats)

        return stats

    def save_events(self, output_dir: Path):
        """
        Save triggered events to JSON file for visualization.

        Args:
            output_dir: Directory to save events file
        """
        events_output = output_dir / "triggered_events.json"
        if self.event_manager.triggered_events:
            events_data = [
                {"time": e.time, "action": e.action, "value": e.value}
                for e in self.event_manager.triggered_events
            ]
            with open(events_output, "w") as f:
                json.dump(events_data, f, indent=2)
            logger.info(f"Saved {len(events_data)} triggered events to {events_output}")
        else:
            # No events triggered - clear any old events file
            if events_output.exists():
                events_output.unlink()
                logger.info("No events triggered - cleared old events file")

    def enable_llm(self):
        """Enable LLM processing for agent decisions."""
        self._llm_enabled = True
        logger.info("LLM processing enabled for simulation")

    def _log_message_history_summary(self):
        """Log summary of message histories across all agents."""
        logger.info("\n" + "=" * 60)
        logger.info("MESSAGE HISTORY SUMMARY")
        logger.info("=" * 60)

        total_messages = 0
        agents_with_messages = 0
        message_sources = {"system": 0, "agents": 0}
        sent_messages = 0

        for agent in self.agents:
            if not hasattr(agent, "decision_maker"):
                continue
            if not hasattr(agent.decision_maker, "message_history"):
                continue

            history = agent.decision_maker.message_history
            if len(history) > 0:
                agents_with_messages += 1
                total_messages += len(history)

                # Count sources and types
                for msg in history:
                    if msg.get("type") == "sent":
                        sent_messages += 1
                    elif msg.get("from") == "system":
                        message_sources["system"] += 1
                    else:
                        message_sources["agents"] += 1

        logger.info(f"Total agents: {len(self.agents)}")
        logger.info(f"Agents who received/sent messages: {agents_with_messages}")
        logger.info(f"Total messages received: {total_messages - sent_messages}")
        logger.info(f"  From system: {message_sources['system']}")
        logger.info(f"  From other agents: {message_sources['agents']}")
        logger.info(f"Total messages sent by agents: {sent_messages}")

        # Sample a few agent histories
        if agents_with_messages > 0:
            logger.info("\nSample Message Histories (first 3 agents):")
            count = 0
            for agent in self.agents:
                if not hasattr(agent, "decision_maker"):
                    continue
                if not hasattr(agent.decision_maker, "message_history"):
                    continue

                history = agent.decision_maker.message_history
                if len(history) > 0:
                    logger.info(f"\n  Agent {agent.id}:")
                    for msg in history:
                        if msg.get("type") == "sent":
                            logger.info(
                                f"    [{msg['time']:.1f}s] SENT to {msg.get('to', 'unknown')}: "
                                f"\"{msg['message'][:60]}...\" ({msg.get('recipients', 0)} recipients)"
                            )
                        else:
                            logger.info(
                                f"    [{msg['time']:.1f}s] {msg['type']} "
                                f"FROM {msg['from']}: \"{msg['message'][:60]}...\""
                            )
                    count += 1
                    if count >= 3:
                        break

        logger.info("=" * 60)

    async def _process_llm_decisions(self):
        """
        Process LLM decisions for agents with pending messages.

        This is called after events are broadcast to batch process all
        agents who need LLM decisions.
        """
        try:
            # Import here to avoid circular dependency
            from scenarios.common.decision_makers.llm_decision_maker import LLMDecisionMaker
            from scenarios.common.llm.llm_provider import LLMResponse

            # Find agents with LLM decision makers who have pending messages
            agents_needing_llm = [
                agent
                for agent in self.agents
                if agent.is_spawned
                and hasattr(agent, "decision_maker")
                and isinstance(agent.decision_maker, LLMDecisionMaker)
                and agent.decision_maker.pending_messages
            ]

            logger.debug(f"Found {len(agents_needing_llm)} agents with pending LLM messages")

            if not agents_needing_llm:
                logger.debug("No agents needing LLM processing - returning")
                return

            logger.info(f"Processing {len(agents_needing_llm)} agents with LLM")

            # Batch process all agents with LLM
            decisions = await LLMDecisionMaker.batch_process_agents(agents_needing_llm)

            # Apply decisions
            sim_time = self.sim.get_simulation_time()
            for agent in agents_needing_llm:
                decision = decisions.get(agent.id, {"action": "continue"})
                response_obj = decisions.get(agent.id + "_response")
                response: LLMResponse | None = (
                    response_obj if isinstance(response_obj, LLMResponse) else None
                )

                # Log the decision
                if response and hasattr(response, "decision"):
                    self.agent_logger.log_decision(
                        agent.id,
                        sim_time,
                        decision=response.decision,
                        reasoning=response.reasoning,
                        confidence=response.confidence,
                        full_response={
                            "decision": response.decision,
                            "reasoning": response.reasoning,
                            "confidence": response.confidence,
                            "broadcast_message": response.broadcast_message,
                            "broadcast_radius": response.broadcast_radius,
                        },
                    )

                if decision["action"] == "evacuate":
                    # Trigger evacuation for this agent
                    if hasattr(agent, "start_evacuation"):
                        old_state = "waiting" if not agent.is_evacuating else "evacuating"
                        agent.start_evacuation()
                        self.agent_logger.log_state_change(
                            agent.id,
                            sim_time,
                            old_state=old_state,
                            new_state="evacuating",
                            reason=f"LLM decision: {decision.get('reasoning', 'N/A')[:100]}",
                        )
                        logger.info(
                            f"Agent {agent.id} deciding to evacuate "
                            f"(confidence: {decision.get('confidence', 0.0):.2f})"
                        )

                # Phase 2: Execute broadcast actions
                if (
                    response
                    and hasattr(response, "broadcast_message")
                    and response.broadcast_message
                ):
                    num_recipients = agent.send_message_to_nearby(
                        response.broadcast_message, response.broadcast_radius or 10.0
                    )

                    # Log sent message to agent's history
                    if hasattr(agent.decision_maker, "message_history"):
                        agent.decision_maker.message_history.append(
                            {
                                "time": sim_time,
                                "type": "sent",
                                "to": "broadcast",
                                "message": response.broadcast_message,
                                "recipients": num_recipients,
                                "radius": response.broadcast_radius,
                            }
                        )

                    # Log to structured logger
                    self.agent_logger.log_message_sent(
                        agent.id,
                        sim_time,
                        message=response.broadcast_message,
                        recipients_count=num_recipients,
                        broadcast_radius=response.broadcast_radius or 10.0,
                    )

                    logger.info(
                        f"Agent {agent.id} broadcast message to {num_recipients} agents "
                        f"(radius={response.broadcast_radius}m): '{response.broadcast_message}'"
                    )

        except Exception as e:
            logger.error(f"LLM processing failed: {e}", exc_info=True)
