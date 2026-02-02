"""
Hybrid simulation runner that integrates Concordia with JuPedSim.

This module implements the translation layer between:
- Concordia: Agent cognition and decision-making
- JuPedSim: Pedestrian movement simulation

Key features:
- Event-driven LLM queries (not every timestep)
- Batch processing of agent decisions
- Translation of NL actions to waypoints
- Observation generation from simulation state
"""

import json
import time
from pathlib import Path
from typing import Any

from concordia.associative_memory import basic_associative_memory
from concordia.language_model import language_model
from concordia.typing import entity as entity_lib
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

try:
    import jupedsim as jps
except ImportError:
    jps = None
    print("Warning: jupedsim not available")

from scenarios.common.logger import get_logger
from scenarios.station_concordia.core.evacuation_agent import EvacuationAgent
from scenarios.station_concordia.core.game_master import ActionTranslator, ObservationGenerator

logger = get_logger(__name__)


class HybridSimulationRunner:
    """
    Manages the hybrid Concordia + JuPedSim simulation.

    Architecture:
    1. JuPedSim runs continuously at fine time resolution (dt=0.05s)
    2. Concordia agents make decisions at coarse intervals (5-10s)
    3. Decisions are triggered by events (announcements, observations)
    4. Actions are translated to JuPedSim waypoints
    5. Simulation state is converted to observations for agents
    """

    def __init__(
        self,
        jupedsim_simulation: Any,  # StationSimulation instance
        agents_config: list[dict[str, Any]],
        station_layout: dict[str, Any],
        language_model: language_model.LanguageModel,
        embedder: Any,  # Sentence embedder function
        decision_interval: float = 5.0,
        max_steps: int = 3600,
        output_file: Path | None = None,
    ):
        """
        Initialize the hybrid simulation runner.

        Args:
            jupedsim_simulation: Configured JuPedSim simulation
            agents_config: List of agent configuration dictionaries
            station_layout: Station geometry and exit information
            language_model: LLM for Concordia agents
            embedder: Sentence embedding function
            decision_interval: Time between Concordia decisions (seconds)
            max_steps: Maximum simulation steps
        """
        self.jps_sim = jupedsim_simulation
        self.station_layout = station_layout
        self.model = language_model
        self.embedder = embedder
        self.decision_interval = decision_interval
        self.max_steps = max_steps
        self.output_file = output_file

        # Translation layer components
        self.action_translator = ActionTranslator(station_layout, language_model)
        self.observation_generator = ObservationGenerator(station_layout)

        # Create memory bank for all agents
        logger.info("Creating memory bank with sentence embedder...")

        self.memory_bank = basic_associative_memory.AssociativeMemoryBank(
            sentence_embedder=embedder
        )

        # Build Concordia agents
        self.concordia_agents: dict[str, entity_lib.Entity] = {}
        self.agent_configs = agents_config

        self._build_agents()

        # Tracking
        self.last_decision_time = (
            -decision_interval
        )  # Start negative so first decision happens immediately
        self.current_sim_time = 0.0
        self.agent_decisions: dict[str, dict[str, Any]] = {}
        self.event_history: list[dict[str, Any]] = []

    def _build_agents(self):
        """Build Concordia agents from configurations."""
        logger.info(f"Building {len(self.agent_configs)} Concordia agents...")

        for agent_config in self.agent_configs:
            agent_id = agent_config["id"]
            logger.info(f"Building {agent_id}...")

            # Create agent prefab
            prefab = EvacuationAgent(params=agent_config)

            # Build agent
            agent = prefab.build(
                model=self.model,
                memory_bank=self.memory_bank,
            )

            self.concordia_agents[agent_id] = agent

            # Add initial memories
            self._initialize_agent_memory(agent, agent_config)

        logger.info(f"Built {len(self.concordia_agents)} Concordia agents")

    def _initialize_agent_memory(self, agent: entity_lib.Entity, config: dict[str, Any]):
        """Initialize an agent's memory with background knowledge."""
        initial_memories = [
            "I am at a train station.",
            f"I am in the {config.get('initial_zone', 'platform')} area.",
            f"My destination is {config.get('destination', 'exit')}.",
            "The station has multiple exits and clearly marked signs.",
            "I am waiting for my train and observing my surroundings.",
        ]

        for memory in initial_memories:
            agent.observe(memory)

    def run(self) -> dict[str, Any]:
        """
        Run the hybrid simulation.

        Returns:
            Dictionary with simulation results and statistics
        """
        logger.info("Starting hybrid Concordia + JuPedSim simulation")
        start_time = time.time()

        results = {
            "steps": 0,
            "sim_time": 0.0,
            "decisions_made": 0,
            "events_triggered": 0,
            "agents": {},
        }

        try:
            # Main simulation loop with progress bar
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]Simulating:"),
                BarColumn(complete_style="green", finished_style="bold green"),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TextColumn("•"),
                TextColumn("Step {task.completed}/{task.total}"),
                TextColumn("•"),
                TimeElapsedColumn(),
                TextColumn("•"),
                TimeRemainingColumn(),
            ) as progress:
                task = progress.add_task("simulation", total=self.max_steps)

                for step in range(self.max_steps):
                    # Advance JuPedSim simulation
                    if not self._step_jupedsim():
                        logger.info("JuPedSim simulation complete")
                        break

                    self.current_sim_time = step * self.jps_sim.dt

                    # Check if it's time for Concordia decisions
                    if self._should_make_decisions():
                        self._process_agent_decisions()

                    # Check for events
                    self._check_and_trigger_events()

                    results["steps"] = step + 1
                    results["sim_time"] = self.current_sim_time

                    # Update progress bar
                    progress.update(task, advance=1)

        except KeyboardInterrupt:
            logger.info("Simulation interrupted by user")
        except Exception as e:
            logger.error(f"Simulation error: {e}", exc_info=True)

        # Compute final statistics
        elapsed_time = time.time() - start_time
        results["elapsed_time"] = elapsed_time
        results["decisions_made"] = sum(
            len(d.get("decisions", [])) for d in self.agent_decisions.values()
        )
        results["events_triggered"] = len(self.event_history)

        logger.info(
            f"Simulation complete: {results['steps']} steps, "
            f"{results['sim_time']:.1f}s sim time, "
            f"{elapsed_time:.1f}s real time"
        )

        return results

    def _step_jupedsim(self) -> bool:
        """
        Advance JuPedSim simulation by one timestep.

        Returns:
            True if simulation should continue, False if complete
        """
        try:
            # TODO: This works with MockJuPedSim, verify with real JuPedSim
            return self.jps_sim.step()
        except Exception as e:
            logger.error(f"JuPedSim step error: {e}")
            return False

    def _should_make_decisions(self) -> bool:
        """Check if it's time for agents to make decisions."""
        return (self.current_sim_time - self.last_decision_time) >= self.decision_interval

    def _process_agent_decisions(self):
        """Process decision-making for all agents (batch processing)."""
        logger.info(f"Agent decisions at t={self.current_sim_time:.1f}s")

        # Generate observations for all agents
        observations = self._generate_observations()

        # Process each agent
        for agent_id, agent in self.concordia_agents.items():
            try:
                # Provide observation to agent
                observation = observations.get(agent_id, "")
                agent.observe(observation)

                # Get agent's action
                action_spec = entity_lib.ActionSpec(
                    call_to_action="What will you do next in this evacuation?",
                    output_type=entity_lib.OutputType.FREE,
                )
                action = agent.act(action_spec)

                # Extract component states for display
                reasoning = self._extract_agent_reasoning(agent)

                # Translate action to JuPedSim command
                position = self._get_agent_position(agent_id)
                translated = self.action_translator.translate(agent_id, action, position)

                # Store decision
                if agent_id not in self.agent_decisions:
                    self.agent_decisions[agent_id] = {"decisions": []}

                self.agent_decisions[agent_id]["decisions"].append(
                    {
                        "time": self.current_sim_time,
                        "action": action,
                        "reasoning": reasoning,
                        "translated": translated,
                    }
                )

                # Save incrementally for live viewing
                if self.output_file:
                    self._save_incremental()

                # Apply to JuPedSim
                self._apply_action_to_jupedsim(agent_id, translated)

                logger.info(f"{agent_id} action: {action}")

            except Exception as e:
                logger.error(f"Error processing {agent_id}: {e}", exc_info=True)

        self.last_decision_time = self.current_sim_time

    def _extract_agent_reasoning(self, agent: entity_lib.Entity) -> dict[str, str]:
        """
        Extract reasoning from agent's components for display.

        Returns dict with component outputs.
        """
        reasoning = {}

        try:
            # Try to get component states
            components = agent.get_component_states()

            # Extract key reasoning components
            if "SelfPerception" in components:
                reasoning["self_perception"] = components["SelfPerception"]
            if "SituationPerception" in components:
                reasoning["situation"] = components["SituationPerception"]
            if "RiskPerception" in components:
                reasoning["risk"] = components["RiskPerception"]
            if "SocialInfluence" in components:
                reasoning["social"] = components["SocialInfluence"]
            if "PersonBySituation" in components:
                reasoning["strategy"] = components["PersonBySituation"]

        except Exception as e:
            logger.debug(f"Could not extract reasoning: {e}")

        return reasoning

    def _generate_observations(self) -> dict[str, str]:
        """Generate observations for all agents based on simulation state."""
        observations = {}

        for agent_id in self.concordia_agents.keys():
            try:
                # Get agent state from JuPedSim
                position = self._get_agent_position(agent_id)
                nearby_agents = self._get_nearby_agents(agent_id, radius=10.0)
                recent_events = self._get_recent_events()

                # Generate observation
                obs = self.observation_generator.generate_observation(
                    agent_id=agent_id,
                    position=position,
                    nearby_agents=nearby_agents,
                    events=recent_events,
                    sim_time=self.current_sim_time,
                )

                observations[agent_id] = obs

            except Exception as e:
                logger.error(f"Error generating observation for {agent_id}: {e}")
                observations[agent_id] = (
                    f"[Time: {self.current_sim_time:.1f}s] You are in the station."
                )

        return observations

    def _get_agent_position(self, agent_id: str) -> tuple[float, float]:
        """Get agent's current position from JuPedSim."""
        # TODO: Verify this works with real JuPedSim simulation
        try:
            return self.jps_sim.get_agent_position(agent_id)
        except Exception as e:
            logger.warning(f"Failed to get position for {agent_id}: {e}")
            return (0.0, 0.0)

    def _get_nearby_agents(self, agent_id: str, radius: float) -> list[dict[str, Any]]:
        """Get information about nearby agents."""
        # TODO: Verify this works with real JuPedSim simulation
        try:
            return self.jps_sim.get_nearby_agents(agent_id, radius)
        except Exception as e:
            logger.warning(f"Failed to get nearby agents for {agent_id}: {e}")
            return []

    def _get_recent_events(self) -> list[str]:
        """Get recent events relevant to agents."""
        # Return last few events
        return [e["message"] for e in self.event_history[-3:]]

    def _apply_action_to_jupedsim(self, agent_id: str, translated_action: dict[str, Any]):
        """Apply a translated action to the JuPedSim simulation."""
        action_type = translated_action["action_type"]
        target = translated_action["target"]

        logger.info(
            f"Agent {agent_id}: {action_type} to {target} "
            f"(confidence: {translated_action['confidence']:.2f}) - {translated_action['reasoning']}"
        )

        # TODO: With real JuPedSim, use proper waypoint/goal setting API
        # For now, using simple target setting
        try:
            if action_type == "move" and target:
                self.jps_sim.set_agent_target(agent_id, target)
            # "wait" actions don't need to set a new target
        except Exception as e:
            logger.error(f"Failed to apply action for {agent_id}: {e}")

    def _check_and_trigger_events(self):
        """Check for and trigger simulation events."""
        # TODO: Implement event triggering logic
        # Events could include:
        # - Announcements at specific times
        # - Alarms based on conditions
        # - Dynamic obstacles
        pass

    def broadcast_event(self, event_message: str):
        """
        Broadcast an event to all agents.

        Args:
            event_message: The event message to broadcast
        """
        logger.info(f"Broadcasting event: {event_message}")

        # Store event
        self.event_history.append(
            {
                "time": self.current_sim_time,
                "message": event_message,
            }
        )

        # Notify all agents
        for agent in self.concordia_agents.values():
            agent.observe(f"[ANNOUNCEMENT] {event_message}")

    def _save_incremental(self):
        """Save current results incrementally for live viewing."""
        if not self.output_file:
            return

        results = {
            "agent_decisions": self.agent_decisions,
            "events": self.event_history,
            "config": {
                "decision_interval": self.decision_interval,
                "max_steps": self.max_steps,
                "num_agents": len(self.concordia_agents),
            },
        }

        try:
            self.output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.output_file, "w") as f:
                json.dump(results, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save incremental results: {e}")

    def save_results(self, output_path: Path):
        """Save simulation results to file."""
        results = {
            "agent_decisions": self.agent_decisions,
            "events": self.event_history,
            "config": {
                "decision_interval": self.decision_interval,
                "max_steps": self.max_steps,
                "num_agents": len(self.concordia_agents),
            },
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results saved to {output_path}")
