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

import time
from pathlib import Path
from typing import Any

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
from scenarios.station_concordia.behaviors import MessageSystem
from scenarios.station_concordia.core.action_executor import ActionExecutor
from scenarios.station_concordia.core.agent_builder import AgentBuilder
from scenarios.station_concordia.core.decision_processor import DecisionProcessor
from scenarios.station_concordia.core.event_manager import EventManager
from scenarios.station_concordia.core.exit_tracker import ExitTracker
from scenarios.station_concordia.core.helping_system import HelpingSystemManager
from scenarios.station_concordia.core.observation_coordinator import ObservationCoordinator
from scenarios.station_concordia.core.performance_monitor import PerformanceTimer
from scenarios.station_concordia.core.simulation_state_queries import SimulationStateQueries
from scenarios.station_concordia.reporting.financial_reporter import FinancialReporter
from scenarios.station_concordia.reporting.results_writer import ResultsWriter
from scenarios.station_concordia.translation import ActionTranslator, ObservationGenerator

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
        test_scenarios: dict[str, Any] | None = None,
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
        self.test_scenarios = test_scenarios or {}

        # Simulation state queries
        self.state_queries = SimulationStateQueries(jupedsim_simulation)

        # Store LLM provider reference (for usage stats)
        # The language_model is an AzureLLMConcordia instance directly
        self.llm_provider = language_model if hasattr(language_model, "get_usage_stats") else None

        # Translation layer components
        self.action_translator = ActionTranslator(station_layout, language_model)
        self.observation_generator = ObservationGenerator(station_layout)

        # Build Concordia agents (each with their own memory bank)
        self.concordia_agents: dict[str, entity_lib.Entity] = {}
        self.agent_configs = agents_config

        # Help behavior tracking (must be initialized before _build_agents)
        self.agent_status: dict[str, str] = (
            {}
        )  # agent_id -> "EVACUATING"|"HELPING"|"WAITING"|"INJURED"
        self.agents_being_helped: dict[str, str] = {}  # helped_agent_id -> helper_agent_id
        self.help_events: list[dict[str, Any]] = []  # Track all help interactions
        self.active_helping_pairs: dict[str, dict[str, Any]] = (
            {}
        )  # helper_id -> {helped, start_time, duration}
        self.agent_original_speeds: dict[str, float] = (
            {}
        )  # Store original walking speeds for restoration

        # Build agents using AgentBuilder
        agent_builder = AgentBuilder(
            language_model=language_model,
            embedder=embedder,
            station_layout_description=self.observation_generator._describe_geometry(),
        )
        self.concordia_agents, self.agent_status = agent_builder.build_agents(agents_config)

        # Tracking
        self.last_decision_time = (
            -decision_interval
        )  # Start negative so first decision happens immediately
        self.current_sim_time = 0.0
        self.current_step = 0  # Track current simulation step for logging
        self.agent_decisions: dict[str, dict[str, Any]] = {}
        self.last_observations: dict[str, str] = {}  # Cache observations for change detection
        self.last_actions: dict[str, str] = {}  # Cache actions to reuse

        # Route changing tracking
        self.agent_destinations: dict[str, str] = {}  # agent_id -> current exit name

        # Track exited agents (those who have evacuated)
        self.exited_agents: set[str] = set()  # agent_ids who have reached exits

        # Event management
        self.event_manager = EventManager(station_layout, jupedsim_simulation)
        self.event_manager.setup_test_scenario(test_scenarios)

        # Helping system management
        self.helping_system_manager = HelpingSystemManager(
            active_helping_pairs=self.active_helping_pairs,
            agents_being_helped=self.agents_being_helped,
            agent_original_speeds=self.agent_original_speeds,
            agent_status=self.agent_status,
            agent_destinations=self.agent_destinations,
            exited_agents=self.exited_agents,
            test_scenarios=test_scenarios or {},
            jps_sim=jupedsim_simulation,
            state_queries=self.state_queries,
        )

        # Exit tracking
        self.exit_tracker = ExitTracker(
            concordia_agents=self.concordia_agents,
            exited_agents=self.exited_agents,
            agent_destinations=self.agent_destinations,
            jps_sim=jupedsim_simulation,
        )

        # Waiting and information seeking tracking
        self.wait_events: list[dict[str, Any]] = []  # Track all wait decisions with reasons

        # Agent-to-agent messaging
        self.message_system = MessageSystem(
            default_radius=10.0,
            memory_window=60.0,
        )
        # Performance profiling (must be initialized before decision_processor)
        self.perf_timer = PerformanceTimer()
        # Action execution
        self.action_executor = ActionExecutor(
            jps_sim=jupedsim_simulation,
            state_queries=self.state_queries,
            event_manager=self.event_manager,
            station_layout=station_layout,
            agent_status=self.agent_status,
            agents_being_helped=self.agents_being_helped,
            agent_destinations=self.agent_destinations,
            active_helping_pairs=self.active_helping_pairs,
            agent_original_speeds=self.agent_original_speeds,
            help_events=self.help_events,
            wait_events=self.wait_events,
            agent_configs=agents_config,
            test_scenarios=test_scenarios or {},
        )

        # Decision processing
        self.decision_processor = DecisionProcessor(
            concordia_agents=self.concordia_agents,
            exited_agents=self.exited_agents,
            action_translator=self.action_translator,
            action_executor=self.action_executor,
            message_system=self.message_system,
            state_queries=self.state_queries,
            station_layout=station_layout,
            agent_decisions=self.agent_decisions,
            agent_destinations=self.agent_destinations,
            last_observations=self.last_observations,
            last_actions=self.last_actions,
            perf_timer=self.perf_timer,
            helping_system_manager=self.helping_system_manager,
        )

        # Observation coordination
        self.observation_coordinator = ObservationCoordinator(
            concordia_agents=self.concordia_agents,
            exited_agents=self.exited_agents,
            observation_generator=self.observation_generator,
            state_queries=self.state_queries,
            event_manager=self.event_manager,
            message_system=self.message_system,
            agent_destinations=self.agent_destinations,
            agent_status=self.agent_status,
            test_scenarios=test_scenarios or {},
        )

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
                    step_start = time.perf_counter()
                    self.current_step = step

                    # Advance JuPedSim simulation
                    with self.perf_timer.measure("jupedsim_step"):
                        if not self._step_jupedsim():
                            logger.info("JuPedSim simulation complete")
                            break

                    self.current_sim_time = step * self.jps_sim.dt

                    # Check for agents who have exited and remove them
                    self.exit_tracker.check_exited_agents(self.current_sim_time, self.current_step)

                    # Update helping relationships (check for expired help durations)
                    self.helping_system_manager.update_helping_relationships(self.current_sim_time)

                    # Check if it's time for Concordia decisions
                    if self._should_make_decisions():
                        with self.perf_timer.measure("agent_decisions_total"):
                            # Generate observations for all agents
                            observations = self.observation_coordinator.generate_all_observations(
                                self.current_sim_time
                            )
                            # Process all agent decisions in parallel
                            self.last_decision_time = self.decision_processor.process_all_agents(
                                observations, self.current_sim_time
                            )

                    # Check for events
                    with self.perf_timer.measure("event_checking"):
                        self.event_manager.check_and_trigger_events(self.current_sim_time)

                    # Save positions every 10 steps (0.5s) for smooth visualization
                    # Writing every single step (0.05s) is too slow for file I/O
                    if self.output_file and step % 10 == 0:
                        with self.perf_timer.measure("file_io"):
                            ResultsWriter.save_incremental(
                                self.output_file,
                                self.agent_decisions,
                                self.jps_sim.get_all_agent_positions(),
                                self.current_sim_time,
                                self.event_manager.event_history,
                                self.event_manager.blocked_exits,
                                self.active_helping_pairs,
                                self.message_system.message_history,
                                self.decision_interval,
                                self.max_steps,
                                len(self.concordia_agents),
                            )

                    results["steps"] = step + 1
                    results["sim_time"] = self.current_sim_time

                    # Update progress bar
                    progress.update(task, advance=1)

                    # Pace simulation to real time for smooth visualization
                    if self.jps_sim.dt > 0:
                        elapsed = time.perf_counter() - step_start
                        sleep_time = self.jps_sim.dt - elapsed
                        if sleep_time > 0:
                            time.sleep(sleep_time)

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
        results["events_triggered"] = len(self.event_manager.event_history)

        logger.info(
            f"Simulation complete: {results['steps']} steps, "
            f"{results['sim_time']:.1f}s sim time, "
            f"{elapsed_time:.1f}s real time"
        )

        # Print performance profile
        print(self.perf_timer.report())

        # Print financial report
        print(FinancialReporter.generate_report(self.llm_provider, len(self.concordia_agents)))

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
