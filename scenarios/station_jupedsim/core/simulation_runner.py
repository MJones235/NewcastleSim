"""
Main simulation runner for JuPedSim station simulation.
Handles the simulation loop, event processing, and observer notifications.
Uses observer pattern to decouple GUI and other output mechanisms.
"""

import time
import random
import json
from pathlib import Path
from typing import List, Optional

from scenarios.station_jupedsim.core.event_system import EventManager
from scenarios.station_jupedsim.core.simulation_observer import SimulationObserver
from scenarios.common.logger import get_logger

logger = get_logger(__name__)


class SimulationRunner:
    """Manages the main simulation execution loop with observer pattern."""
    
    def __init__(
        self,
        sim,
        agents: List,
        event_manager: EventManager,
        max_iterations: int = 3600,
        spawn_interval: float = 2.0,
        observers: Optional[List[SimulationObserver]] = None
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
        logger.debug(f"Agent spawn order randomized")
        
        self.last_spawn_time = -spawn_interval  # Allow first spawn immediately
        self.last_event_message = None
        self.last_event_time = -100.0
        
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
                if self.agents_to_spawn and (self.sim.get_simulation_time() - self.last_spawn_time >= self.spawn_interval):
                    agent = self.agents_to_spawn.pop(0)
                    self.last_spawn_time = self.sim.get_simulation_time()
                    try:
                        agent.spawn()
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
                triggered_events = self.event_manager.check_and_trigger_events(sim_time, self.agents)
                
                # Store latest event message for observer notifications
                if triggered_events:
                    self.last_event_message = triggered_events[-1].value
                    self.last_event_time = sim_time
                
                # Update all spawned agents
                for agent in self.agents:
                    if agent.is_spawned:
                        agent.update(sim_time)
                
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
                        'event_message': event_message,
                        'spawned_count': spawned_count,
                        'total_agents': len(self.agents)
                    }
                    
                    for observer in self.observers:
                        observer.on_simulation_step(
                            sim_time, 
                            self.sim.iteration,
                            agent_count, 
                            agent_positions,
                            metadata
                        )
                    
        except KeyboardInterrupt:
            print("\n\nSimulation interrupted by user")
        
        # Calculate statistics
        end_time = time.time()
        real_time_elapsed = end_time - start_time
        simulated_time = self.sim.get_simulation_time()
        
        stats = {
            'iterations': self.sim.iteration,
            'simulated_time': simulated_time,
            'real_time': real_time_elapsed,
            'remaining_agents': self.sim.simulation.agent_count(),
            'total_agents': len(self.agents)
        }
        
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
                {
                    "time": e.time,
                    "action": e.action,
                    "value": e.value
                }
                for e in self.event_manager.triggered_events
            ]
            with open(events_output, 'w') as f:
                json.dump(events_data, f, indent=2)
            logger.info(f"Saved {len(events_data)} triggered events to {events_output}")
        else:
            # No events triggered - clear any old events file
            if events_output.exists():
                events_output.unlink()
                logger.info(f"No events triggered - cleared old events file")
