"""
Main simulation runner for JuPedSim station simulation.
Handles the simulation loop, event processing, and GUI updates.
"""

import time
import random
import json
from pathlib import Path
from typing import List, Optional

from event_system import EventManager
from visualization import LiveViewer


class SimulationRunner:
    """Manages the main simulation execution loop."""
    
    def __init__(
        self,
        sim,
        agents: List,
        event_manager: EventManager,
        max_iterations: int = 3600,
        spawn_interval: float = 2.0
    ):
        """
        Initialize simulation runner.
        
        Args:
            sim: StationSimulation instance
            agents: List of StationAgent objects
            event_manager: EventManager for handling timed events
            max_iterations: Maximum simulation iterations
            spawn_interval: Time between agent spawns in seconds
        """
        self.sim = sim
        self.agents = agents
        self.event_manager = event_manager
        self.max_iterations = max_iterations
        self.spawn_interval = spawn_interval
        
        # Queue agents for spawning - randomize order so entrances are mixed
        self.agents_to_spawn = list(agents)
        random.shuffle(self.agents_to_spawn)
        print(f"Agent spawn order randomized")
        
        self.last_spawn_time = -spawn_interval  # Allow first spawn immediately
        self.last_event_message = None
        self.last_event_time = -100.0
        
    def run(
        self,
        enable_gui: bool = False,
        gui_update_interval: float = 1.0,
        viewer: Optional[LiveViewer] = None
    ) -> dict:
        """
        Execute the main simulation loop.
        
        Args:
            enable_gui: Whether GUI is enabled
            gui_update_interval: GUI update frequency in seconds
            viewer: Optional LiveViewer instance
            
        Returns:
            Dictionary with simulation statistics
        """
        print("\n[5/5] Running simulation...")
        print("Press Ctrl+C to stop\n")
        
        # Start timer for real execution time
        start_time = time.time()
        last_gui_update = 0.0
        
        try:
            while self.sim.iteration < self.max_iterations:
                # Spawn one agent if interval has passed and any are waiting
                if self.agents_to_spawn and (self.sim.get_simulation_time() - self.last_spawn_time >= self.spawn_interval):
                    agent = self.agents_to_spawn.pop(0)
                    self.last_spawn_time = self.sim.get_simulation_time()
                    try:
                        agent.spawn()
                    except Exception as e:
                        print(f"Failed to spawn {agent.id}: {e}")
                
                # Step simulation (even if no agents yet)
                if not self.sim.step():
                    # Simulation ended - check if we're done
                    if not self.agents_to_spawn and self.sim.simulation.agent_count() == 0:
                        break  # All agents spawned and completed
                
                sim_time = self.sim.get_simulation_time()
                agent_count = self.sim.simulation.agent_count()
                
                # Check and trigger events
                triggered_events = self.event_manager.check_and_trigger_events(sim_time, self.agents)
                
                # Store latest event message for GUI display
                if triggered_events:
                    self.last_event_message = triggered_events[-1].value
                    self.last_event_time = sim_time
                
                # Update all spawned agents
                for agent in self.agents:
                    if agent.is_spawned:
                        agent.update(sim_time)
                
                # Print progress every 100 steps (5 seconds)
                if self.sim.iteration % 100 == 0:
                    spawned_count = sum(1 for a in self.agents if a.is_spawned)
                    print(f"t={sim_time:6.2f}s  agents={agent_count:3d}  spawned={spawned_count:3d}/{len(self.agents)}")
                
                # Update GUI at specified interval
                if viewer and (sim_time - last_gui_update) >= gui_update_interval:
                    # Get current agent positions directly from JuPedSim
                    agent_positions = []
                    for agent in self.sim.simulation.agents():
                        pos = agent.position
                        agent_positions.append((pos[0], pos[1]))
                    
                    # Pass event message if it was recently triggered (within 5 seconds)
                    event_message = None
                    if self.last_event_message and (sim_time - self.last_event_time) < 5.0:
                        event_message = self.last_event_message
                    
                    viewer.update(agent_positions, sim_time, agent_count, event_message)
                    last_gui_update = sim_time
                    
        except KeyboardInterrupt:
            print("\n\nSimulation interrupted by user")
        finally:
            # Close GUI if open
            if viewer:
                viewer.close()
        
        # Calculate statistics
        end_time = time.time()
        real_time_elapsed = end_time - start_time
        simulated_time = self.sim.get_simulation_time()
        
        return {
            'iterations': self.sim.iteration,
            'simulated_time': simulated_time,
            'real_time': real_time_elapsed,
            'remaining_agents': self.sim.simulation.agent_count(),
            'total_agents': len(self.agents)
        }
    
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
            print(f"\nSaved {len(events_data)} triggered events to {events_output}")
        else:
            # No events triggered - clear any old events file
            if events_output.exists():
                events_output.unlink()
                print(f"\nNo events triggered - cleared old events file")
