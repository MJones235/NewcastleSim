"""
Event injection system for JuPedSim station simulation.

Allows events to be triggered at specific simulation times to influence agent behavior.
Events are loaded from a CSV file with format: time, action, value

Supported Event Actions:
    - broadcast_to_all: Send a message to all active agents (e.g., evacuation order)
    
Example CSV Format:
    # time,action,value
    30.0,broadcast_to_all,Evacuate the station immediately
"""

import csv
from pathlib import Path
from typing import List, Set, Any, Optional
from dataclasses import dataclass


@dataclass
class SimulationEvent:
    """Represents a single event to be injected during simulation."""
    time: float
    action: str
    value: str
    
    def __post_init__(self):
        """Validate event data after initialization."""
        if self.time < 0:
            raise ValueError(f"Event time must be non-negative, got {self.time}")
        if not self.action:
            raise ValueError("Event action cannot be empty")
        if self.action not in ['broadcast_to_all']:
            print(f"WARNING: Unknown event action '{self.action}' (may not be implemented)")
    
    def __hash__(self):
        return hash((self.time, self.action, self.value))
    
    def __eq__(self, other):
        if not isinstance(other, SimulationEvent):
            return False
        return (self.time == other.time and 
                self.action == other.action and 
                self.value == other.value)


class EventManager:
    """
    Manages simulation events and triggers them at appropriate times.
    
    Supported actions:
    - broadcast_to_all: Broadcast a message to all agents
    - (future: staff_direction, zone_closure, etc.)
    """
    
    def __init__(self, events_file: Optional[str] = None) -> None:
        """
        Initialize event manager.
        
        Args:
            events_file: Path to CSV file containing events (optional)
        """
        self.events: List[SimulationEvent] = []
        self.triggered_events: Set[SimulationEvent] = set()
        
        if events_file and Path(events_file).exists():
            self.load_events(events_file)
    
    def load_events(self, events_file: str) -> None:
        """
        Load events from CSV file.
        
        CSV format: time, action, value
        Example: 30.0, broadcast_to_all, "Evacuate the station immediately"
        
        Args:
            events_file: Path to CSV file
            
        Raises:
            FileNotFoundError: If events file doesn't exist
            ValueError: If CSV format is invalid
        """
        events_path = Path(events_file)
        
        if not events_path.exists():
            raise FileNotFoundError(f"Events file not found: {events_file}")
        
        try:
            with open(events_file, 'r') as f:
                reader = csv.reader(f)
                
                # Skip comments and header
                for row in reader:
                    if not row or row[0].strip().startswith('#'):
                        continue  # Skip comment lines
                    if row[0].strip().lower() == 'time':
                        continue  # Skip header line
                    self._parse_event_row(row)
        except Exception as e:
            raise RuntimeError(f"Failed to read events file {events_file}: {e}")
        
        # Sort events by time
        self.events.sort(key=lambda e: e.time)
        print(f"Loaded {len(self.events)} events from {events_file}")
    
    def _parse_event_row(self, row: List[str]) -> None:
        """
        Parse a single event row from CSV and add to events list.
        
        Handles malformed rows gracefully by logging warnings.
        Strips quotes from values and validates event data.
        
        Args:
            row: List of strings from CSV reader (time, action, value)
        """
        if len(row) < 3:
            print(f"WARNING: Skipping malformed event row (need 3 columns): {row}")
            return
        
        try:
            time = float(row[0].strip())
            action = row[1].strip()
            value = row[2].strip().strip('"\'')
            
            if not action:
                print(f"WARNING: Skipping event with empty action at time {time}")
                return
            
            event = SimulationEvent(time=time, action=action, value=value)
            self.events.append(event)
        except ValueError as e:
            print(f"WARNING: Could not parse event row {row}: {e}")
        except Exception as e:
            print(f"WARNING: Unexpected error parsing event row {row}: {e}")
    
    def add_event(self, time: float, action: str, value: str) -> None:
        """
        Programmatically add an event.
        
        Args:
            time: Simulation time when event should trigger
            action: Type of event (e.g., 'broadcast_to_all')
            value: Event-specific data
        """
        event = SimulationEvent(time=time, action=action, value=value)
        self.events.append(event)
        self.events.sort(key=lambda e: e.time)
    
    def check_and_trigger_events(self, sim_time: float, agents: List[Any]) -> List[SimulationEvent]:
        """
        Check if any events should trigger at current simulation time and execute them.
        
        Args:
            sim_time: Current simulation time
            agents: List of StationAgent objects
            
        Returns:
            List of events that were triggered
        """
        triggered = []
        
        for event in self.events:
            # Trigger events that haven't been triggered yet and whose time has come
            if event not in self.triggered_events and event.time <= sim_time:
                self._trigger_event(event, agents, sim_time)
                self.triggered_events.add(event)
                triggered.append(event)
        
        return triggered
    
    def _trigger_event(self, event: SimulationEvent, agents: List[Any], sim_time: float) -> None:
        """
        Execute a specific event.
        
        Args:
            event: The event to trigger
            agents: List of StationAgent objects
            sim_time: Current simulation time
        """
        if event.action == 'broadcast_to_all':
            self._broadcast_to_all(event.value, agents, sim_time)
        else:
            print(f"Warning: Unknown event action '{event.action}' at time {event.time}")
    
    def _broadcast_to_all(self, message: str, agents: List[Any], sim_time: float) -> None:
        """
        Broadcast a message to all agents.
        
        Args:
            message: Message to broadcast
            agents: List of StationAgent objects
            sim_time: Current simulation time
        """
        print(f"\n[EVENT @ t={sim_time:.1f}s] Broadcasting to all agents: '{message}'")
        
        delivered_count = 0
        for agent in agents:
            if agent.is_spawned and agent.is_active:
                agent.receive_message(message)
                delivered_count += 1
        
        print(f"  Message delivered to {delivered_count} active agents")
    
    def get_pending_events(self, sim_time: float) -> List[SimulationEvent]:
        """
        Get list of events that haven't triggered yet.
        
        Args:
            sim_time: Current simulation time
            
        Returns:
            List of pending events
        """
        return [e for e in self.events 
                if e not in self.triggered_events and e.time > sim_time]
    
    def get_next_event_time(self, sim_time: float) -> Optional[float]:
        """
        Get time of next pending event.
        
        Args:
            sim_time: Current simulation time
            
        Returns:
            Time of next event, or None if no pending events
        """
        pending = self.get_pending_events(sim_time)
        return pending[0].time if pending else None


if __name__ == "__main__":
    print("Event system module ready")
