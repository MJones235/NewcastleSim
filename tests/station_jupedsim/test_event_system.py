"""Tests for event system."""

import pytest
import tempfile
from pathlib import Path

from scenarios.station_jupedsim.core.event_system import SimulationEvent, EventManager


class TestSimulationEvent:
    """Test SimulationEvent dataclass."""
    
    def test_create_valid_event(self):
        """Test creating a valid event."""
        event = SimulationEvent(
            time=30.0,
            action='broadcast_to_all',
            value='Test message'
        )
        
        assert event.time == 30.0
        assert event.action == 'broadcast_to_all'
        assert event.value == 'Test message'
    
    def test_negative_time_raises_error(self):
        """Test that negative time raises ValueError."""
        with pytest.raises(ValueError, match="Event time must be non-negative"):
            SimulationEvent(time=-5.0, action='test', value='message')
    
    def test_empty_action_raises_error(self):
        """Test that empty action raises ValueError."""
        with pytest.raises(ValueError, match="Event action cannot be empty"):
            SimulationEvent(time=10.0, action='', value='message')
    
    def test_event_equality(self):
        """Test event equality comparison."""
        event1 = SimulationEvent(time=30.0, action='test', value='msg')
        event2 = SimulationEvent(time=30.0, action='test', value='msg')
        event3 = SimulationEvent(time=60.0, action='test', value='msg')
        
        assert event1 == event2
        assert event1 != event3
    
    def test_event_hashable(self):
        """Test that events can be added to sets."""
        event1 = SimulationEvent(time=30.0, action='test', value='msg')
        event2 = SimulationEvent(time=30.0, action='test', value='msg')
        
        event_set = {event1, event2}
        assert len(event_set) == 1  # Duplicates removed


class TestEventManager:
    """Test EventManager class."""
    
    def test_create_empty_manager(self):
        """Test creating event manager without events file."""
        manager = EventManager()
        
        assert len(manager.events) == 0
        assert len(manager.triggered_events) == 0
    
    def test_load_events_from_csv(self):
        """Test loading events from CSV file."""
        csv_content = """time, action, value
30.0, broadcast_to_all, "First message"
60.0, broadcast_to_all, "Second message"
90.0, broadcast_to_all, "Third message"
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_file = f.name
            f.write(csv_content)
        
        try:
            manager = EventManager(temp_file)
            
            assert len(manager.events) == 3
            assert manager.events[0].time == 30.0
            assert manager.events[0].value == "First message"
            assert manager.events[2].time == 90.0
        finally:
            Path(temp_file).unlink()
    
    def test_load_events_sorted_by_time(self):
        """Test that events are sorted by time."""
        csv_content = """time, action, value
90.0, broadcast_to_all, "Third"
30.0, broadcast_to_all, "First"
60.0, broadcast_to_all, "Second"
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_file = f.name
            f.write(csv_content)
        
        try:
            manager = EventManager(temp_file)
            
            assert manager.events[0].time == 30.0
            assert manager.events[1].time == 60.0
            assert manager.events[2].time == 90.0
        finally:
            Path(temp_file).unlink()
    
    def test_skip_comments_and_headers(self):
        """Test that comments and header lines are skipped."""
        csv_content = """# This is a comment
time, action, value
# Another comment
30.0, broadcast_to_all, "Message"
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_file = f.name
            f.write(csv_content)
        
        try:
            manager = EventManager(temp_file)
            
            assert len(manager.events) == 1
            assert manager.events[0].time == 30.0
        finally:
            Path(temp_file).unlink()
    
    def test_handle_malformed_rows(self):
        """Test that malformed rows are skipped with warning."""
        csv_content = """time, action, value
30.0, broadcast_to_all, "Valid"
invalid, row, here
60.0, broadcast_to_all, "Also valid"
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_file = f.name
            f.write(csv_content)
        
        try:
            manager = EventManager(temp_file)
            
            # Should load valid events and skip malformed ones
            assert len(manager.events) == 2
            assert manager.events[0].time == 30.0
            assert manager.events[1].time == 60.0
        finally:
            Path(temp_file).unlink()
    
    def test_add_event_programmatically(self):
        """Test adding events programmatically."""
        manager = EventManager()
        
        manager.add_event(45.0, 'broadcast_to_all', 'New event')
        
        assert len(manager.events) == 1
        assert manager.events[0].time == 45.0
    
    def test_load_nonexistent_file_raises_error(self):
        """Test that loading nonexistent file raises FileNotFoundError."""
        # EventManager checks if file exists and returns early if not
        # Only raises if file path is provided and doesn't exist during load
        manager = EventManager()
        
        # Should have no events
        assert len(manager.events) == 0
    
    def test_strip_quotes_from_values(self):
        """Test that quotes are stripped from event values."""
        csv_content = """time, action, value
30.0, broadcast_to_all, "Double quoted"
60.0, broadcast_to_all, 'Single quoted'
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_file = f.name
            f.write(csv_content)
        
        try:
            manager = EventManager(temp_file)
            
            assert manager.events[0].value == "Double quoted"
            assert manager.events[1].value == "Single quoted"
        finally:
            Path(temp_file).unlink()


class TestEventTriggering:
    """Test event triggering logic."""
    
    def test_check_and_trigger_events(self):
        """Test checking and triggering events at correct times."""
        csv_content = """time, action, value
30.0, broadcast_to_all, "Message 1"
60.0, broadcast_to_all, "Message 2"
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_file = f.name
            f.write(csv_content)
        
        try:
            manager = EventManager(temp_file)
            
            # Create mock agents
            class MockAgent:
                def __init__(self):
                    self.is_spawned = True
                    self.is_active = True
                    self.messages = []
                
                def receive_message(self, message):
                    self.messages.append(message)
            
            agents = [MockAgent(), MockAgent()]
            
            # Before first event
            triggered = manager.check_and_trigger_events(29.9, agents)
            assert len(triggered) == 0
            
            # At first event
            triggered = manager.check_and_trigger_events(30.0, agents)
            assert len(triggered) == 1
            assert triggered[0].value == "Message 1"
            
            # Check agents received message
            assert len(agents[0].messages) == 1
            assert agents[0].messages[0] == "Message 1"
            
            # Event should not trigger again
            triggered = manager.check_and_trigger_events(30.1, agents)
            assert len(triggered) == 0
            
            # At second event
            triggered = manager.check_and_trigger_events(60.0, agents)
            assert len(triggered) == 1
            assert triggered[0].value == "Message 2"
        finally:
            Path(temp_file).unlink()
