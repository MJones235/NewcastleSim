"""Tests for simulation initialization and validation."""

import pytest
from pathlib import Path

from scenarios.station_jupedsim.simulation import StationSimulation


class TestSimulationInitialization:
    """Test simulation initialization."""
    
    def test_create_simulation_valid_network(self):
        """Test creating simulation with valid network path."""
        network_path = "scenarios/station_sim/network"
        
        sim = StationSimulation(network_path, dt=0.05)
        
        assert sim.dt == 0.05
        assert sim.iteration == 0
        assert len(sim.zones) > 0
        assert sim.simulation is not None
    
    def test_invalid_network_path_raises_error(self):
        """Test that invalid network path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Network path does not exist"):
            StationSimulation("/nonexistent/path")
    
    def test_negative_dt_raises_error(self):
        """Test that negative dt raises ValueError."""
        with pytest.raises(ValueError, match="Time step dt must be positive"):
            StationSimulation("scenarios/station_sim/network", dt=-0.1)
    
    def test_zero_dt_raises_error(self):
        """Test that zero dt raises ValueError."""
        with pytest.raises(ValueError, match="Time step dt must be positive"):
            StationSimulation("scenarios/station_sim/network", dt=0.0)
    
    def test_simulation_with_output_file(self):
        """Test creating simulation with trajectory output."""
        import tempfile
        
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            temp_db = f.name
        
        try:
            sim = StationSimulation(
                "scenarios/station_sim/network",
                dt=0.05,
                output_file=temp_db
            )
            
            assert sim.output_file == temp_db
        finally:
            Path(temp_db).unlink(missing_ok=True)
    
    def test_geometry_loaded(self):
        """Test that geometry is properly loaded."""
        sim = StationSimulation("scenarios/station_sim/network", dt=0.05)
        
        # Should have zones (original polygons)
        assert len(sim.zones) > 0
        
        # Should have processed zones with obstacles integrated
        assert len(sim.zones_with_obstacles) > 0
        assert len(sim.zones) == len(sim.zones_with_obstacles)
    
    def test_stage_manager_created(self):
        """Test that stage manager is initialized."""
        sim = StationSimulation("scenarios/station_sim/network", dt=0.05)
        
        assert sim.stage_manager is not None
    
    def test_missing_walking_areas_file_raises_error(self):
        """Test that missing walking_areas.add.xml raises error."""
        # Create a directory without the required file
        import tempfile
        
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(FileNotFoundError, match="Required file not found"):
                StationSimulation(temp_dir, dt=0.05)


class TestSimulationMethods:
    """Test simulation methods."""
    
    def test_step_simulation_no_agents(self):
        """Test stepping simulation with no agents returns False."""
        sim = StationSimulation("scenarios/station_sim/network", dt=0.05)
        
        # With no agents, simulation should return False
        result = sim.step()
        assert result is False
        
        # Iteration not incremented when no agents
        assert sim.iteration == 0
    
    def test_get_simulation_time(self):
        """Test getting current simulation time."""
        sim = StationSimulation("scenarios/station_sim/network", dt=0.05)
        
        # Initial time
        assert sim.get_simulation_time() == 0.0
        
        # Manually increment iteration to test
        sim.iteration = 20
        assert abs(sim.get_simulation_time() - 1.0) < 1e-10
