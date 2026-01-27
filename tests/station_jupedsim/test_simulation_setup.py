"""Tests for simulation setup functions."""

import pytest

from scenarios.station_jupedsim.simulation import StationSimulation
from scenarios.station_jupedsim.simulation_setup import (
    setup_evacuation_exits,
    setup_platform_stages,
    load_geometry
)


class TestLoadGeometry:
    """Test geometry loading."""
    
    def test_load_geometry_valid_path(self):
        """Test loading geometry from valid network path."""
        from pathlib import Path
        network_path = Path("scenarios/station_sim/network")
        
        entrance_areas, platform_areas = load_geometry(network_path)
        
        assert len(entrance_areas) > 0
        assert len(platform_areas) > 0
        assert all(isinstance(name, str) for name in entrance_areas.keys())
        assert all(isinstance(name, str) for name in platform_areas.keys())
    
    def test_load_geometry_missing_file_raises_error(self):
        """Test that missing geometry file raises FileNotFoundError."""
        from pathlib import Path
        import tempfile
        
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(FileNotFoundError, match="Geometry file not found"):
                load_geometry(Path(temp_dir))


class TestSetupEvacuationExits:
    """Test evacuation exit setup."""
    
    def test_setup_exits_creates_exits_and_journeys(self):
        """Test that exits and journeys are created for all entrances."""
        sim = StationSimulation("scenarios/station_sim/network", dt=0.05)
        
        from pathlib import Path
        from scenarios.station_jupedsim.geometry import load_entrance_areas
        
        network_path = Path("scenarios/station_sim/network")
        walking_areas_file = network_path / "walking_areas.add.xml"
        entrance_areas = load_entrance_areas(str(walking_areas_file))
        
        exits, journeys = setup_evacuation_exits(sim, entrance_areas)
        
        assert len(exits) > 0
        assert len(journeys) > 0
        assert len(exits) == len(journeys)
        assert len(exits) == len(entrance_areas)
    
    def test_setup_exits_custom_radius(self):
        """Test creating exits with custom radius."""
        sim = StationSimulation("scenarios/station_sim/network", dt=0.05)
        
        from pathlib import Path
        from scenarios.station_jupedsim.geometry import load_entrance_areas
        
        network_path = Path("scenarios/station_sim/network")
        walking_areas_file = network_path / "walking_areas.add.xml"
        entrance_areas = load_entrance_areas(str(walking_areas_file))
        
        # Should not raise error with different radius
        exits, journeys = setup_evacuation_exits(sim, entrance_areas, exit_radius=5.0)
        
        assert len(exits) > 0
    
    def test_setup_exits_empty_entrances_raises_error(self):
        """Test that empty entrance areas raises ValueError."""
        sim = StationSimulation("scenarios/station_sim/network", dt=0.05)
        
        with pytest.raises(ValueError, match="No entrance areas provided"):
            setup_evacuation_exits(sim, {})


class TestSetupPlatformStages:
    """Test platform stage setup."""
    
    def test_setup_stages_creates_stages_and_journeys(self):
        """Test that stages and journeys are created for platforms."""
        sim = StationSimulation("scenarios/station_sim/network", dt=0.05)
        
        from pathlib import Path
        from scenarios.station_jupedsim.geometry import load_platform_areas
        
        network_path = Path("scenarios/station_sim/network")
        walking_areas_file = network_path / "walking_areas.add.xml"
        platform_areas = load_platform_areas(str(walking_areas_file))
        
        stages, journeys = setup_platform_stages(sim, platform_areas)
        
        assert len(stages) > 0
        assert len(journeys) > 0
        assert len(stages) == len(journeys)
        # Note: Some platforms might be skipped if outside walkable area
        assert len(stages) <= len(platform_areas)
    
    def test_setup_stages_empty_platforms_raises_error(self):
        """Test that empty platform areas raises ValueError."""
        sim = StationSimulation("scenarios/station_sim/network", dt=0.05)
        
        with pytest.raises(ValueError, match="No platform areas provided"):
            setup_platform_stages(sim, {})


class TestIntegration:
    """Integration tests for setup functions."""
    
    def test_full_setup_workflow(self):
        """Test complete setup workflow."""
        from pathlib import Path
        
        # Initialize simulation
        sim = StationSimulation("scenarios/station_sim/network", dt=0.05)
        
        # Load geometry
        network_path = Path("scenarios/station_sim/network")
        entrance_areas, platform_areas = load_geometry(network_path)
        
        # Setup exits
        exits, exit_journeys = setup_evacuation_exits(sim, entrance_areas)
        
        # Setup platforms
        stages, stage_journeys = setup_platform_stages(sim, platform_areas)
        
        # Verify everything is set up
        assert len(exits) > 0
        assert len(stages) > 0
        assert sim.simulation.agent_count() == 0  # No agents yet
