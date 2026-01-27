"""Tests for configuration system."""

import tempfile
from pathlib import Path

import pytest

from scenarios.station_jupedsim.config import (
    Config,
    PathConfig,
    SimulationConfig,
    VisualizationConfig,
    get_default_config,
    load_config,
)


class TestSimulationConfig:
    """Test simulation configuration."""

    def test_default_values(self):
        """Test default simulation configuration values."""
        config = SimulationConfig()

        assert config.dt == 0.05
        assert config.max_iterations == 3600
        assert config.num_agents == 60
        assert config.spawn_interval == 2.0
        assert config.exit_radius == 10.0
        assert config.trajectory_frame_interval == 4

    def test_validation_positive_dt(self):
        """Test that negative dt raises ValueError."""
        config = SimulationConfig(dt=-0.1)

        with pytest.raises(ValueError, match="dt must be positive"):
            config.validate()

    def test_validation_positive_num_agents(self):
        """Test that zero/negative agents raises ValueError."""
        config = SimulationConfig(num_agents=0)

        with pytest.raises(ValueError, match="num_agents must be positive"):
            config.validate()

    def test_validation_valid_config(self):
        """Test that valid config passes validation."""
        config = SimulationConfig()
        config.validate()  # Should not raise


class TestVisualizationConfig:
    """Test visualization configuration."""

    def test_default_values(self):
        """Test default visualization configuration values."""
        config = VisualizationConfig()

        assert config.enable_gui is False
        assert config.gui_update_interval == 1.0
        assert config.animation_interval == 50
        assert config.event_popup_duration == 5.0

    def test_validation_positive_intervals(self):
        """Test that negative intervals raise ValueError."""
        config = VisualizationConfig(gui_update_interval=-1.0)

        with pytest.raises(ValueError, match="gui_update_interval must be positive"):
            config.validate()


class TestPathConfig:
    """Test path configuration."""

    def test_default_values(self):
        """Test default path configuration values."""
        config = PathConfig()

        assert config.network_dir == "scenarios/station_sim/network"
        assert config.output_dir == "scenarios/station_jupedsim/output"
        assert config.events_file is None

    def test_validation_network_dir_exists(self):
        """Test that missing network directory raises error."""
        config = PathConfig(network_dir="/nonexistent/path")

        with pytest.raises(FileNotFoundError, match="Network directory not found"):
            config.validate()

    def test_validation_events_file_missing(self):
        """Test that missing events file raises error."""
        config = PathConfig(
            network_dir="scenarios/station_sim/network", events_file="/nonexistent/events.csv"
        )

        with pytest.raises(FileNotFoundError, match="Events file not found"):
            config.validate()


class TestConfig:
    """Test main configuration class."""

    def test_default_config(self):
        """Test creating default configuration."""
        config = Config()

        assert isinstance(config.simulation, SimulationConfig)
        assert isinstance(config.visualization, VisualizationConfig)
        assert isinstance(config.paths, PathConfig)

    def test_get_default_config(self):
        """Test get_default_config function."""
        config = get_default_config()

        assert isinstance(config, Config)
        assert config.simulation.dt == 0.05

    def test_yaml_save_and_load(self):
        """Test saving and loading YAML configuration."""
        config = Config()
        config.simulation.num_agents = 100
        config.visualization.enable_gui = True

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            temp_file = f.name

        try:
            # Save config
            config.to_yaml(temp_file)

            # Load config
            loaded_config = Config.from_yaml(temp_file)

            # Verify values were preserved
            assert loaded_config.simulation.num_agents == 100
            assert loaded_config.visualization.enable_gui is True
        finally:
            Path(temp_file).unlink()

    def test_yaml_load_empty_file(self):
        """Test loading empty YAML file returns defaults."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            temp_file = f.name
            f.write("")  # Empty file

        try:
            config = Config.from_yaml(temp_file)

            # Should use defaults
            assert config.simulation.dt == 0.05
            assert config.simulation.num_agents == 60
        finally:
            Path(temp_file).unlink()

    def test_yaml_load_partial_config(self):
        """Test loading partial YAML configuration."""
        yaml_content = """
simulation:
  num_agents: 50

visualization:
  enable_gui: true
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            temp_file = f.name
            f.write(yaml_content)

        try:
            config = Config.from_yaml(temp_file)

            # Modified values
            assert config.simulation.num_agents == 50
            assert config.visualization.enable_gui is True

            # Default values
            assert config.simulation.dt == 0.05
            assert config.simulation.spawn_interval == 2.0
        finally:
            Path(temp_file).unlink()

    def test_load_config_nonexistent_file(self):
        """Test load_config with nonexistent file uses defaults."""
        config = load_config("/nonexistent/config.yaml")

        # Should use defaults
        assert config.simulation.dt == 0.05
        assert config.simulation.num_agents == 60

    def test_validation_cascades(self):
        """Test that Config.validate() validates all sections."""
        config = Config()
        config.simulation.dt = -0.1  # Invalid

        with pytest.raises(ValueError, match="dt must be positive"):
            config.validate()
