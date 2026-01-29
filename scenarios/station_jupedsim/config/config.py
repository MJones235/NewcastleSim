"""
Configuration management for JuPedSim station simulation.

Provides dataclass-based configuration with defaults and YAML file loading.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv


@dataclass
class SimulationConfig:
    """Core simulation parameters."""

    # Time parameters
    dt: float = 0.05  # Simulation timestep in seconds (0.05s = 20 fps)
    max_iterations: int = 3600  # Maximum simulation steps (180s at dt=0.05)

    # Agent parameters
    num_agents: int = 60  # Total number of agents to create
    spawn_interval: float = 2.0  # Time between agent spawns in seconds (for gradual spawning)
    spawn_mode: str = (
        "random"  # "entrances" for gradual spawning, "random" for immediate random placement
    )

    # Exit parameters
    exit_radius: float = 10.0  # Radius of circular evacuation exits in meters

    # Trajectory recording
    trajectory_frame_interval: int = 4  # Save every Nth frame (4 = 0.2s intervals)

    def validate(self) -> None:
        """Validate configuration values.

        Raises:
            ValueError: If any configuration value is invalid
        """
        if self.dt <= 0:
            raise ValueError(f"dt must be positive, got {self.dt}")
        if self.max_iterations <= 0:
            raise ValueError(f"max_iterations must be positive, got {self.max_iterations}")
        if self.num_agents <= 0:
            raise ValueError(f"num_agents must be positive, got {self.num_agents}")
        if self.spawn_interval <= 0:
            raise ValueError(f"spawn_interval must be positive, got {self.spawn_interval}")
        if self.spawn_mode not in ["entrances", "random"]:
            raise ValueError(f"spawn_mode must be 'entrances' or 'random', got {self.spawn_mode}")
        if self.exit_radius <= 0:
            raise ValueError(f"exit_radius must be positive, got {self.exit_radius}")
        if self.trajectory_frame_interval <= 0:
            raise ValueError(
                f"trajectory_frame_interval must be positive, got {self.trajectory_frame_interval}"
            )


@dataclass
class VisualizationConfig:
    """Visualization parameters."""

    # GUI parameters
    enable_gui: bool = False  # Enable real-time visualization
    gui_update_interval: float = 1.0  # GUI update frequency in seconds

    # Post-run visualization
    enable_post_run_viz: bool = True  # Enable post-run animation
    animation_interval: int = 50  # Milliseconds between animation frames
    event_popup_duration: float = 5.0  # Seconds to show event popups

    def validate(self) -> None:
        """Validate configuration values.

        Raises:
            ValueError: If any configuration value is invalid
        """
        if self.gui_update_interval <= 0:
            raise ValueError(
                f"gui_update_interval must be positive, got {self.gui_update_interval}"
            )
        if self.animation_interval <= 0:
            raise ValueError(f"animation_interval must be positive, got {self.animation_interval}")
        if self.event_popup_duration <= 0:
            raise ValueError(
                f"event_popup_duration must be positive, got {self.event_popup_duration}"
            )


@dataclass
class PathConfig:
    """File and directory paths."""

    network_dir: str = "scenarios/station_sim/network"
    output_dir: str = "scenarios/station_jupedsim/output"
    events_file: str | None = None  # Path to events CSV, or None for no events

    def validate(self) -> None:
        """Validate paths exist.

        Raises:
            FileNotFoundError: If required paths don't exist
        """
        network_path = Path(self.network_dir)
        if not network_path.exists():
            raise FileNotFoundError(f"Network directory not found: {self.network_dir}")

        walking_areas = network_path / "walking_areas.add.xml"
        if not walking_areas.exists():
            raise FileNotFoundError(f"Required file not found: {walking_areas}")

        if self.events_file and not Path(self.events_file).exists():
            raise FileNotFoundError(f"Events file not found: {self.events_file}")


@dataclass
class LLMConfig:
    """LLM provider configuration for agent decision-making."""

    enabled: bool = False  # Whether to use LLM for decisions
    endpoint: str | None = None  # Azure AI model endpoint URL
    api_key: str | None = None  # Azure AI API key
    model: str | None = None  # Model name (optional for serverless endpoints)

    def validate(self) -> None:
        """Validate LLM configuration.

        Raises:
            ValueError: If LLM is enabled but credentials are missing
        """
        if self.enabled:
            if not self.endpoint:
                raise ValueError("LLM enabled but endpoint not provided")
            if not self.api_key:
                raise ValueError("LLM enabled but api_key not provided")


@dataclass
class Config:
    """Complete simulation configuration."""

    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)

    def validate(self) -> None:
        """Validate all configuration sections.

        Raises:
            ValueError: If any configuration value is invalid
            FileNotFoundError: If required paths don't exist
        """
        self.simulation.validate()
        self.visualization.validate()
        self.paths.validate()
        self.llm.validate()

    @classmethod
    def from_yaml(cls, yaml_file: str) -> "Config":
        """Load configuration from YAML file.

        Args:
            yaml_file: Path to YAML configuration file

        Returns:
            Config instance with values from file

        Raises:
            FileNotFoundError: If YAML file doesn't exist
            ValueError: If YAML format is invalid
        """
        # Load environment variables from .env file
        load_dotenv()

        yaml_path = Path(yaml_file)

        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_file}")

        try:
            with open(yaml_path) as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML format in {yaml_file}: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to read configuration file: {e}")

        if not data:
            return cls()  # Empty file, use defaults

        # Build config from YAML data
        config = cls()

        if "simulation" in data:
            for key, value in data["simulation"].items():
                if hasattr(config.simulation, key):
                    setattr(config.simulation, key, value)

        if "visualization" in data:
            for key, value in data["visualization"].items():
                if hasattr(config.visualization, key):
                    setattr(config.visualization, key, value)

        if "paths" in data:
            for key, value in data["paths"].items():
                if hasattr(config.paths, key):
                    setattr(config.paths, key, value)

        if "llm" in data:
            for key, value in data["llm"].items():
                if hasattr(config.llm, key):
                    setattr(config.llm, key, value)

        # Override LLM config with environment variables if present
        if os.getenv("AZURE_LLM_ENDPOINT"):
            config.llm.endpoint = os.getenv("AZURE_LLM_ENDPOINT")
        if os.getenv("AZURE_LLM_API_KEY"):
            config.llm.api_key = os.getenv("AZURE_LLM_API_KEY")
        if os.getenv("AZURE_LLM_MODEL"):
            config.llm.model = os.getenv("AZURE_LLM_MODEL")

        # Validate after loading
        config.validate()

        return config

    def to_yaml(self, yaml_file: str) -> None:
        """Save configuration to YAML file.

        Args:
            yaml_file: Path to save YAML configuration
        """
        data = {
            "simulation": {
                "dt": self.simulation.dt,
                "max_iterations": self.simulation.max_iterations,
                "num_agents": self.simulation.num_agents,
                "spawn_interval": self.simulation.spawn_interval,
                "exit_radius": self.simulation.exit_radius,
                "trajectory_frame_interval": self.simulation.trajectory_frame_interval,
            },
            "visualization": {
                "enable_gui": self.visualization.enable_gui,
                "gui_update_interval": self.visualization.gui_update_interval,
                "animation_interval": self.visualization.animation_interval,
                "event_popup_duration": self.visualization.event_popup_duration,
            },
            "paths": {
                "network_dir": self.paths.network_dir,
                "output_dir": self.paths.output_dir,
                "events_file": self.paths.events_file,
            },
        }

        with open(yaml_file, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

        print(f"Configuration saved to {yaml_file}")


def get_default_config() -> Config:
    """Get default configuration.

    Returns:
        Config instance with default values
    """
    return Config()


def load_config(yaml_file: str | None = None) -> Config:
    """Load configuration from file or use defaults.

    Args:
        yaml_file: Optional path to YAML configuration file

    Returns:
        Config instance
    """
    if yaml_file and Path(yaml_file).exists():
        print(f"Loading configuration from {yaml_file}")
        return Config.from_yaml(yaml_file)
    else:
        print("Using default configuration")
        return Config()
