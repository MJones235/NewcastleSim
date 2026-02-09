"""
Configuration loading and validation for Station Concordia simulations.

This module is responsible for:
- Loading configuration from YAML files
- Applying command-line overrides
- Validating configuration structure
"""

from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from scenarios.common.logger import get_logger

logger = get_logger(__name__)


class ConfigLoader:
    """Handles loading and validation of simulation configuration."""

    @staticmethod
    def load_and_validate(
        config_path: str,
        agents: Optional[int] = None,
        max_steps: Optional[int] = None,
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Load, validate, and apply overrides to configuration in one step.

        This is the primary entry point for configuration loading.

        Args:
            config_path: Path to the YAML configuration file
            agents: Number of agents (overrides config if provided)
            max_steps: Maximum simulation steps (overrides config if provided)
            output_dir: Output directory (overrides config if provided)

        Returns:
            Dictionary containing the loaded and validated configuration

        Raises:
            FileNotFoundError: If the configuration file doesn't exist
            yaml.YAMLError: If the YAML file is malformed
            ValueError: If required configuration is missing
        """
        # Load base configuration
        config = ConfigLoader.load_config(config_path)

        # Apply CLI overrides
        config = ConfigLoader.apply_cli_overrides(config, agents, max_steps, output_dir)

        # Validate
        ConfigLoader.validate_config(config)

        return config

    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to the YAML configuration file

        Returns:
            Dictionary containing the loaded configuration

        Raises:
            FileNotFoundError: If the configuration file doesn't exist
            yaml.YAMLError: If the YAML file is malformed
        """
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_file) as f:
            config = yaml.safe_load(f)

        logger.info(f"Loaded configuration from {config_path}")
        return config

    @staticmethod
    def apply_cli_overrides(
        config: Dict[str, Any],
        agents: Optional[int] = None,
        max_steps: Optional[int] = None,
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Apply command-line argument overrides to configuration.

        Args:
            config: Base configuration dictionary
            agents: Number of agents (overrides config if provided)
            max_steps: Maximum simulation steps (overrides config if provided)
            output_dir: Output directory (overrides config if provided)

        Returns:
            Modified configuration dictionary
        """
        if agents is not None:
            config.setdefault("agents", {})["count"] = agents
            logger.info(f"Override: agents count = {agents}")

        if max_steps is not None:
            config.setdefault("simulation", {})["max_iterations"] = max_steps
            logger.info(f"Override: max_iterations = {max_steps}")

        if output_dir is not None:
            config.setdefault("output", {})["directory"] = output_dir
            logger.info(f"Override: output directory = {output_dir}")

        return config

    @staticmethod
    def validate_config(config: Dict[str, Any]) -> None:
        """
        Validate that required configuration sections exist.

        Args:
            config: Configuration dictionary to validate

        Raises:
            ValueError: If required configuration is missing
        """
        required_sections = ["agents", "simulation"]
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Required configuration section '{section}' is missing")

        # Validate agents section
        agents_config = config["agents"]
        if "count" not in agents_config:
            raise ValueError("agents.count is required in configuration")

        # Validate simulation section
        sim_config = config["simulation"]
        if "network_path" not in sim_config:
            raise ValueError("simulation.network_path is required in configuration")

        logger.debug("Configuration validation passed")
