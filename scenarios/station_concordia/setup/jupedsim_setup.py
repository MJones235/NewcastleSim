"""
JuPedSim simulation setup for Station Concordia simulations.

This module is responsible for:
- Initializing JuPedSim simulation instances
- Loading station geometry from network files
- Configuring simulation parameters
"""

from pathlib import Path

from scenarios.common.logger import get_logger
from scenarios.station_concordia.core.jupedsim_integration import ConcordiaJuPedSimulation

logger = get_logger(__name__)


class JuPedSimSetup:
    """Handles JuPedSim simulation initialization."""

    @staticmethod
    def create_simulation(config: dict) -> ConcordiaJuPedSimulation:
        """
        Create and configure a JuPedSim simulation instance.

        Args:
            config: Configuration dictionary containing simulation settings

        Returns:
            Initialized ConcordiaJuPedSimulation instance with geometry loaded
        """
        sim_config = config.get("simulation", {})
        dt = sim_config.get("dt", 0.05)
        network_path = Path(sim_config.get("network_path", "scenarios/station_sim/network"))

        logger.info(f"Loading real station geometry from {network_path}...")
        jps_sim = ConcordiaJuPedSimulation(
            network_path=network_path,
            dt=dt,
            exit_radius=10.0,
        )

        logger.info("JuPedSim simulation created successfully")
        return jps_sim
