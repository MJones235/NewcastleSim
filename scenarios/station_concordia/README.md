# Station Concordia - Evacuation Simulation with Concordia

An experimental scenario that integrates Google DeepMind's Concordia library for agent decision-making in station evacuation simulations, using JuPedSim for pedestrian movement.

## Overview

This scenario explores using Concordia's rich cognitive architecture for modeling evacuation decision-making, while maintaining JuPedSim for realistic pedestrian dynamics.

### Architecture

```
Concordia Layer (Decision Making)
    ↓ Natural Language Actions
Translation Layer
    ↓ Waypoints & Goals
JuPedSim Layer (Movement Simulation)
```

## Key Features

- **Rich Agent Cognition**: Agents use Concordia's component system
  - Associative memory for past experiences
  - Self-perception (personality-driven behavior)
  - Situation perception (understanding evacuation context)
  - Planning capabilities for multi-step actions

- **Event-Driven Decision Making**: Agents only query LLM on significant events
  - Emergency announcements
  - Observing crowd behavior
  - Encountering obstacles
  - Receiving messages from other agents

- **Hybrid Time Stepping**:
  - Concordia: Coarse time scale (5-10 second intervals)
  - JuPedSim: Continuous simulation (0.05 second timesteps)

## Installation

```bash
# Install Concordia
pip install gdm-concordia

# Install sentence transformer for embeddings
pip install sentence-transformers
```

## Usage

```python
from scenarios.station_concordia import run_station_concordia

# Run simulation
results = run_station_concordia.run_simulation(
    config_path="scenarios/station_concordia/config/config.yaml",
    max_steps=1000,
    agents=50
)
```

## Configuration

See [config/config.yaml](config/config.yaml) for configuration options.

## Architecture Details

### Concordia Components

1. **ConcordiaEvacuationAgent** - Custom agent prefab with:
   - Evacuation-specific memory components
   - Risk perception reasoning
   - Social influence observation
   - Exit knowledge mental map

2. **StationGameMaster** - Custom GM that:
   - Interfaces with JuPedSim state
   - Generates observations from simulation
   - Translates actions to waypoints
   - Manages event broadcasting

3. **ActionTranslator** - Parses natural language actions:
   - "I'll evacuate via north exit" → waypoint coordinates
   - "I'll help the person nearby" → approach target
   - "I'll wait for more information" → stay in place

### Performance Optimization

- Event-driven LLM queries (not every timestep)
- Batch processing of agent decisions
- Cooldown periods between queries
- Caching of similar decision patterns

## Development Status

🚧 **Experimental** - This is a research prototype exploring Concordia integration.

### Current Status

- ✅ Feasibility assessment complete
- ✅ Basic project structure
- 🚧 ConcordiaEvacuationAgent prefab
- 🚧 StationGameMaster implementation
- ⏳ Action translation layer
- ⏳ Observation generation
- ⏳ Performance benchmarks

## References

- [Concordia Documentation](https://github.com/google-deepmind/concordia)
- [Concordia Paper](https://arxiv.org/abs/2312.03664)
- [JuPedSim Documentation](https://www.jupedsim.org/)
- [Feasibility Assessment](FEASIBILITY_ASSESSMENT.md)

## License

See LICENSE file in repository root.
