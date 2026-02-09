# Optimal File Structure Analysis for Station Concordia

## Current Structure Assessment

### Current Directory Layout

```
scenarios/station_concordia/
├── config/
│   ├── config_loader.py (144 lines) - Configuration management
│   └── config.yaml
├── setup/
│   ├── agent_factory.py (111 lines) - Agent configuration creation
│   ├── agent_manager.py (86 lines) - Agent lifecycle management
│   ├── jupedsim_setup.py (44 lines) - JuPedSim initialization
│   ├── llm_setup.py (84 lines) - LLM setup
│   ├── output_manager.py (58 lines) - Output directory management
│   ├── simulation_runner_factory.py (130 lines) - Runner creation
│   ├── spawn_manager.py (195 lines) - Spawn position generation
│   └── station_layout_builder.py (111 lines) - Station layout construction
├── core/
│   ├── action_utils.py (40 lines) ⚠️ NEW - Action parsing
│   ├── agent_builder.py (119 lines) ⚠️ NEW - Agent creation
│   ├── azure_llm_concordia.py (342 lines) - Azure LLM integration
│   ├── evacuation_agent.py (159 lines) - Agent prefab definition
│   ├── game_master.py (801 lines) ⚠️ LARGE - ActionTranslator + ObservationGenerator
│   ├── hybrid_simulation.py (1557 lines) ⚠️ TOO LARGE - Main orchestrator
│   ├── jupedsim_integration.py (531 lines) - JuPedSim wrapper
│   ├── performance_monitor.py (78 lines) ⚠️ NEW - Performance profiling
│   ├── simulation_state_queries.py (73 lines) ⚠️ NEW - State queries
│   └── speed_utils.py (47 lines) ⚠️ NEW - Speed conversion
```

## Issues with Current Structure

### 1. **Poor Module Cohesion**

- `game_master.py` (801 lines) contains TWO major classes:
  - `ActionTranslator` - translates actions to JuPedSim commands
  - `ObservationGenerator` - generates agent observations
  - These should be separate modules

### 2. **Inconsistent Placement**

- `agent_builder.py` is in `core/` but related `agent_factory.py` is in `setup/`
- Setup-related utilities mixed with core simulation logic
- Utility modules (`speed_utils`, `action_utils`) in `core/` but they're helpers, not core logic

### 3. **Unclear Module Boundaries**

- What belongs in `core/` vs `setup/`?
- Where should utilities go?
- No clear separation between:
  - **Domain logic** (evacuation behavior, agent cognition)
  - **Infrastructure** (JuPedSim integration, state management)
  - **Translation** (Concordia ↔ JuPedSim)
  - **Utilities** (speed conversion, action parsing)

### 4. **Still-Too-Large Files**

- `hybrid_simulation.py` (1557 lines) - needs further splitting
- `game_master.py` (801 lines) - should be split into separate concerns
- `jupedsim_integration.py` (531 lines) - acceptable but could be refined

## Proposed Optimal Structure

### Principle: **Organize by Concern, Not by Type**

```
scenarios/station_concordia/
├── config/
│   ├── config_loader.py
│   └── config.yaml
│
├── agents/                          # NEW - Agent-related modules
│   ├── __init__.py
│   ├── evacuation_agent.py          # Concordia agent prefab (MOVE from core/)
│   ├── agent_builder.py             # Agent instantiation (MOVE from core/)
│   └── agent_factory.py             # Agent config creation (MOVE from setup/)
│
├── simulation/                      # NEW - Core simulation orchestration
│   ├── __init__.py
│   ├── runner.py                    # SPLIT from hybrid_simulation.py (~300 lines)
│   ├── decision_processor.py        # EXTRACT from hybrid_simulation.py
│   ├── event_manager.py             # EXTRACT from hybrid_simulation.py
│   └── state_tracker.py             # EXTRACT from hybrid_simulation.py
│
├── behaviors/                       # NEW - Specialized agent behaviors
│   ├── __init__.py
│   ├── helping_coordinator.py       # EXTRACT from hybrid_simulation.py
│   └── message_system.py            # EXTRACT from hybrid_simulation.py
│
├── translation/                     # NEW - Concordia ↔ JuPedSim translation
│   ├── __init__.py
│   ├── action_translator.py         # SPLIT from game_master.py
│   ├── observation_generator.py     # SPLIT from game_master.py
│   └── action_executor.py           # EXTRACT from hybrid_simulation.py
│
├── infrastructure/                  # NEW - External system integration
│   ├── __init__.py
│   ├── jupedsim_integration.py      # MOVE from core/
│   ├── jupedsim_setup.py            # MOVE from setup/
│   ├── state_queries.py             # RENAME simulation_state_queries.py
│   └── azure_llm.py                 # RENAME azure_llm_concordia.py
│
├── utils/                           # NEW - Pure utility functions
│   ├── __init__.py
│   ├── speed_conversion.py          # RENAME speed_utils.py
│   ├── action_parsing.py            # RENAME action_utils.py
│   ├── performance_monitor.py       # MOVE from core/
│   └── geometry_utils.py            # NEW - if needed
│
├── setup/                           # REFINED - Initial setup only
│   ├── __init__.py
│   ├── llm_setup.py
│   ├── output_manager.py
│   ├── station_layout_builder.py
│   ├── spawn_manager.py
│   ├── agent_manager.py             # Coordinates agent + spawn (orchestrator)
│   └── simulation_runner_factory.py # Main factory - uses all setup modules
│
└── run_station_concordia.py         # Entry point (~200 lines)
```

## Rationale for Structure

### 1. **agents/** - Agent Concerns

All agent-related code together:

- Definition (evacuation_agent.py)
- Instantiation (agent_builder.py)
- Configuration (agent_factory.py)

**Benefit:** Easy to understand agent lifecycle

### 2. **simulation/** - Core Orchestration

The heart of the simulation loop:

- Main runner (orchestrator)
- Decision processing (LLM calls, async)
- Event management (announcements, blocking)
- State tracking (exits, destinations, status)

**Benefit:** Clear simulation logic without infrastructure noise

### 3. **behaviors/** - Domain-Specific Behaviors

Complex agent behaviors that deserve their own modules:

- Helping (helper ↔ injured coordination)
- Messaging (agent-to-agent communication)

**Benefit:** Isolates complex domain logic

### 4. **translation/** - Concordia ↔ JuPedSim Bridge

All translation code in one place:

- Actions (NL → waypoints)
- Observations (positions → NL)
- Execution (apply actions to JuPedSim)

**Benefit:** Clear interface between two systems

### 5. **infrastructure/** - External System Wrappers

Integration with external libraries:

- JuPedSim wrapper and setup
- Azure LLM client
- State query helpers

**Benefit:** Isolates external dependencies

### 6. **utils/** - Pure Utility Functions

Stateless helper functions:

- Speed conversion
- Action parsing
- Performance monitoring
- Geometry calculations

**Benefit:** Reusable, testable utilities

### 7. **setup/** - One-Time Initialization

Setup code that runs once at start:

- LLM configuration
- Station layout construction
- Spawn position generation
- Factory that ties it all together

**Benefit:** Clear separation of setup vs runtime

## Migration Priority

### Phase 1: Split game_master.py (High Impact) ✅ NEXT

1. Extract `ActionTranslator` → `translation/action_translator.py`
2. Extract `ObservationGenerator` → `translation/observation_generator.py`
3. Delete `game_master.py` (no longer needed)

**Benefit:** Reduces 801 → 0 lines, creates 2 focused modules (~400 lines each)

### Phase 2: Split hybrid_simulation.py (Critical)

1. Extract message system → `behaviors/message_system.py` (~200 lines)
2. Extract helping coordinator → `behaviors/helping_coordinator.py` (~200 lines)
3. Extract decision processor → `simulation/decision_processor.py` (~400 lines)
4. Extract event manager → `simulation/event_manager.py` (~100 lines)
5. Extract action executor → `translation/action_executor.py` (~200 lines)
6. Keep orchestration → `simulation/runner.py` (~300 lines)

**Benefit:** Reduces 1557 → 300 lines, creates 6 focused modules

### Phase 3: Reorganize Directory Structure (Low Risk)

1. Create new directories
2. Move files to new locations
3. Update imports

**Benefit:** Better organization, clearer module purpose

### Phase 4: Refine Naming (Polish)

1. Rename files for consistency
2. Update documentation

**Benefit:** Professional, maintainable codebase

## Expected Final Statistics

```
Before Refactoring:
- core/ : 3,757 lines (7 large files)
- setup/: 819 lines (7 files)
Total: 4,576 lines

After Refactoring:
- agents/: ~500 lines (3 files)
- simulation/: ~1,000 lines (4 files, largest ~400)
- behaviors/: ~400 lines (2 files)
- translation/: ~1,000 lines (3 files, largest ~400)
- infrastructure/: ~800 lines (4 files)
- utils/: ~250 lines (4 files)
- setup/: ~800 lines (7 files)
Total: ~4,750 lines (similar, but organized)

Largest file: ~400 lines (vs 1557 today)
Average file: ~150 lines (vs 200+ today)
```

## Immediate Recommendations

### ✅ What We've Done Well So Far

1. Extracted small utilities (speed_utils, action_utils) - Good!
2. Created agent_builder - Good!
3. Created performance_monitor - Good!
4. Created simulation_state_queries - Good!

### ⚠️ What Needs Adjustment

1. **Don't extract more tiny utilities** - We've done enough
2. **Focus on splitting large files** - game_master (801), hybrid_simulation (1557)
3. **Consider directory reorganization** - After extracting large files

### 🎯 Next Steps

1. **Split game_master.py** into action_translator.py + observation_generator.py
2. **Extract message_system.py** from hybrid_simulation.py
3. **Extract helping_coordinator.py** from hybrid_simulation.py
4. **Extract decision_processor.py** from hybrid_simulation.py
5. **Then** consider directory reorganization

## Conclusion

**Current approach:** ✅ Good start, but need to shift focus
**Issue:** Extracting too many small utilities before tackling the large files
**Solution:** Focus on splitting the 801-line and 1557-line files into major components

The goal is not just fewer lines per file, but **clear separation of concerns** with modules organized by **what they do** (agents, simulation, translation) rather than **what they are** (core, setup, utils).
